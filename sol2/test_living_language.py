"""CPU contracts for the DMON-L0 frozen-language closed loop."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .config import Sol2Config
from .language_backbone import HuggingFaceFrozenBackbone, ToyFrozenLanguageBackbone
from .language_memory import (
    ContextErasureTask,
    LanguageMemoryTrainConfig,
    evaluate_context_erasure_controls,
    train_context_erasure,
)
from .language_organs import ContinuousLanguageOrgan
from .living_language import LivingLanguageSystem, graft_language_backbone
from .model import Sol2


LANGUAGE_CFG = Sol2Config(
    n_input=2,
    n_memory=4,
    n_compute=6,
    n_output=2,
    n_relay=4,
    hidden=12,
    n_dendrites=5,
    initial_active_dendrites=3,
    steps_per_token=2,
    vocab_size=11,
    batch_size=2,
    chunk_length=4,
    updates=10,
    warmup_updates=1,
    eval_every=5,
    eval_tokens=16,
    seed=17,
)


def build_system() -> tuple[LivingLanguageSystem, ContinuousLanguageOrgan]:
    torch.manual_seed(LANGUAGE_CFG.seed)
    organism = Sol2(LANGUAGE_CFG)
    with torch.random.fork_rng():
        torch.manual_seed(LANGUAGE_CFG.seed + 1)
        organ = ContinuousLanguageOrgan(
            LANGUAGE_CFG.n_input,
            LANGUAGE_CFG.hidden,
            language_width=16,
            n_queries=LANGUAGE_CFG.organ_queries,
            n_control_tokens=3,
            control_rank=4,
            bounded_operators=LANGUAGE_CFG.bounded_operators,
            operator_bound=LANGUAGE_CFG.operator_bound,
            value_gain=LANGUAGE_CFG.value_gain,
            attention_temperature=LANGUAGE_CFG.attention_temperature,
        )
    organism.attach_organ_module("language", organ)
    backbone = ToyFrozenLanguageBackbone(
        LANGUAGE_CFG.vocab_size, 16, seed=LANGUAGE_CFG.seed + 2
    )
    return LivingLanguageSystem(organism, backbone), organ


def test_new_language_graft_is_an_exact_noop() -> None:
    system, organ = build_system()
    context = torch.tensor([[1, 2, 3], [4, 5, 6]])
    baseline = system.backbone.controlled_logits(context)
    step = system.advance(
        context,
        system.initial_state(2, "cpu"),
        collect_health=True,
    )

    assert step.controls.shape == (2, 3, 16)
    assert torch.count_nonzero(step.controls) == 0
    assert torch.equal(step.logits, baseline[:, -1])
    assert step.health is not None
    assert step.health.logit_absmax == 0.0

    sensed = system.backbone.encode(context)[:, -1]
    expected_memory = torch.tanh(organ.sensor(sensed))
    memory_cell = system.organism.memory_idx[0]
    assert torch.allclose(step.state.hidden[:, memory_cell], expected_memory)


def test_language_loss_recruits_the_organ_but_not_the_backbone() -> None:
    system, organ = build_system()
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in system.backbone.named_parameters()
    }
    optimizer = torch.optim.SGD(system.organism.parameters(), lr=0.2)

    loss, _, controls = system.teacher_forced_loss(
        inputs, targets, system.initial_state(2, "cpu")
    )
    assert torch.count_nonzero(controls) == 0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert organ.output.decoder.weight.grad is not None
    assert float(organ.output.decoder.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in system.backbone.parameters())
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second_loss, _, second_controls = system.teacher_forced_loss(
        inputs, targets, system.initial_state(2, "cpu")
    )
    second_loss.backward()
    assert float(second_controls.detach().abs().sum()) > 0.0
    assert organ.sensor.weight.grad is not None
    assert float(organ.sensor.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in system.backbone.parameters())
    for name, parameter in system.backbone.named_parameters():
        assert torch.equal(parameter, frozen_before[name])


def test_vectorized_teacher_forcing_matches_causal_prefix_steps() -> None:
    system, organ = build_system()
    with torch.no_grad():
        nn.init.normal_(organ.output.decoder.weight, std=0.02)
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
    fast_loss, fast_state, fast_controls = system.teacher_forced_loss(
        inputs, targets, system.initial_state(2, "cpu")
    )
    slow_state = system.initial_state(2, "cpu")
    slow_logits = []
    slow_controls = []
    for position in range(inputs.shape[1]):
        step = system.advance(inputs[:, : position + 1], slow_state)
        slow_logits.append(step.logits)
        slow_controls.append(step.controls)
        slow_state = step.state
    slow_logits = torch.stack(slow_logits, dim=1)
    slow_loss = torch.nn.functional.cross_entropy(
        slow_logits.flatten(0, 1), targets.flatten()
    )
    assert torch.allclose(fast_loss, slow_loss, atol=1e-6)
    assert torch.allclose(fast_state.hidden, slow_state.hidden, atol=1e-6)
    assert torch.allclose(fast_controls, torch.stack(slow_controls, dim=1), atol=1e-6)


def test_observation_scoring_and_masked_loss_preserve_sequence_causality() -> None:
    system, organ = build_system()
    with torch.no_grad():
        nn.init.normal_(organ.output.decoder.weight, std=0.02)
    exposure = torch.tensor([[1, 2, 3], [3, 2, 1]])
    question = torch.tensor([[4, 5], [5, 4]])
    initial = system.initial_state(2, "cpu")
    exposed_state, exposure_controls = system.observe_sequence(exposure, initial)
    scored = system.score_next_after_sequence(question, exposed_state)
    assert exposure_controls.shape[:2] == exposure.shape
    assert scored.state.memory_cursor == exposure.shape[1] + question.shape[1]

    reference_state = initial
    for position in range(exposure.shape[1]):
        reference_state = system.advance(
            exposure[:, : position + 1], reference_state
        ).state
    reference = None
    for position in range(question.shape[1]):
        reference = system.advance(
            question[:, : position + 1], reference_state
        )
        reference_state = reference.state
    assert reference is not None
    assert torch.allclose(scored.logits, reference.logits, atol=1e-6)
    assert torch.allclose(scored.state.hidden, reference.state.hidden, atol=1e-6)

    targets = torch.tensor([[2, 3], [4, 5]])
    mask = torch.tensor([[False, True], [False, True]])
    masked_loss, _, _ = system.teacher_forced_loss(
        question, targets, exposed_state, loss_mask=mask
    )
    unmasked_loss, _, _ = system.teacher_forced_loss(
        question, targets, exposed_state
    )
    assert masked_loss.ndim == 0
    assert not torch.equal(masked_loss, unmasked_loss)


def test_generation_feeds_tokens_back_into_continuing_state() -> None:
    system, _ = build_system()
    prompt = torch.tensor([[1, 2, 3]])
    generated, state = system.generate(
        prompt,
        system.initial_state(1, "cpu"),
        max_new_tokens=3,
    )
    assert generated.shape == (1, 6)
    assert state.memory_cursor == 6


def test_context_erasure_does_not_erase_the_organism() -> None:
    system, _ = build_system()
    first = system.advance(
        torch.tensor([[1]]), system.initial_state(1, "cpu")
    )
    second = system.advance(torch.tensor([[2]]), first.state)
    reset = system.advance(
        torch.tensor([[2]]), system.initial_state(1, "cpu")
    )
    assert second.state.memory_cursor == 2
    assert reset.state.memory_cursor == 1
    assert not torch.equal(second.state.hidden, reset.state.hidden)


def test_specialized_organ_detaches_and_reattaches_exactly() -> None:
    system, organ = build_system()
    detached = system.organism.detach_organ("language")
    assert detached is organ
    system.organism.reattach_organ("language", detached)
    assert system.organism.attached_organs["language"] is organ


def test_huggingface_adapter_freezes_and_controls_an_exposed_head() -> None:
    class FakeCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8, vocab_size=9)
            self.embedding = nn.Embedding(9, 8)
            self.head = nn.Linear(8, 9, bias=False)

        def get_output_embeddings(self):
            return self.head

        def forward(self, *, input_ids, **_kwargs):
            features = torch.tanh(self.embedding(input_ids))
            return SimpleNamespace(
                hidden_states=(features,),
                logits=self.head(features),
            )

    adapter = HuggingFaceFrozenBackbone(FakeCausalLM())
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    features = adapter.encode(input_ids)
    baseline = adapter.controlled_logits_from_features(features)
    zeros = torch.zeros(2, 3, 8, requires_grad=True)
    assert torch.equal(
        baseline, adapter.controlled_logits_from_features(features, zeros)
    )
    controls = torch.randn(2, 3, 8, requires_grad=True)
    adapter.controlled_logits_from_features(features, controls).sum().backward()
    assert controls.grad is not None and float(controls.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert adapter.native_logit_error(input_ids) == 0.0


def test_graft_factory_is_deterministic_and_rng_neutral() -> None:
    torch.manual_seed(91)
    left = Sol2(LANGUAGE_CFG)
    torch.manual_seed(91)
    right = Sol2(LANGUAGE_CFG)
    left_backbone = ToyFrozenLanguageBackbone(11, 16, seed=8)
    right_backbone = ToyFrozenLanguageBackbone(11, 16, seed=8)
    rng_before = torch.random.get_rng_state().clone()
    left_system = graft_language_backbone(left, left_backbone, seed=55)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    right_system = graft_language_backbone(right, right_backbone, seed=55)
    left_parameters = left_system.organism.attached_organs["language"].state_dict()
    right_parameters = right_system.organism.attached_organs["language"].state_dict()
    assert left_parameters.keys() == right_parameters.keys()
    assert all(
        torch.equal(left_parameters[name], right_parameters[name])
        for name in left_parameters
    )


def test_mixed_backbone_and_organism_dtypes_are_bridged() -> None:
    system, _ = build_system()
    system.backbone.to(dtype=torch.float64)
    context = torch.tensor([[1, 2, 3]])
    step = system.advance(context, system.initial_state(1, "cpu"))
    assert step.state.hidden.dtype == torch.float32
    assert step.controls.dtype == torch.float32
    assert step.logits.dtype == torch.float64
    assert torch.equal(
        step.logits,
        system.backbone.controlled_logits(context)[:, -1],
    )


def test_context_erasure_learning_requires_persistent_internal_state() -> None:
    system, _ = build_system()
    task = ContextErasureTask(exposure_steps=4, distractor_steps=0)
    train_context_erasure(
        system,
        task,
        LanguageMemoryTrainConfig(
            updates=120,
            batch_size=16,
            lr=5e-3,
            seed=19,
            persistent_lifetimes=False,
        ),
    )
    controls = evaluate_context_erasure_controls(system, task, batch_size=32)
    assert controls["normal"].accuracy == 1.0
    assert controls["zero_control"].accuracy <= 0.5
    assert controls["reset"].accuracy <= 0.5
    assert controls["internal_lesion"].accuracy <= 0.5
    assert float(controls["normal"].loss) < float(controls["reset"].loss) - 0.1


def main() -> None:
    tests = [
        test_new_language_graft_is_an_exact_noop,
        test_language_loss_recruits_the_organ_but_not_the_backbone,
        test_vectorized_teacher_forcing_matches_causal_prefix_steps,
        test_observation_scoring_and_masked_loss_preserve_sequence_causality,
        test_generation_feeds_tokens_back_into_continuing_state,
        test_context_erasure_does_not_erase_the_organism,
        test_specialized_organ_detaches_and_reattaches_exactly,
        test_huggingface_adapter_freezes_and_controls_an_exposed_head,
        test_graft_factory_is_deterministic_and_rng_neutral,
        test_mixed_backbone_and_organism_dtypes_are_bridged,
        test_context_erasure_learning_requires_persistent_internal_state,
    ]
    print("DMON-L0 CPU contract gate")
    for test in tests:
        print(f"- {test.__name__}", flush=True)
        test()
    print("all DMON-L0 contracts hold")


if __name__ == "__main__":
    main()
