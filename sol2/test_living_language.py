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

    prefix_step = system.advance(
        context,
        system.initial_state(2, "cpu"),
        control_mode="prefix",
    )
    null_prefix = system.backbone.prefix_logits(
        context, torch.zeros_like(prefix_step.controls)
    )
    assert torch.equal(prefix_step.logits, null_prefix[:, -1])
    embedded = system.backbone.embedding(context).detach()
    anchors = embedded[:, :1].expand(-1, prefix_step.controls.shape[1], -1)
    anchored_features, _ = system.backbone.recurrent(
        torch.cat((anchors, embedded), dim=1)
    )
    anchored_logits = system.backbone.decoder(
        anchored_features[:, anchors.shape[1] :]
    )
    assert torch.equal(null_prefix, anchored_logits)

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


def test_prefix_control_backpropagates_through_every_frozen_layer() -> None:
    system, organ = build_system()
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = torch.tensor([5, 0])
    step = system.score_next_after_sequence(
        inputs,
        system.initial_state(2, "cpu"),
        control_mode="prefix",
    )
    loss = torch.nn.functional.cross_entropy(step.logits, targets)
    loss.backward()
    assert organ.output.decoder.weight.grad is not None
    assert float(organ.output.decoder.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in system.backbone.parameters())


def test_reference_centered_prefix_is_exact_noop_with_live_student_gradient() -> None:
    system, organ = build_system()
    with torch.no_grad():
        nn.init.normal_(organ.output.decoder.weight, std=0.02)
    inputs = torch.tensor([[1, 2, 3, 4]])
    state = system.initial_state(1, "cpu")
    raw = system.score_next_after_sequence(inputs, state, control_mode="prefix")
    centered = system.score_next_after_sequence(
        inputs,
        state,
        control_mode="prefix",
        control_reference=raw.controls,
    )
    expected = system.backbone.prefix_logits(
        inputs, torch.zeros_like(centered.controls)
    )[:, -1]
    assert torch.allclose(
        centered.controls, torch.zeros_like(centered.controls), atol=1e-9, rtol=0.0
    )
    assert torch.allclose(centered.logits, expected, atol=1e-7, rtol=0.0)
    zero_scaled = system.score_next_after_sequence(
        inputs,
        state,
        control_scale=0.0,
        control_mode="prefix",
        control_reference=raw.controls,
    )
    assert torch.equal(zero_scaled.controls, torch.zeros_like(zero_scaled.controls))
    torch.nn.functional.cross_entropy(centered.logits, torch.tensor([5])).backward()
    assert organ.output.decoder.weight.grad is not None
    assert float(organ.output.decoder.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in system.backbone.parameters())


def test_stream_memory_write_and_content_recall_are_differentiable() -> None:
    system, organ = build_system()
    exposure = torch.tensor([[1, 2, 3]])
    state, _ = system.observe_sequence(
        exposure, system.initial_state(1, "cpu")
    )
    written = state.hidden[:, system.organism.memory_idx]
    assert written.requires_grad
    written.square().mean().backward()
    assert organ.sensor.weight.grad is not None
    assert float(organ.sensor.weight.grad.abs().sum()) > 0.0

    system, organ = build_system()
    query = torch.randn(1, LANGUAGE_CFG.hidden)
    memory = torch.randn(1, LANGUAGE_CFG.n_memory, LANGUAGE_CFG.hidden)
    memory.requires_grad_()
    recalled, health = organ.recall(
        query, memory, valid_count=2, collect_health=True
    )
    changed_inactive = memory.detach().clone()
    changed_inactive[:, 2:] = 100.0
    repeated, _ = organ.recall(query, changed_inactive, valid_count=2)
    assert torch.allclose(recalled, repeated, atol=1e-6)
    assert float(recalled.detach().abs().max()) <= organ.recall_gain + 1e-6
    assert health is not None
    assert 1.0 <= health.effective_cells <= 2.0 + 1e-6
    recalled.square().mean().backward()
    assert memory.grad is not None
    assert float(memory.grad[:, :2].abs().sum()) > 0.0
    assert torch.count_nonzero(memory.grad[:, 2:]) == 0
    assert system.organism.graph.targets_read_only_from(
        system.organism.output_idx, system.organism.internal_idx
    )


def test_coherent_sparse_recall_preserves_coordinates_and_fifo_order() -> None:
    system, _ = build_system()
    hidden = torch.zeros(1, system.organism.n_cells, LANGUAGE_CFG.hidden)
    for index, cell in enumerate(system.organism.memory_idx):
        hidden[:, cell] = float(index + 1)
    ordered, count = system.organism._chronological_memory(hidden, memory_cursor=6)
    assert count == 4
    assert ordered[:, :, 0].tolist() == [[3.0, 4.0, 1.0, 2.0]]

    organ = ContinuousLanguageOrgan(
        2,
        LANGUAGE_CFG.hidden,
        language_width=16,
        n_queries=2,
        n_control_tokens=2,
        control_rank=2,
        bounded_operators=True,
        operator_bound=1.0,
        value_gain=0.85,
        attention_temperature=0.7,
        recall_gain=1.0,
        coherent_recall=True,
        recall_residual_gain=0.0,
        recall_top_k=1,
        recall_recency_bias=100.0,
    )
    query = torch.zeros(1, LANGUAGE_CFG.hidden)
    memory = torch.stack(
        [torch.full((LANGUAGE_CFG.hidden,), value) for value in (0.1, 0.2, 0.3)],
        dim=0,
    ).unsqueeze(0)
    recalled, health = organ.recall(query, memory, valid_count=3, collect_health=True)
    assert torch.allclose(recalled, torch.tanh(memory[:, -1]), atol=1e-6)
    assert health is not None and abs(health.effective_cells - 1.0) < 1e-6

    residual = ContinuousLanguageOrgan(
        2,
        LANGUAGE_CFG.hidden,
        language_width=16,
        n_queries=2,
        n_control_tokens=2,
        control_rank=2,
        bounded_operators=True,
        operator_bound=1.0,
        value_gain=0.85,
        attention_temperature=0.7,
        coherent_recall=True,
        recall_residual_gain=0.1,
    )
    live_memory = torch.randn(1, 3, LANGUAGE_CFG.hidden, requires_grad=True)
    live_query = torch.randn(1, LANGUAGE_CFG.hidden, requires_grad=True)
    residual.recall(live_query, live_memory, valid_count=3)[0].square().mean().backward()
    assert residual.recall_value.weight.grad is not None
    assert residual.recall_output.weight.grad is not None
    assert float(residual.recall_value.weight.grad.abs().sum()) > 0.0
    assert float(residual.recall_output.weight.grad.abs().sum()) > 0.0
    assert live_memory.grad is not None and float(live_memory.grad.abs().sum()) > 0.0


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

    exposure_features = system.backbone.encode(exposure)
    cached_state, cached_controls = system.observe_feature_sequence(
        exposure_features, initial
    )
    question_features = system.backbone.encode(question)
    cached_scored = system.score_next_from_features(question_features, cached_state)
    assert torch.allclose(cached_controls, exposure_controls, atol=1e-6)
    assert torch.allclose(cached_scored.logits, scored.logits, atol=1e-6)
    assert torch.allclose(cached_scored.state.hidden, scored.state.hidden, atol=1e-6)

    gated = system.score_next_after_sequence(
        question, exposed_state, write_memory=False
    )
    assert gated.state.memory_cursor == exposed_state.memory_cursor
    assert torch.equal(
        gated.state.hidden[:, system.organism.memory_idx],
        exposed_state.hidden[:, system.organism.memory_idx],
    )
    cached_gated = system.score_next_from_features(
        question_features, cached_state, write_memory=False
    )
    assert torch.allclose(cached_gated.logits, gated.logits, atol=1e-6)
    assert torch.allclose(cached_gated.state.hidden, gated.state.hidden, atol=1e-6)

    prefix_scored = system.score_next_after_sequence(
        question, exposed_state, write_memory=False, control_mode="prefix"
    )
    cached_prefix = system.score_next_from_features(
        question_features,
        cached_state,
        write_memory=False,
        control_mode="prefix",
        input_ids=question,
    )
    assert torch.allclose(cached_prefix.logits, prefix_scored.logits, atol=1e-6)
    assert torch.allclose(
        cached_prefix.state.hidden, prefix_scored.state.hidden, atol=1e-6
    )

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

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, input_ids=None, inputs_embeds=None, **_kwargs):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("provide exactly one token representation")
            embedded = (
                self.embedding(input_ids)
                if inputs_embeds is None
                else inputs_embeds
            )
            features = torch.tanh(embedded.cumsum(dim=1))
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
    prefix = torch.randn(2, 3, 8, requires_grad=True)
    prefix_logits = adapter.prefix_logits(input_ids, prefix)
    assert prefix_logits.shape == (2, 3, 9)
    prefix_logits.sum().backward()
    assert prefix.grad is not None and float(prefix.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert adapter.native_logit_error(input_ids) == 0.0


def test_huggingface_adapter_accepts_nested_text_config() -> None:
    class FakeMultimodalCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=8, vocab_size=9)
            )
            self.embedding = nn.Embedding(9, 8)
            self.head = nn.Linear(8, 9, bias=False)

        def get_output_embeddings(self):
            return self.head

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, input_ids=None, inputs_embeds=None, **_kwargs):
            embedded = (
                self.embedding(input_ids)
                if inputs_embeds is None
                else inputs_embeds
            )
            features = torch.tanh(embedded.cumsum(dim=1))
            return SimpleNamespace(
                hidden_states=(features,),
                logits=self.head(features),
            )

    adapter = HuggingFaceFrozenBackbone(FakeMultimodalCausalLM())
    assert adapter.width == 8
    assert adapter.vocab_size == 9
    assert adapter.native_logit_error(torch.tensor([[1, 2, 3]])) == 0.0


def test_graft_factory_is_deterministic_and_rng_neutral() -> None:
    torch.manual_seed(91)
    left = Sol2(LANGUAGE_CFG)
    torch.manual_seed(91)
    right = Sol2(LANGUAGE_CFG)
    left_backbone = ToyFrozenLanguageBackbone(11, 16, seed=8)
    right_backbone = ToyFrozenLanguageBackbone(11, 16, seed=8)
    rng_before = torch.random.get_rng_state().clone()
    left_system = graft_language_backbone(
        left, left_backbone, control_gain=64.0, recall_gain=1.0, seed=55
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    right_system = graft_language_backbone(
        right, right_backbone, control_gain=64.0, recall_gain=1.0, seed=55
    )
    left_parameters = left_system.organism.attached_organs["language"].state_dict()
    right_parameters = right_system.organism.attached_organs["language"].state_dict()
    assert left_parameters.keys() == right_parameters.keys()
    assert all(
        torch.equal(left_parameters[name], right_parameters[name])
        for name in left_parameters
    )
    assert left_system.organism.attached_organs["language"].recall_gain == 1.0
    assert left_system.organism.attached_organs["language"].control_gain == 64.0


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
        test_prefix_control_backpropagates_through_every_frozen_layer,
        test_reference_centered_prefix_is_exact_noop_with_live_student_gradient,
        test_stream_memory_write_and_content_recall_are_differentiable,
        test_coherent_sparse_recall_preserves_coordinates_and_fifo_order,
        test_vectorized_teacher_forcing_matches_causal_prefix_steps,
        test_observation_scoring_and_masked_loss_preserve_sequence_causality,
        test_generation_feeds_tokens_back_into_continuing_state,
        test_context_erasure_does_not_erase_the_organism,
        test_specialized_organ_detaches_and_reattaches_exactly,
        test_huggingface_adapter_freezes_and_controls_an_exposed_head,
        test_huggingface_adapter_accepts_nested_text_config,
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
