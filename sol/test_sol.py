"""Behavioral and credit-path tests for the first SOL character organism."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from .baselines import (
    CausalCharacterTransformer,
    CharacterGRU,
    evaluate_gru,
    evaluate_transformer,
    match_gru_hidden_size,
    match_transformer_hidden_size,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .evaluate import (
    _shuffle_cell_state,
    evaluate_state_ablations,
    evaluate_warmup_sweep,
)
from .model import SolConfig, SparseAxonField
from .promote import promote_best_checkpoint
from .report import compare_runs, load_run, markdown_report
from .schedule import (
    cosine_decay_learning_rate,
    set_optimizer_learning_rate,
)
from .serve import LiveOrganism
from .stability import (
    summarize_exploratory_survival,
    summarize_stability,
)
from .stream import CharacterVocabulary, ContinuousCharStream
from .structure import StructuralConfig, apply_structural_phase
from .topology import analyze_topology
from .train import ContinuousTrainer


def _model(**overrides) -> SparseAxonField:
    torch.manual_seed(4)
    cfg = SolConfig(
        vocab_size=3,
        cells=12,
        channels=12,
        dendrites=4,
        sensory_cells=2,
        output_cells=2,
        message_steps=2,
        topology_seed=7,
    )
    return SparseAxonField(replace(cfg, **overrides))


def test_topology_is_sparse_directed_and_output_reachable() -> None:
    model = _model()
    assert model.sources.shape == (12, 4)
    assert all(
        int(source) in model.sources[int(target)].tolist()
        for target, source in zip(model.output_indices, model.sensory_indices)
    )
    edges = {
        (int(source), target)
        for target, row in enumerate(model.sources)
        for source in row
    }
    assert any((target, source) not in edges for source, target in edges)
    metrics = analyze_topology(
        model.sources, model.sensory_indices, model.output_indices
    )
    assert metrics.directed_edges == model.cfg.cells * model.cfg.dendrites
    assert metrics.output_reachable_fraction == 1.0
    assert metrics.reachable_fraction == 1.0
    assert metrics.mean_output_distance is not None


def test_stream_windows_are_adjacent_not_reset() -> None:
    vocabulary = CharacterVocabulary.from_text("abc")
    stream = ContinuousCharStream("abcabcabcabc", vocabulary, batch_size=2)
    first_x, first_y = stream.next(2)
    second_x, _ = stream.next(2)
    assert torch.equal(first_y[:, -1], second_x[:, 0])


def test_learning_rate_decay_is_absolute_bounded_and_optional() -> None:
    base = 3e-3
    assert cosine_decay_learning_rate(base, 1) == base
    assert cosine_decay_learning_rate(base, 400, 400, 800, 0.1) == base
    midpoint = cosine_decay_learning_rate(base, 600, 400, 800, 0.1)
    assert midpoint == pytest.approx(base * 0.55)
    assert cosine_decay_learning_rate(
        base, 800, 400, 800, 0.1
    ) == pytest.approx(base * 0.1)
    assert cosine_decay_learning_rate(
        base, 1200, 400, 800, 0.1
    ) == pytest.approx(base * 0.1)

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=base)
    set_optimizer_learning_rate(optimizer, midpoint)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(midpoint)


@pytest.mark.parametrize(
    ("start", "end", "ratio"),
    [(-1, 0, 0.1), (4, 3, 0.1), (0, 4, -0.1), (0, 4, 1.1)],
)
def test_learning_rate_decay_rejects_invalid_policy(
    start: int,
    end: int,
    ratio: float,
) -> None:
    with pytest.raises(ValueError):
        cosine_decay_learning_rate(3e-3, 1, start, end, ratio)


def test_history_changes_prediction_for_the_same_character() -> None:
    model = _model()
    fresh = model.initial_state(1)
    _, remembered, _ = model.forward_sequence(torch.tensor([[0, 1]]), fresh)
    same = torch.tensor([2])
    logits_fresh, _, _ = model.tick(model.initial_state(1), same)
    logits_remembered, _, _ = model.tick(remembered, same)
    assert not torch.allclose(logits_fresh, logits_remembered)


def test_energy_depletes_without_external_input() -> None:
    model = _model()
    state = model.initial_state(2)
    before = state.energy.mean().item()
    for _ in range(8):
        _, state, _ = model.tick(state, token=None)
    assert state.energy.mean().item() < before
    assert state.stimulation.max().item() == 0.0


def test_unfed_cells_reach_quiescence_in_finite_time() -> None:
    model = _model(
        message_steps=1,
        energy_start=0.2,
        basal_cost=0.01,
        activity_cost=0.0,
        energy_transport_rate=0.0,
        quiescence_energy=0.01,
        full_activity_energy=0.05,
    )
    state = model.initial_state(1)
    for _ in range(24):
        _, state, diagnostics = model.tick(state, token=None)
    assert diagnostics["quiescent_fraction"].item() == 1
    assert diagnostics["mean_viability"].item() == 0
    frozen = state.hidden.clone()
    _, state, _ = model.tick(state, token=None)
    assert torch.equal(state.hidden, frozen)


def test_directed_energy_transport_never_mints_and_follows_named_axons() -> None:
    model = _model(energy_transport_rate=0.5)
    control = _model(energy_transport_rate=0.0)
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    torch.manual_seed(9)
    energy = 0.2 + 0.6 * torch.rand(3, model.cfg.cells)
    flow = torch.rand(
        3,
        model.cfg.cells,
        model.cfg.dendrites,
    )
    probe_flow = torch.rand(3, model.cfg.cells)
    transported, drift = model._transport_energy(
        energy,
        flow,
        probe_flow,
    )
    assert torch.all(transported >= 0)
    assert torch.all(transported <= 1)
    assert torch.allclose(
        transported.sum(dim=1),
        energy.sum(dim=1),
        atol=2e-6,
    )
    assert torch.allclose(drift, torch.zeros_like(drift), atol=2e-6)

    target, slot = next(
        (target, slot)
        for target in range(model.cfg.cells)
        for slot in range(model.cfg.dendrites)
        if int(model.sources[target, slot]) != target
    )
    source = int(model.sources[target, slot])
    isolated_energy = torch.zeros(1, model.cfg.cells)
    isolated_energy[0, source] = 0.8
    isolated_flow = torch.zeros(
        1,
        model.cfg.cells,
        model.cfg.dendrites,
    )
    isolated_flow[0, target, slot] = 1.0
    moved, isolated_drift = model._transport_energy(
        isolated_energy,
        isolated_flow,
        torch.zeros(1, model.cfg.cells),
    )
    assert moved[0, target] > 0
    assert moved[0, source] < isolated_energy[0, source]
    assert moved.sum().item() == pytest.approx(
        isolated_energy.sum().item(),
        abs=1e-6,
    )
    assert isolated_drift.item() == pytest.approx(0.0, abs=1e-6)


def test_directed_maintenance_flow_funds_silent_named_targets() -> None:
    model = _model(
        energy_transport_rate=0.5,
        energy_maintenance_flow=0.1,
    )
    control = _model(
        energy_transport_rate=0.5,
        energy_maintenance_flow=0.0,
    )
    target, slot = next(
        (target, slot)
        for target in range(model.cfg.cells)
        for slot in range(model.cfg.dendrites)
        if int(model.sources[target, slot]) != target
    )
    source = int(model.sources[target, slot])
    energy = torch.zeros(1, model.cfg.cells)
    energy[0, source] = 0.8
    silent_edges = torch.zeros(
        1,
        model.cfg.cells,
        model.cfg.dendrites,
    )
    silent_probes = torch.zeros(1, model.cfg.cells)

    maintained, drift = model._transport_energy(
        energy,
        silent_edges,
        silent_probes,
    )
    unmaintained, control_drift = control._transport_energy(
        energy,
        silent_edges,
        silent_probes,
    )

    assert maintained[0, target] > 0
    assert maintained[0, source] < energy[0, source]
    assert torch.equal(unmaintained, energy)
    assert maintained.sum().item() == pytest.approx(
        energy.sum().item(),
        abs=1e-6,
    )
    assert drift.item() == pytest.approx(0.0, abs=1e-6)
    assert control_drift.item() == pytest.approx(0.0, abs=1e-6)


def test_recurrent_stimulation_cannot_mint_metabolic_energy() -> None:
    model = _model(
        basal_cost=0.0,
        activity_cost=0.0,
        stimulation_gain=1.0,
        energy_transport_rate=0.5,
    )
    state = model.initial_state(2)
    state.energy.fill_(0.4)
    state.stimulation.fill_(1.0)
    before = state.energy.sum(dim=1)
    _, after, diagnostics = model.tick(state, token=None)
    assert torch.all(after.energy.sum(dim=1) <= before + 1e-6)
    assert torch.count_nonzero(diagnostics["energy_input"]).item() == 0
    assert torch.all(diagnostics["energy_transport_drift"] <= 1e-6)


def test_external_input_is_the_only_energy_source() -> None:
    model = _model(
        basal_cost=0.0,
        activity_cost=0.0,
        stimulation_gain=0.5,
        energy_transport_rate=0.25,
    )
    state = model.initial_state(2)
    state.energy.fill_(0.2)
    before = state.energy.sum(dim=1)
    _, after, diagnostics = model.tick(state, torch.tensor([0, 1]))
    expected = (
        before
        + diagnostics["energy_input"]
        + diagnostics["energy_transport_drift"]
    )
    assert torch.all(diagnostics["energy_input"] > 0)
    assert torch.allclose(after.energy.sum(dim=1), expected, atol=2e-6)


def test_energy_quiescence_is_reversible_from_new_input() -> None:
    model = _model(
        message_steps=1,
        basal_cost=0.0,
        activity_cost=0.0,
        stimulation_gain=1.0,
        energy_transport_rate=0.0,
        quiescence_energy=0.1,
        full_activity_energy=0.2,
    )
    state = model.initial_state(1)
    state.energy.zero_()
    hidden = state.hidden.clone()
    _, quiet, quiet_diagnostics = model.tick(state, token=None)
    assert torch.equal(quiet.hidden, hidden)
    assert quiet_diagnostics["mean_viability"].item() == 0
    assert quiet_diagnostics["quiescent_fraction"].item() == 1

    _, recovered, recovered_diagnostics = model.tick(
        quiet,
        torch.tensor([0]),
    )
    sensory = model.sensory_indices
    assert torch.all(recovered.energy[:, sensory] > 0.2)
    assert not torch.equal(recovered.hidden[:, sensory], hidden[:, sensory])
    assert recovered_diagnostics["mean_viability"].item() > 0
    assert recovered_diagnostics["quiescent_fraction"].item() < 1


def test_delayed_reward_acts_through_event_eligibility() -> None:
    model = _model()
    base = model.initial_state(1)
    tagged = base.clone()
    tagged.eligibility[:, 3] = 1.0
    reward = torch.ones(1)
    _, untagged_next, _ = model.tick(base, token=None, reward=reward)
    _, tagged_next, _ = model.tick(tagged, token=None, reward=reward)
    assert not torch.allclose(
        untagged_next.hidden[:, 3], tagged_next.hidden[:, 3]
    )


def test_backward_credit_moves_from_targets_to_named_sources() -> None:
    model = _model(backward_credit_gain=1.0)
    control = _model(backward_credit_gain=0.0)
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    credit = torch.zeros(1, model.cfg.cells)
    coefficient = torch.zeros(
        1,
        model.cfg.cells,
        model.cfg.dendrites,
    )
    target, slot = next(
        (target, slot)
        for target in range(model.cfg.cells)
        for slot in range(model.cfg.dendrites)
        if int(model.sources[target, slot]) != target
    )
    source = int(model.sources[target, slot])
    credit[0, target] = 1.0
    coefficient[0, target, slot] = 0.5

    transported = model._transport_backward_credit(
        credit,
        coefficient,
    )
    expected = 0.5 / math.sqrt(model.cfg.dendrites)
    assert transported[0, source].item() == pytest.approx(expected)
    transported[0, source] = 0
    assert torch.count_nonzero(transported).item() == 0


def test_output_reward_launches_persistent_backward_credit() -> None:
    model = _model(
        reward_gain=0.0,
        backward_credit_gain=1.0,
        backward_credit_decay=0.5,
        message_steps=1,
    )
    state = model.initial_state(1)
    _, after_reward, diagnostics = model.tick(
        state,
        token=None,
        reward=torch.ones(1),
    )
    assert after_reward.backward_credit.abs().sum().item() > 0
    assert diagnostics["mean_backward_credit"].item() > 0

    _, after_quiet, _ = model.tick(
        after_reward,
        token=None,
        reward=torch.zeros(1),
    )
    assert after_quiet.backward_credit.abs().sum().item() > 0


def test_backward_credit_meets_event_specific_cell_memory() -> None:
    model = _model(
        reward_gain=0.0,
        backward_credit_gain=1.0,
        backward_credit_decay=0.0,
        message_steps=1,
    )
    base = model.initial_state(1)
    base.backward_credit[:, 3] = 1.0
    tagged = base.clone()
    tagged.eligibility[:, 3] = 1.0

    _, untagged_next, _ = model.tick(
        base,
        token=None,
        reward=torch.zeros(1),
    )
    _, tagged_next, _ = model.tick(
        tagged,
        token=None,
        reward=torch.zeros(1),
    )
    assert not torch.allclose(
        untagged_next.hidden[:, 3],
        tagged_next.hidden[:, 3],
    )


def test_zero_reward_cannot_create_backward_credit() -> None:
    model = _model(
        reward_gain=0.0,
        backward_credit_gain=1.0,
    )
    state = model.initial_state(1)
    _, next_state, _ = model.tick(
        state,
        token=None,
        reward=torch.zeros(1),
    )
    assert torch.count_nonzero(next_state.backward_credit).item() == 0


def test_delayed_reward_changes_only_tagged_fast_synapses() -> None:
    model = _model(reward_gain=0.0, fast_weight_decay=0.0)
    state = model.initial_state(1)
    state.edge_eligibility[:, 3, 1] = 1.0
    _, next_state, _ = model.tick(
        state, token=None, reward=torch.ones(1)
    )
    expected = model.cfg.fast_weight_limit * torch.tanh(
        torch.tensor(
            model.cfg.fast_plasticity_gain
            / model.cfg.fast_weight_limit
        )
    ).item()
    assert next_state.fast_weight[0, 3, 1].item() == pytest.approx(expected)
    untagged = next_state.fast_weight.clone()
    untagged[:, 3, 1] = 0.0
    assert torch.count_nonzero(untagged).item() == 0


def test_learned_edge_tags_are_competitive_per_target() -> None:
    model = _model(edge_eligibility_decay=0.0)
    state = model.initial_state(2)
    _, next_state, _ = model.tick(state, torch.tensor([0, 1]))
    tag_sums = next_state.edge_eligibility.sum(dim=2)
    assert torch.allclose(tag_sums, torch.zeros_like(tag_sums), atol=1e-6)
    assert next_state.edge_eligibility.abs().sum().item() > 0


def test_fast_synapses_are_reward_dependent_bounded_and_differentiable() -> None:
    model = _model(reward_gain=0.0)
    no_reward = model.initial_state(1)
    no_reward.edge_eligibility.fill_(1.0)
    _, unchanged, _ = model.tick(
        no_reward, token=None, reward=torch.zeros(1)
    )
    assert torch.count_nonzero(unchanged.fast_weight).item() == 0

    bounded = model.initial_state(1)
    bounded.edge_eligibility.fill_(1.0)
    for _ in range(40):
        _, bounded, _ = model.tick(
            bounded, token=None, reward=torch.full((1,), 100.0)
        )
    assert bounded.fast_weight.abs().max().item() <= model.cfg.fast_weight_limit

    gradient_state = model.initial_state(1)
    gradient_state.fast_weight.requires_grad_(True)
    logits, _, _ = model.tick(gradient_state, torch.tensor([1]))
    logits.square().sum().backward()
    assert gradient_state.fast_weight.grad is not None
    assert gradient_state.fast_weight.grad.abs().sum().item() > 0


def test_pending_reward_is_consumed_exactly_once() -> None:
    model = _model(reward_gain=0.0)
    state = model.initial_state(1)
    state.edge_eligibility.fill_(1.0)
    state.reward.fill_(1.0)
    _, after_reward, _ = model.tick(state, token=None)
    _, after_quiet, _ = model.tick(after_reward, token=None)
    assert torch.count_nonzero(after_reward.fast_weight).item() > 0
    expected_quiet = model.cfg.fast_weight_limit * torch.tanh(
        model.cfg.fast_weight_decay
        * after_reward.fast_weight
        / model.cfg.fast_weight_limit
    )
    assert torch.allclose(
        after_quiet.fast_weight,
        expected_quiet,
    )
    assert torch.count_nonzero(after_quiet.reward).item() == 0


def test_surprise_reward_is_signed_around_persistent_expectation() -> None:
    model = _model(reward_baseline_decay=0.5)
    state = model.initial_state(1)
    baseline = state.reward_baseline.clone()

    neutral = model.observe_surprise(state.clone(), baseline)
    assert neutral.reward.item() == pytest.approx(0.0)

    better = model.observe_surprise(state.clone(), baseline - 0.5)
    worse = model.observe_surprise(state.clone(), baseline + 0.5)
    assert better.reward.item() > 0
    assert worse.reward.item() < 0
    assert better.reward_baseline.item() == pytest.approx(
        baseline.item() - 0.25
    )


def test_reward_plasticity_does_not_mint_energy() -> None:
    model = _model(stimulation_gain=0.0, reward_gain=1.0)
    state = model.initial_state(1)
    state.eligibility.fill_(1.0)
    state.edge_eligibility.fill_(1.0)
    before = state.energy.clone()
    _, after, diagnostics = model.tick(
        state, token=None, reward=torch.ones(1)
    )
    assert after.energy.sum().item() <= before.sum().item() + 1e-6
    assert diagnostics["energy_input"].item() == 0


def test_structural_probe_measures_a_causal_candidate_effect() -> None:
    model = _model(structural_probe_gain=0.03)
    state = model.initial_state(2)
    _, state, diagnostics = model.tick(state, torch.tensor([0, 1]))
    assert diagnostics["probe_flow"].mean().item() > 0
    assert state.probe_eligibility.abs().sum().item() > 0

    state.reward.fill_(1.0)
    _, _, rewarded = model.tick(state, torch.tensor([1, 2]))
    assert rewarded["structural_probe_evidence"].abs().sum().item() > 0
    assert all(
        int(candidate) not in model.sources[target].tolist()
        for target, candidate in enumerate(model.probe_sources)
    )


def test_structural_probe_mask_gates_only_selected_exploratory_traffic() -> None:
    model = _model(structural_probe_gain=0.03, message_steps=1)
    state = model.initial_state(2)
    token = torch.tensor([0, 1])
    _, _, full = model.tick(state.clone(), token)
    mask = torch.ones(model.cfg.cells)
    target = 3
    mask[target] = 0
    _, masked_state, masked = model.tick(
        state.clone(),
        token,
        structural_probe_mask=mask,
    )

    assert masked["probe_flow"][target].item() == 0
    keep = torch.arange(model.cfg.cells) != target
    assert torch.allclose(
        masked["probe_flow"][keep], full["probe_flow"][keep]
    )
    assert masked_state.probe_eligibility[:, target].abs().sum() == 0
    assert masked_state.probe_eligibility[:, keep].abs().sum() > 0


def test_structural_probe_fitness_uses_global_loss_gradient() -> None:
    model = _model(structural_probe_gain=0.03)
    tokens = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])
    targets = torch.tensor([[1, 2, 0, 1], [2, 0, 1, 2]])
    logits, _, trace = model.forward_sequence(
        tokens, targets=targets, retain_credit=True
    )
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    fitness = trace.probe_fitness()
    assert fitness.shape == (tokens.shape[1], model.cfg.cells)
    assert torch.isfinite(fitness).all()
    assert fitness.abs().sum().item() > 0


def test_structural_phase_rewires_one_slot_without_losing_integrity() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=2,
        replacements_per_phase=1,
        credit_decay=0.0,
        min_edge_age=0,
        growth_cost=0.02,
        min_endpoint_energy=0.0,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
            energy_start=1.0,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    trainer.step()
    model = trainer.model
    model.structural_edge_credit.fill_(1.0)
    model.structural_probe_credit.zero_()
    model.structural_edge_age.fill_(10)

    chosen: tuple[int, int, int] | None = None
    for target in range(model.cfg.cells):
        candidate = int(model.probe_sources[target].item())
        for slot in range(model.cfg.dendrites):
            proposed = model.sources.clone()
            proposed[target, slot] = candidate
            topology = analyze_topology(
                proposed, model.sensory_indices, model.output_indices
            )
            if (
                topology.reachable_fraction == 1.0
                and topology.output_reachable_fraction == 1.0
            ):
                chosen = (target, slot, candidate)
                break
        if chosen is not None:
            break
    assert chosen is not None
    target, slot, candidate = chosen
    model.structural_edge_credit[target, slot] = 0.0
    model.structural_probe_credit[target] = 1.0

    sources_before = model.sources.clone()
    weights_before = model.edge_weight.detach().clone()
    energy_before = trainer.state.energy.sum().item()
    edge_moment = trainer.optimizer.state[model.edge_weight]["exp_avg"]
    assert edge_moment.abs().sum().item() > 0

    update = apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=2
    )
    assert update.rewired_edges == 1
    assert update.total_rewires == 1
    assert int(model.sources[target, slot].item()) == candidate
    unchanged = torch.ones_like(model.sources, dtype=torch.bool)
    unchanged[target, slot] = False
    assert torch.equal(model.sources[unchanged], sources_before[unchanged])
    expected_graft = math.atanh(
        model.cfg.structural_probe_gain * model.cfg.dendrites
    )
    assert model.edge_weight[target, slot].item() == pytest.approx(
        expected_graft
    )
    assert torch.equal(
        model.edge_weight.detach()[unchanged], weights_before[unchanged]
    )
    assert edge_moment[target, slot].item() == 0
    assert torch.count_nonzero(
        trainer.state.edge_eligibility[:, target, slot]
    ).item() == 0
    assert torch.count_nonzero(
        trainer.state.fast_weight[:, target, slot]
    ).item() == 0
    assert energy_before - trainer.state.energy.sum().item() == pytest.approx(
        trainer.stream.batch_size * config.growth_cost,
        abs=2e-6,
    )
    topology = analyze_topology(
        model.sources, model.sensory_indices, model.output_indices
    )
    assert topology.reachable_fraction == 1.0
    assert topology.output_reachable_fraction == 1.0
    assert len(set(model.sources[target].tolist())) == model.cfg.dendrites
    assert int(model.probe_sources[target]) not in model.sources[target].tolist()


def test_probes_only_control_rotates_without_rewiring() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=StructuralConfig(
            enabled=True,
            allow_rewiring=False,
            interval=1,
            warmup_updates=0,
            credit_decay=0.0,
            min_edge_age=0,
        ),
    )
    sources = trainer.model.sources.clone()
    probes = trainer.model.probe_sources.clone()
    metrics = trainer.step()
    assert torch.equal(trainer.model.sources, sources)
    assert not torch.equal(trainer.model.probe_sources, probes)
    assert metrics.rewired_edges == 0
    assert metrics.total_rewires == 0


def test_structural_candidate_requires_consecutive_confirmations() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=0,
        replacements_per_phase=1,
        confirmation_phases=3,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    model = trainer.model
    model.structural_edge_credit.fill_(0.0)
    model.structural_probe_credit.fill_(1.0)
    model.structural_edge_age.fill_(10)
    sources = model.sources.clone()
    probes = model.probe_sources.clone()

    first = apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=1
    )
    assert first.rewired_edges == 0
    assert torch.equal(model.sources, sources)
    assert torch.equal(model.probe_sources, probes)
    assert torch.all(model.structural_probe_confirmations == 1)

    model.structural_probe_credit.fill_(1.0)
    second = apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=2
    )
    assert second.rewired_edges == 0
    assert torch.equal(model.sources, sources)
    assert torch.equal(model.probe_sources, probes)
    assert torch.all(model.structural_probe_confirmations == 2)

    model.structural_probe_credit.fill_(1.0)
    third = apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=3
    )
    assert third.rewired_edges == 1
    assert not torch.equal(model.sources, sources)
    assert not torch.equal(model.probe_sources, probes)
    assert torch.count_nonzero(
        model.structural_probe_confirmations
    ).item() == 0


def test_structural_confirmation_streak_resets_on_negative_phase() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=0,
        confirmation_phases=3,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    model = trainer.model
    model.structural_edge_credit.fill_(0.0)
    model.structural_edge_age.fill_(10)
    sources = model.sources.clone()
    probes = model.probe_sources.clone()

    model.structural_probe_credit.fill_(1.0)
    apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=1
    )
    assert torch.all(model.structural_probe_confirmations == 1)

    model.structural_probe_credit.fill_(-1.0)
    apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=2
    )
    assert torch.count_nonzero(
        model.structural_probe_confirmations
    ).item() == 0

    model.structural_probe_credit.fill_(1.0)
    third = apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=3
    )
    assert third.rewired_edges == 0
    assert torch.equal(model.sources, sources)
    assert not torch.equal(model.probe_sources, probes)


def test_global_fitness_can_veto_locally_credited_rewire() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=0,
        confirmation_phases=1,
        require_global_fitness=True,
        global_fitness_margin=0.0,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    model = trainer.model
    model.structural_edge_credit.fill_(0.0)
    model.structural_probe_credit.fill_(1.0)
    model.structural_probe_fitness.fill_(-1.0)
    model.structural_edge_age.fill_(10)
    sources = model.sources.clone()

    rejected = apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=1
    )
    assert rejected.rewired_edges == 0
    assert torch.equal(model.sources, sources)

    model.structural_probe_credit.fill_(1.0)
    model.structural_probe_fitness.fill_(1.0)
    accepted = apply_structural_phase(
        model, trainer.state, trainer.optimizer, config, update=2
    )
    assert accepted.rewired_edges == 1
    assert not torch.equal(model.sources, sources)


def _start_forced_probation(
    trainer: ContinuousTrainer,
    config: StructuralConfig,
    update: int,
) -> tuple[int, int, int]:
    model = trainer.model
    model.structural_edge_credit.fill_(1.0)
    model.structural_probe_credit.zero_()
    model.structural_edge_age.fill_(10)
    for target in range(model.cfg.cells):
        candidate = int(model.probe_sources[target].item())
        for slot in range(model.cfg.dendrites):
            proposed = model.sources.clone()
            proposed[target, slot] = candidate
            topology = analyze_topology(
                proposed, model.sensory_indices, model.output_indices
            )
            if (
                topology.reachable_fraction == 1.0
                and topology.output_reachable_fraction == 1.0
            ):
                model.structural_edge_credit[target, slot] = 0.0
                model.structural_probe_credit[target] = 1.0
                result = apply_structural_phase(
                    model,
                    trainer.state,
                    trainer.optimizer,
                    config,
                    update,
                    trainer.structural_probation,
                )
                assert result.probation_started
                return target, slot, candidate
    raise AssertionError("no viable probation candidate found")


def test_negative_probation_restores_exact_graft_slot() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=2,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.02,
        min_endpoint_energy=0.0,
        probation_updates=2,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    trainer.step()
    model = trainer.model
    sources = model.sources.clone()
    weights = model.edge_weight.detach().clone()
    biases = model.edge_bias.detach().clone()
    edge_eligibility = trainer.state.edge_eligibility.clone()
    fast_weight = trainer.state.fast_weight.clone()
    weight_moment = trainer.optimizer.state[model.edge_weight][
        "exp_avg"
    ].clone()
    bias_moment = trainer.optimizer.state[model.edge_bias][
        "exp_avg"
    ].clone()
    energy = trainer.state.energy.sum().item()

    target, slot, candidate = _start_forced_probation(
        trainer, config, update=2
    )
    backed_credit = trainer.structural_probation.edge_credit.clone()
    backed_age = trainer.structural_probation.edge_age.clone()
    assert int(model.sources[target, slot]) == candidate
    with torch.no_grad():
        model.token_embedding.weight.add_(1.0)
        model.edge_weight[target, slot].add_(3.0)
        model.edge_bias[target, slot].add_(2.0)
    body_after_experience = model.token_embedding.weight.detach().clone()
    trainer.optimizer.state[model.edge_weight]["exp_avg"][
        target, slot
    ] = 9.0
    trainer.optimizer.state[model.edge_bias]["exp_avg"][
        target, slot
    ] = 8.0
    trainer.state.edge_eligibility[:, target, slot] = 7.0
    trainer.state.fast_weight[:, target, slot] = 6.0
    trainer.structural_probation.observe(-0.5)
    committed = trainer.structural_probation.resolve(
        model, trainer.state, trainer.optimizer, margin=0.0
    )

    assert not committed
    assert torch.equal(model.sources, sources)
    assert model.edge_weight[target, slot] == weights[target, slot]
    assert model.edge_bias[target, slot] == biases[target, slot]
    assert (
        model.structural_edge_credit[target, slot]
        == backed_credit
    )
    assert model.structural_edge_age[target, slot] == backed_age
    assert torch.equal(
        trainer.state.edge_eligibility[:, target, slot],
        edge_eligibility[:, target, slot],
    )
    assert torch.equal(
        trainer.state.fast_weight[:, target, slot],
        fast_weight[:, target, slot],
    )
    assert (
        trainer.optimizer.state[model.edge_weight]["exp_avg"][
            target, slot
        ]
        == weight_moment[target, slot]
    )
    assert (
        trainer.optimizer.state[model.edge_bias]["exp_avg"][
            target, slot
        ]
        == bias_moment[target, slot]
    )
    assert torch.equal(
        model.token_embedding.weight, body_after_experience
    )
    assert energy - trainer.state.energy.sum().item() == pytest.approx(
        trainer.stream.batch_size * config.growth_cost,
        abs=2e-6,
    )
    assert trainer.structural_probation.total_rolled_back == 1


def test_positive_probation_commits_adapted_graft() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=0,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
        probation_updates=2,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    sources = trainer.model.sources.clone()
    target, slot, candidate = _start_forced_probation(
        trainer, config, update=1
    )
    with torch.no_grad():
        trainer.model.edge_weight[target, slot].add_(0.5)
    adapted = trainer.model.edge_weight[target, slot].detach().clone()
    trainer.structural_probation.observe(0.25)
    committed = trainer.structural_probation.resolve(
        trainer.model,
        trainer.state,
        trainer.optimizer,
        margin=0.0,
    )
    assert committed
    assert int(trainer.model.sources[target, slot]) == candidate
    assert trainer.model.edge_weight[target, slot] == adapted
    assert not torch.equal(trainer.model.sources, sources)
    assert trainer.structural_probation.total_committed == 1


def test_probation_scores_against_pre_graft_developmental_baseline() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=0,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
        probation_updates=2,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    sources = trainer.model.sources.clone()
    _start_forced_probation(trainer, config, update=1)
    trainer.structural_probation.baseline_advantage = 0.2
    trainer.structural_probation.observe(0.1)
    assert trainer.structural_probation.mean_advantage == pytest.approx(-0.1)
    assert not trainer.structural_probation.resolve(
        trainer.model,
        trainer.state,
        trainer.optimizer,
        margin=0.0,
    )
    assert torch.equal(trainer.model.sources, sources)


def test_probes_only_uses_virtual_probation_without_mutation() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        allow_rewiring=False,
        interval=1,
        warmup_updates=0,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
        probation_updates=2,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    sources = trainer.model.sources.clone()
    _start_forced_probation(trainer, config, update=1)
    assert trainer.structural_probation.active
    assert trainer.structural_probation.virtual
    assert torch.equal(trainer.model.sources, sources)
    trainer.structural_probation.observe(1.0)
    assert not trainer.structural_probation.resolve(
        trainer.model,
        trainer.state,
        trainer.optimizer,
        margin=0.0,
    )
    assert torch.equal(trainer.model.sources, sources)
    assert trainer.structural_probation.total_rolled_back == 0


def test_exploratory_probation_uses_abba_traffic_before_grafting() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=2,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.02,
        min_endpoint_energy=0.0,
        probation_updates=4,
        probation_exploratory_traffic=True,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
            energy_start=1.0,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    trainer.step()
    model = trainer.model
    sources = model.sources.clone()
    energy = trainer.state.energy.sum().item()
    target, slot, candidate = _start_forced_probation(
        trainer, config, update=2
    )

    assert torch.equal(model.sources, sources)
    assert trainer.state.energy.sum().item() == energy
    arms = []
    for reward in (0.4, 0.1, 0.2, 0.5):
        mask, exposed = (
            trainer.structural_probation.exploratory_probe_mask(model)
        )
        assert mask is not None
        assert mask[target].item() == float(exposed)
        arms.append(exposed)
        trainer.structural_probation.observe(reward, exposed)
    assert arms == [True, False, False, True]
    assert (
        trainer.structural_probation.candidate_observations
        == trainer.structural_probation.incumbent_observations
        == 2
    )
    assert trainer.structural_probation.mean_advantage == pytest.approx(0.3)

    committed = trainer.structural_probation.resolve(
        model,
        trainer.state,
        trainer.optimizer,
        margin=0.0,
        growth_cost=config.growth_cost,
        resolved_update=6,
    )
    assert committed
    assert int(model.sources[target, slot]) == candidate
    assert model.total_rewires.item() == 1
    assert energy - trainer.state.energy.sum().item() == pytest.approx(
        trainer.stream.batch_size * config.growth_cost,
        abs=2e-6,
    )
    trial = trainer.structural_probation.trial_history[-1]
    assert trial["mode"] == "exploratory_traffic"
    assert trial["outcome"] == "committed"
    assert trial["started_update"] == 2
    assert trial["resolved_update"] == 6
    assert trial["target"] == target
    assert trial["candidate_source"] == candidate
    assert trial["candidate_observations"] == 2
    assert trial["incumbent_observations"] == 2
    assert trial["decision_advantage"] == pytest.approx(0.3)
    assert trial["body_energy_after"] < trial["body_energy_before"]


def test_exploratory_rejection_preserves_live_anatomy_and_energy() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=2,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.02,
        min_endpoint_energy=0.0,
        probation_updates=2,
        probation_exploratory_traffic=True,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
            energy_start=1.0,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    trainer.step()
    sources = trainer.model.sources.clone()
    energy = trainer.state.energy.clone()
    _start_forced_probation(trainer, config, update=2)
    trainer.structural_probation.observe(0.1, True)
    trainer.structural_probation.observe(0.2, False)

    assert not trainer.structural_probation.resolve(
        trainer.model,
        trainer.state,
        trainer.optimizer,
        margin=0.0,
        growth_cost=config.growth_cost,
    )
    assert torch.equal(trainer.model.sources, sources)
    assert torch.equal(trainer.state.energy, energy)
    assert trainer.structural_probation.total_rejected == 1
    assert trainer.structural_probation.total_rolled_back == 0


def test_exploratory_commit_rechecks_endpoint_energy() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=2,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.02,
        min_endpoint_energy=0.1,
        probation_updates=2,
        probation_exploratory_traffic=True,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
            energy_start=1.0,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    trainer.step()
    sources = trainer.model.sources.clone()
    target, _, candidate = _start_forced_probation(
        trainer, config, update=2
    )
    trainer.structural_probation.observe(0.4, True)
    trainer.structural_probation.observe(0.1, False)
    trainer.state.energy[:, target] = 0
    trainer.state.energy[:, candidate] = 0

    assert not trainer.structural_probation.resolve(
        trainer.model,
        trainer.state,
        trainer.optimizer,
        margin=0.0,
        growth_cost=config.growth_cost,
        min_endpoint_energy=config.min_endpoint_energy,
    )
    assert torch.equal(trainer.model.sources, sources)
    assert trainer.structural_probation.total_rejected == 1


def test_exploratory_traffic_keeps_shared_body_learning_in_both_arms() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=2,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
        probation_updates=4,
        probation_exploratory_traffic=True,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    trainer.step()
    _start_forced_probation(trainer, config, update=2)
    before = trainer.model.token_embedding.weight.detach().clone()
    candidate_metrics = trainer.step()
    after_candidate = (
        trainer.model.token_embedding.weight.detach().clone()
    )
    incumbent_metrics = trainer.step()
    after_incumbent = (
        trainer.model.token_embedding.weight.detach().clone()
    )

    assert candidate_metrics.probation_candidate_exposed
    assert not incumbent_metrics.probation_candidate_exposed
    assert not torch.equal(after_candidate, before)
    assert not torch.equal(after_incumbent, after_candidate)
    assert trainer.structural_probation.active
    assert trainer.structural_probation.candidate_observations == 1
    assert trainer.structural_probation.incumbent_observations == 1


def test_harmful_structural_probe_cannot_rewire() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=0,
        min_edge_age=0,
        credit_margin=0.0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        structural_config=config,
    )
    trainer.model.structural_edge_credit.fill_(-1.0)
    trainer.model.structural_probe_credit.fill_(-0.5)
    trainer.model.structural_edge_age.fill_(10)
    sources = trainer.model.sources.clone()
    update = apply_structural_phase(
        trainer.model,
        trainer.state,
        trainer.optimizer,
        config,
        update=1,
    )
    assert update.rewired_edges == 0
    assert torch.equal(trainer.model.sources, sources)


def test_backward_credit_reaches_cells_and_synapses() -> None:
    model = _model()
    tokens = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])
    targets = torch.tensor([[1, 2, 0, 1], [2, 0, 1, 2]])
    logits, _, trace = model.forward_sequence(
        tokens, targets=targets, retain_credit=True
    )
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    assert model.edge_weight.grad is not None
    assert model.edge_weight.grad.abs().sum().item() > 0
    credit = trace.cell_credit()
    assert credit.shape == (tokens.shape[1], model.cfg.cells)
    assert credit[0].sum().item() > 0


def test_tiny_continuous_corpus_loss_falls() -> None:
    torch.manual_seed(9)
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    model = _model(vocab_size=len(vocabulary))
    trainer = ContinuousTrainer(
        model,
        text,
        vocabulary,
        batch_size=4,
        chunk_length=6,
        learning_rate=6e-3,
    )
    losses = [trainer.step().loss for _ in range(60)]
    assert sum(losses[-8:]) / 8 < 0.55 * (sum(losses[:8]) / 8)
    assert trainer.state.eligibility.abs().sum().item() > 0


def test_checkpoint_resume_preserves_the_next_update(tmp_path: Path) -> None:
    torch.manual_seed(12)
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
            reward_gain=0.0,
            backward_credit_gain=0.25,
        ),
        text,
        vocabulary,
        batch_size=3,
        chunk_length=4,
        learning_rate=4e-3,
        structural_config=StructuralConfig(
            enabled=True,
            allow_rewiring=False,
            interval=2,
            warmup_updates=0,
            credit_decay=0.5,
            min_edge_age=0,
        ),
    )
    for _ in range(3):
        trainer.step()
    checkpoint = save_checkpoint(tmp_path / "resume.pt", trainer, {"tag": "test"})
    expected = trainer.step()
    resumed, metadata = load_checkpoint(checkpoint, text)
    actual = resumed.step()
    assert metadata == {"tag": "test"}
    assert resumed.updates == trainer.updates
    assert resumed.stream.position == trainer.stream.position
    assert actual.loss == expected.loss
    assert torch.equal(resumed.state.hidden, trainer.state.hidden)
    assert torch.equal(
        resumed.state.backward_credit,
        trainer.state.backward_credit,
    )
    assert torch.equal(resumed.state.edge_eligibility, trainer.state.edge_eligibility)
    assert torch.equal(resumed.state.probe_eligibility, trainer.state.probe_eligibility)
    assert torch.equal(resumed.state.fast_weight, trainer.state.fast_weight)
    assert torch.equal(resumed.state.reward_baseline, trainer.state.reward_baseline)
    assert torch.equal(resumed.model.probe_sources, trainer.model.probe_sources)
    assert torch.equal(
        resumed.model.structural_edge_credit,
        trainer.model.structural_edge_credit,
    )
    assert torch.equal(
        resumed.model.structural_probe_credit,
        trainer.model.structural_probe_credit,
    )
    assert torch.equal(
        resumed.model.structural_probe_fitness,
        trainer.model.structural_probe_fitness,
    )
    assert torch.equal(
        resumed.model.structural_probe_confirmations,
        trainer.model.structural_probe_confirmations,
    )
    assert torch.equal(
        resumed.model.structural_edge_age,
        trainer.model.structural_edge_age,
    )
    assert resumed.structural_config == trainer.structural_config


def test_checkpoint_resume_preserves_active_probation(tmp_path: Path) -> None:
    torch.manual_seed(21)
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=2,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
        probation_updates=2,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        learning_rate=4e-3,
        structural_config=config,
    )
    trainer.step()
    _start_forced_probation(trainer, config, update=2)
    trainer.updates = 2
    checkpoint = save_checkpoint(
        tmp_path / "probation.pt", trainer, {"tag": "probation"}
    )

    expected_first = trainer.step()
    expected_second = trainer.step()
    resumed, metadata = load_checkpoint(checkpoint, text)
    actual_first = resumed.step()
    actual_second = resumed.step()

    assert metadata == {"tag": "probation"}
    assert actual_first.loss == expected_first.loss
    assert actual_second.loss == expected_second.loss
    assert resumed.updates == trainer.updates
    assert torch.equal(resumed.model.sources, trainer.model.sources)
    assert torch.equal(
        resumed.model.edge_weight, trainer.model.edge_weight
    )
    assert torch.equal(resumed.state.hidden, trainer.state.hidden)
    assert (
        resumed.structural_probation.total_committed
        == trainer.structural_probation.total_committed
    )
    assert (
        resumed.structural_probation.total_rolled_back
        == trainer.structural_probation.total_rolled_back
    )
    assert (
        resumed.structural_probation.active
        == trainer.structural_probation.active
    )
    assert (
        resumed.prequential_advantage_ema
        == trainer.prequential_advantage_ema
    )


def test_checkpoint_resume_preserves_exploratory_traffic_arm(
    tmp_path: Path,
) -> None:
    torch.manual_seed(22)
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    config = StructuralConfig(
        enabled=True,
        interval=1,
        warmup_updates=10,
        confirmation_phases=1,
        credit_decay=0.0,
        credit_margin=0.0,
        min_edge_age=0,
        growth_cost=0.0,
        min_endpoint_energy=0.0,
        probation_updates=4,
        probation_exploratory_traffic=True,
    )
    trainer = ContinuousTrainer(
        _model(
            vocab_size=len(vocabulary),
            structural_probe_gain=0.03,
        ),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        learning_rate=4e-3,
        structural_config=config,
    )
    trainer.step()
    _start_forced_probation(trainer, config, update=10)
    trainer.updates = 10
    first = trainer.step()
    assert first.probation_candidate_exposed
    checkpoint = save_checkpoint(
        tmp_path / "exploratory.pt",
        trainer,
        {"tag": "exploratory"},
    )

    expected = [trainer.step() for _ in range(3)]
    resumed, metadata = load_checkpoint(checkpoint, text)
    actual = [resumed.step() for _ in range(3)]

    assert metadata == {"tag": "exploratory"}
    assert [row.loss for row in actual] == [
        row.loss for row in expected
    ]
    assert [row.probation_candidate_exposed for row in actual] == [
        row.probation_candidate_exposed for row in expected
    ]
    assert torch.equal(resumed.model.sources, trainer.model.sources)
    assert torch.equal(resumed.state.hidden, trainer.state.hidden)
    assert (
        resumed.structural_probation.total_committed
        == trainer.structural_probation.total_committed
    )
    assert (
        resumed.structural_probation.total_rejected
        == trainer.structural_probation.total_rejected
    )
    assert (
        resumed.structural_probation.last_mean_advantage
        == trainer.structural_probation.last_mean_advantage
    )
    assert (
        resumed.structural_probation.trial_history
        == trainer.structural_probation.trial_history
    )


def test_pre_plasticity_checkpoint_state_is_upgraded(tmp_path: Path) -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    trainer = ContinuousTrainer(
        _model(vocab_size=len(vocabulary)),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
    )
    checkpoint = save_checkpoint(tmp_path / "old.pt", trainer)
    payload = torch.load(checkpoint, weights_only=False)
    del payload["trainer"]["field_state"]["edge_eligibility"]
    del payload["trainer"]["field_state"]["probe_eligibility"]
    del payload["trainer"]["field_state"]["fast_weight"]
    del payload["trainer"]["field_state"]["reward_baseline"]
    del payload["trainer"]["field_state"]["backward_credit"]
    del payload["trainer"]["model_config"]["structural_probe_gain"]
    del payload["trainer"]["model_config"]["energy_transport_rate"]
    del payload["trainer"]["model_config"]["energy_maintenance_flow"]
    del payload["trainer"]["model_config"]["quiescence_energy"]
    del payload["trainer"]["model_config"]["full_activity_energy"]
    del payload["trainer"]["structural_config"]
    del payload["trainer"]["structural_probation"]
    del payload["trainer"]["prequential_advantage_ema"]
    for name in (
        "probe_sources",
        "structural_edge_credit",
        "structural_probe_credit",
        "structural_probe_fitness",
        "structural_probe_confirmations",
        "structural_edge_age",
        "total_rewires",
    ):
        del payload["trainer"]["model"][name]
    torch.save(payload, checkpoint)
    resumed, _ = load_checkpoint(checkpoint, text)
    assert torch.count_nonzero(resumed.state.edge_eligibility).item() == 0
    assert torch.count_nonzero(resumed.state.probe_eligibility).item() == 0
    assert torch.count_nonzero(resumed.state.fast_weight).item() == 0
    assert torch.count_nonzero(resumed.state.backward_credit).item() == 0
    assert torch.count_nonzero(
        resumed.model.structural_edge_credit
    ).item() == 0
    assert resumed.model.total_rewires.item() == 0
    assert resumed.model.cfg.energy_transport_rate == pytest.approx(0.50)
    assert resumed.model.cfg.energy_maintenance_flow == pytest.approx(0.0)
    assert resumed.model.cfg.quiescence_energy == pytest.approx(0.01)
    assert resumed.model.cfg.full_activity_energy == pytest.approx(0.05)
    assert torch.allclose(
        resumed.state.reward_baseline,
        torch.full_like(
            resumed.state.reward_baseline,
            torch.log(torch.tensor(float(len(vocabulary)))).item(),
        ),
    )


def test_frozen_connectome_survives_training_and_resume(tmp_path: Path) -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    trainer = ContinuousTrainer(
        _model(vocab_size=len(vocabulary)),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
        frozen_parameters=("edge_weight", "edge_bias"),
    )
    before_weight = trainer.model.edge_weight.detach().clone()
    before_bias = trainer.model.edge_bias.detach().clone()
    trainer.step()
    assert torch.equal(trainer.model.edge_weight, before_weight)
    assert torch.equal(trainer.model.edge_bias, before_bias)
    checkpoint = save_checkpoint(tmp_path / "frozen.pt", trainer)
    resumed, _ = load_checkpoint(checkpoint, text)
    assert resumed.frozen_parameters == ("edge_bias", "edge_weight")
    assert not resumed.model.edge_weight.requires_grad
    assert not resumed.model.edge_bias.requires_grad


def test_live_checkpoint_bridge_generates_with_real_credit(
    tmp_path: Path,
) -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    trainer = ContinuousTrainer(
        _model(vocab_size=len(vocabulary)),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
    )
    trainer.step()
    checkpoint = save_checkpoint(tmp_path / "live.pt", trainer)
    organism = LiveOrganism(checkpoint)
    result = organism.generate("abca", 6, seed=3)
    assert result["mode"] == "live-checkpoint"
    assert len(result["output"]) == 6
    assert result["checkpoint"]["updates"] == 1
    assert result["metrics"]["cellCredit"] > 0
    assert result["metrics"]["edgeCredit"] > 0
    assert result["metrics"]["energy"] >= 0
    assert 0 <= result["metrics"]["viability"] <= 1
    assert 0 <= result["metrics"]["quiescentFraction"] <= 1
    assert result["metrics"]["energyInput"] >= 0
    assert result["metrics"]["energySpent"] >= 0
    assert result["metrics"]["energyTransportDrift"] <= 1e-6
    assert result["metrics"]["fastWeight"] >= 0
    assert 0 <= result["metrics"]["fastSaturation"] <= 1
    snapshot = organism.snapshot()
    assert snapshot["metrics"]["edgeEligibility"] != 0
    assert 0 <= snapshot["metrics"]["fastSaturation"] <= 1
    assert snapshot["metrics"]["rewardBaseline"] > 0
    assert "backwardCredit" in snapshot["metrics"]
    assert 0 <= snapshot["metrics"]["viability"] <= 1
    assert 0 <= snapshot["metrics"]["quiescentFraction"] <= 1
    assert snapshot["metrics"]["structuralRewires"] == 0
    assert "probeSources" in snapshot["topology"]
    assert len(snapshot["topology"]["fastWeights"]) == trainer.model.cfg.cells


def test_checkpoint_promotion_validates_and_selects_lowest_bpc(
    tmp_path: Path,
) -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    trainer = ContinuousTrainer(
        _model(vocab_size=len(vocabulary)),
        text,
        vocabulary,
        batch_size=2,
        chunk_length=4,
    )
    trainer.step()
    runs = []
    for name, bpc in (("first", 2.8), ("second", 2.4)):
        run = tmp_path / name
        run.mkdir()
        save_checkpoint(run / "best.pt", trainer, {"best_bpc": bpc})
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "model": "sol",
                    "best_bpc": bpc,
                    "updates": trainer.updates,
                    "parameters": sum(
                        parameter.numel()
                        for parameter in trainer.model.parameters()
                    ),
                    "evaluation": {
                        "persistent": {"bits_per_character": bpc}
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "kind": "evaluation",
                    "model": "sol",
                    "update": trainer.updates,
                    "ablations": {
                        "persistent": {
                            "bits_per_character": bpc
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runs.append(run)

    unstable = tmp_path / "unstable"
    unstable.mkdir()
    save_checkpoint(
        unstable / "best.pt", trainer, {"best_bpc": 2.2}
    )
    (unstable / "summary.json").write_text(
        json.dumps(
            {
                "model": "sol",
                "best_bpc": 2.2,
                "updates": 2,
                "parameters": sum(
                    parameter.numel()
                    for parameter in trainer.model.parameters()
                ),
                "evaluation": {
                    "persistent": {"bits_per_character": 4.0}
                },
            }
        ),
        encoding="utf-8",
    )
    (unstable / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "evaluation",
                    "model": "sol",
                    "update": update,
                    "ablations": {
                        "persistent": {
                            "bits_per_character": bpc
                        }
                    },
                }
            )
            for update, bpc in ((1, 2.2), (2, 4.0))
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "live.pt"
    manifest = promote_best_checkpoint(runs + [unstable], destination)
    assert manifest["source_run"] == str(runs[1])
    assert manifest["best_bpc"] == 2.4
    assert manifest["stability"]["stable"]
    assert manifest["rejected_candidates"][0]["run"] == str(unstable)
    assert not manifest["rejected_candidates"][0]["stability"]["stable"]
    assert destination.exists()
    assert destination.with_suffix(".json").exists()
    promoted = LiveOrganism(destination)
    assert promoted.loaded.updates == trainer.updates


def test_heldout_evaluation_reports_state_ablations() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    model = _model(vocab_size=len(vocabulary))
    sources = model.sources.clone()
    probes = model.probe_sources.clone()
    metrics = evaluate_state_ablations(
        model, vocabulary, text, tokens=12, warmup=4
    )
    assert set(metrics) == {
        "persistent",
        "reset_each_token",
        "shuffled_cells",
        "zero_fast_efficacy",
        "birth_topology",
    }
    assert torch.equal(model.sources, sources)
    assert torch.equal(model.probe_sources, probes)
    assert all(value["tokens"] == 12 for value in metrics.values())
    assert all(value["bits_per_character"] > 0 for value in metrics.values())
    assert all("mean_fast_weight" in value for value in metrics.values())
    assert all("mean_edge_eligibility" in value for value in metrics.values())
    assert all("fast_weight_saturation" in value for value in metrics.values())
    assert all("mean_probe_flow" in value for value in metrics.values())
    assert all(
        "mean_backward_credit" in value
        for value in metrics.values()
    )
    assert all("mean_viability" in value for value in metrics.values())
    assert all("quiescent_fraction" in value for value in metrics.values())
    assert all("energy_input" in value for value in metrics.values())
    assert all("energy_spent" in value for value in metrics.values())
    assert all(
        value["energy_transport_drift"] <= 1e-6
        for value in metrics.values()
    )
    assert all("total_rewires" in value for value in metrics.values())
    assert metrics["zero_fast_efficacy"]["mean_fast_weight"] == 0
    sweep = evaluate_warmup_sweep(
        model,
        vocabulary,
        text,
        [0, 2, 4],
        tokens=12,
        score_start=8,
    )
    assert set(sweep) == {"0", "2", "4"}
    assert all(value["tokens"] == 12 for value in sweep.values())


def test_stability_ignores_early_learning_but_rejects_post_best_collapse() -> None:
    stable = summarize_stability(
        [(1, 5.0), (2, 3.0), (3, 2.5), (4, 2.7)],
        max_regression_bpc=0.5,
    )
    assert stable["stable"]
    assert stable["best_update"] == 3
    assert stable["final_regression_bpc"] == pytest.approx(0.2)

    collapsed = summarize_stability(
        [(1, 5.0), (2, 2.4), (3, 3.1), (3, 3.2)],
        max_regression_bpc=0.5,
    )
    assert not collapsed["stable"]
    assert collapsed["evaluations"] == 3
    assert collapsed["worst_regression_bpc"] == pytest.approx(0.8)


def test_exploratory_survival_aligns_each_trial_with_the_living_body() -> None:
    trials = [
        {
            "mode": "exploratory_traffic",
            "outcome": "committed",
            "virtual": False,
            "started_update": 100,
            "resolved_update": 140,
            "decision_advantage": 0.04,
        },
        {
            "mode": "exploratory_traffic",
            "outcome": "rejected",
            "virtual": False,
            "started_update": 220,
            "resolved_update": 260,
            "decision_advantage": -0.01,
        },
        {
            "mode": "exploratory_traffic",
            "outcome": "virtual",
            "virtual": True,
            "started_update": 300,
            "resolved_update": 340,
            "decision_advantage": 0.02,
        },
    ]
    summary = summarize_exploratory_survival(
        [
            (50, 3.0),
            (100, 2.8),
            (150, 2.9),
            (200, 2.7),
            (250, 2.6),
            (300, 3.4),
        ],
        trials,
        max_regression_bpc=0.5,
    )
    assert summary["trials"] == 2
    assert summary["committed"] == 1
    assert summary["rejected"] == 1
    assert summary["evaluated_trials"] == 2
    assert summary["survived_trials"] == 1
    assert summary["unstable_trials"] == 1
    first, second = summary["trial_history"]
    assert first["baseline_update"] == 100
    assert first["first_post_update"] == 150
    assert first["survived"]
    assert second["baseline_update"] == 200
    assert second["worst_regression_bpc"] == pytest.approx(0.7)
    assert not second["survived"]


def test_cell_shuffle_keeps_target_owned_edge_state_aligned() -> None:
    model = _model()
    state = model.initial_state(1)
    markers = torch.arange(model.cfg.cells, dtype=state.fast_weight.dtype)
    state.edge_eligibility[:, :, 0] = markers
    state.fast_weight[:, :, 0] = markers + 100
    state.probe_eligibility[:] = markers
    state.backward_credit[:] = markers
    permutation = torch.arange(model.cfg.cells - 1, -1, -1)
    shuffled = _shuffle_cell_state(state, permutation)
    assert torch.equal(
        shuffled.edge_eligibility[:, :, 0], markers[permutation].unsqueeze(0)
    )
    assert torch.equal(
        shuffled.fast_weight[:, :, 0],
        (markers[permutation] + 100).unsqueeze(0),
    )
    assert torch.equal(
        shuffled.probe_eligibility, markers[permutation].unsqueeze(0)
    )
    assert torch.equal(
        shuffled.backward_credit, markers[permutation].unsqueeze(0)
    )


def test_gru_control_matches_budget_and_scores_stream() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    target = sum(parameter.numel() for parameter in _model().parameters())
    hidden = match_gru_hidden_size(len(vocabulary), target)
    model = CharacterGRU(len(vocabulary), hidden)
    nearest_error = abs(model.parameter_count() - target)
    assert nearest_error / target < 0.08
    metrics = evaluate_gru(
        model, vocabulary.encode(text), tokens=12, warmup=4
    )
    assert metrics["tokens"] == 12
    assert metrics["bits_per_character"] > 0


def test_transformer_control_is_causal_matched_and_stateful() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    target = sum(parameter.numel() for parameter in _model().parameters())
    hidden = match_transformer_hidden_size(
        len(vocabulary), target, layers=1, heads=2, context=16, maximum=64
    )
    model = CausalCharacterTransformer(
        len(vocabulary), hidden, layers=1, heads=2, context=16
    )
    assert abs(model.parameter_count() - target) / target < 0.20
    state = model.initial_state(1, "cpu")
    logits, state = model(vocabulary.encode("ab").view(1, -1), state)
    assert logits.shape == (1, 2, len(vocabulary))
    assert state.shape == (1, 2)
    metrics = evaluate_transformer(
        model, vocabulary.encode(text), tokens=12, warmup=4
    )
    assert metrics["tokens"] == 12
    assert metrics["bits_per_character"] > 0


def test_report_guards_budgets_and_measures_state_penalties(
    tmp_path: Path,
) -> None:
    sol_dir = tmp_path / "sol"
    gru_dir = tmp_path / "gru"
    sol_dir.mkdir()
    gru_dir.mkdir()
    (sol_dir / "summary.json").write_text(
        """{
          "model": "sol",
          "parameters": 1000,
          "updates": 20,
          "best_bpc": 2.4,
          "evaluation": {
            "persistent": {"bits_per_character": 2.5},
            "reset_each_token": {"bits_per_character": 4.0},
            "shuffled_cells": {"bits_per_character": 3.3}
          }
        }""",
        encoding="utf-8",
    )
    (gru_dir / "summary.json").write_text(
        """{
          "model": "gru",
          "parameters": 1020,
          "updates": 20,
          "best_bpc": 2.2,
          "evaluation": {"bits_per_character": 2.3}
        }""",
        encoding="utf-8",
    )
    comparison = compare_runs(
        [load_run("sol", sol_dir), load_run("gru", gru_dir)]
    )
    assert comparison["winner"] == "gru"
    assert comparison["rows"][0]["reset_penalty_bpc"] == 1.5
    assert "| sol |" in markdown_report(comparison)

    mismatched = json.loads((gru_dir / "summary.json").read_text())
    mismatched["parameters"] = 2000
    (gru_dir / "summary.json").write_text(
        json.dumps(mismatched), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="parameter ratio"):
        compare_runs(
            [load_run("sol", sol_dir), load_run("gru", gru_dir)]
        )
