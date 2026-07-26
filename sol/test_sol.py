"""Behavioral and credit-path tests for the first SOL character organism."""

from __future__ import annotations

import json
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
from .evaluate import evaluate_state_ablations, evaluate_warmup_sweep
from .model import SolConfig, SparseAxonField
from .promote import promote_best_checkpoint
from .report import compare_runs, load_run, markdown_report
from .serve import LiveOrganism
from .stream import CharacterVocabulary, ContinuousCharStream
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


def test_delayed_reward_changes_only_tagged_fast_synapses() -> None:
    model = _model(reward_gain=0.0, fast_weight_decay=0.0)
    state = model.initial_state(1)
    state.edge_eligibility[:, 3, 1] = 1.0
    _, next_state, _ = model.tick(
        state, token=None, reward=torch.ones(1)
    )
    expected = model.cfg.fast_plasticity_gain
    assert next_state.fast_weight[0, 3, 1].item() == pytest.approx(expected)
    untagged = next_state.fast_weight.clone()
    untagged[:, 3, 1] = 0.0
    assert torch.count_nonzero(untagged).item() == 0


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


def test_reward_plasticity_does_not_mint_energy() -> None:
    model = _model(stimulation_gain=0.0, reward_gain=1.0)
    state = model.initial_state(1)
    state.eligibility.fill_(1.0)
    state.edge_eligibility.fill_(1.0)
    before = state.energy.clone()
    _, after, _ = model.tick(
        state, token=None, reward=torch.ones(1)
    )
    assert torch.all(after.energy <= before)


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
        _model(vocab_size=len(vocabulary)),
        text,
        vocabulary,
        batch_size=3,
        chunk_length=4,
        learning_rate=4e-3,
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
    assert torch.equal(resumed.state.edge_eligibility, trainer.state.edge_eligibility)
    assert torch.equal(resumed.state.fast_weight, trainer.state.fast_weight)


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
    del payload["trainer"]["field_state"]["fast_weight"]
    torch.save(payload, checkpoint)
    resumed, _ = load_checkpoint(checkpoint, text)
    assert torch.count_nonzero(resumed.state.edge_eligibility).item() == 0
    assert torch.count_nonzero(resumed.state.fast_weight).item() == 0


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
    assert result["metrics"]["fastWeight"] >= 0
    snapshot = organism.snapshot()
    assert snapshot["metrics"]["edgeEligibility"] != 0
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
        runs.append(run)
    destination = tmp_path / "live.pt"
    manifest = promote_best_checkpoint(runs, destination)
    assert manifest["source_run"] == str(runs[1])
    assert manifest["best_bpc"] == 2.4
    assert destination.exists()
    assert destination.with_suffix(".json").exists()
    promoted = LiveOrganism(destination)
    assert promoted.loaded.updates == trainer.updates


def test_heldout_evaluation_reports_state_ablations() -> None:
    text = "abcabcabcabcabcabcabcabc" * 4
    vocabulary = CharacterVocabulary.from_text(text)
    model = _model(vocab_size=len(vocabulary))
    metrics = evaluate_state_ablations(
        model, vocabulary, text, tokens=12, warmup=4
    )
    assert set(metrics) == {"persistent", "reset_each_token", "shuffled_cells"}
    assert all(value["tokens"] == 12 for value in metrics.values())
    assert all(value["bits_per_character"] > 0 for value in metrics.values())
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
