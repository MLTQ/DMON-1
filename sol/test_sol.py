"""Behavioral and credit-path tests for the first SOL character organism."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from .model import SolConfig, SparseAxonField
from .stream import CharacterVocabulary, ContinuousCharStream
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
