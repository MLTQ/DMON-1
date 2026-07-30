"""Contract + learning smoke for the rebuilt grok organism."""

from __future__ import annotations

import math
import sys

import torch
from torch.nn import functional as F

from .config import TrainConfig
from .evaluate import evaluate_with_ablations
from .model import StreamingCreature
from .stream import CharacterVocabulary, CharCorpus, ContinuousCharStream


def _cfg(**kwargs) -> TrainConfig:
    base = dict(
        n_cells=40,
        n_input=4,
        n_output=4,
        n_mirror=8,
        n_dendrites=6,
        hidden=32,
        steps_per_token=2,
        batch_size=2,
        chunk_length=16,
        use_attention=True,
        output_error_credit_gain=0.5,
        reward_gain=0.25,
        fast_plasticity_gain=0.04,
        structural_probe_gain=0.0,
        seed=0,
    )
    base.update(kwargs)
    return TrainConfig(**base)  # type: ignore[arg-type]


def test_mirror_stream_only() -> None:
    cfg = _cfg()
    model = StreamingCreature(cfg, 10)
    state = model.initial_state(1)
    logits, state2 = model.step(torch.tensor([1]), state)
    assert logits.shape == (1, 10)
    prev = (state2.mirror_cursor - 1) % cfg.n_mirror
    for i, mi in enumerate(model.mirror_idx.tolist()):
        if i == prev:
            assert state2.h[0, mi].abs().sum() > 0
        else:
            assert state2.h[0, mi].abs().sum() == 0


def test_eligibility_and_credit_move() -> None:
    cfg = _cfg()
    model = StreamingCreature(cfg, 12)
    state = model.initial_state(2)
    for t in range(5):
        logits, state = model.step(torch.tensor([t % 12, (t + 1) % 12]), state)
        state = model.observe_prediction(
            state, logits, torch.tensor([(t + 1) % 12, (t + 2) % 12])
        )
    assert state.eligibility.abs().sum() > 0
    assert state.output_error_credit.abs().sum() > 0
    assert state.edge_eligibility.abs().sum() > 0
    assert state.fast_weight.abs().sum() > 0


def test_gradients_flow() -> None:
    cfg = _cfg()
    model = StreamingCreature(cfg, 10)
    state = model.initial_state(2)
    for i in range(3):
        logits, state = model.step(torch.tensor([i % 10, (i + 1) % 10]), state)
        state = model.observe_prediction(
            state, logits, torch.tensor([(i + 1) % 10, (i + 2) % 10])
        )
        state = state.detach()
    tokens = torch.randint(0, 10, (2, 8))
    targets = torch.randint(0, 10, (2, 8))
    _, state, loss = model.forward_chunk(tokens, targets, state)
    loss.backward()
    assert model.graph.weights.grad is not None
    assert model.graph.weights.grad.abs().sum() > 0
    assert model.rule.gru.weight_ih.grad is not None
    assert model.graph.value.weight.grad is not None


def test_chunk_stream() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 20
    vocab = CharacterVocabulary.from_text(text)
    stream = ContinuousCharStream(text, vocab, batch_size=4, device="cpu")
    x, y = stream.next(8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_ablations_hurt_after_training() -> None:
    """On a toy regular language, state should matter after brief training."""

    text = ("abc" * 300) + ("xyz" * 300)
    vocab = CharacterVocabulary.from_text(text)
    cfg = _cfg(
        lr=5e-3,
        updates=40,
        batch_size=4,
        chunk_length=16,
        n_cells=36,
        hidden=32,
        n_mirror=6,
        n_input=4,
        n_output=4,
    )
    torch.manual_seed(0)
    model = StreamingCreature(cfg, len(vocab))
    stream = ContinuousCharStream(text, vocab, cfg.batch_size, "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    state = model.initial_state(cfg.batch_size)
    model.train()
    for _ in range(cfg.updates):
        tokens, targets = stream.next(cfg.chunk_length)
        _, state, loss = model.forward_chunk(tokens, targets, state)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        state = state.detach()
    ab = evaluate_with_ablations(
        model, vocab, text[-400:], device="cpu", tokens=200, warmup=32
    )
    print(
        f"ablation normal={ab['normal']['bits_per_character']:.3f} "
        f"resetΔ={ab['reset_delta_bpc']:+.3f} shuffleΔ={ab['shuffle_delta_bpc']:+.3f}"
    )
    # Reset should hurt; shuffle usually hurts. Soft assert on reset.
    assert ab["reset_delta_bpc"] > 0.05, ab


def test_online_loss_drops() -> None:
    text = ("abc" * 200) + ("xyz" * 200)
    vocab = CharacterVocabulary.from_text(text)
    cfg = _cfg(lr=5e-3, batch_size=4, chunk_length=16, hidden=32)
    torch.manual_seed(0)
    model = StreamingCreature(cfg, len(vocab))
    stream = ContinuousCharStream(text, vocab, cfg.batch_size, "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    state = model.initial_state(cfg.batch_size)
    model.train()

    def run(n: int) -> float:
        nonlocal state
        total = 0.0
        for _ in range(n):
            tokens, targets = stream.next(cfg.chunk_length)
            _, state, loss = model.forward_chunk(tokens, targets, state)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            state = state.detach()
            total += float(loss.item())
        return total / n

    first = run(8)
    last = run(16)
    chance = math.log(len(vocab))
    print(f"loss first={first:.4f} last={last:.4f} chance={chance:.4f}")
    assert last < first
    assert last < chance * 0.95


def test_attention_optional() -> None:
    cfg = _cfg(use_attention=False)
    model = StreamingCreature(cfg, 10)
    state = model.initial_state(2)
    logits, _ = model.step(torch.tensor([0, 1]), state)
    assert logits.shape == (2, 10)


def test_readout_modes() -> None:
    for mode in ("mean", "concat", "attn"):
        cfg = _cfg(readout_mode=mode)
        model = StreamingCreature(cfg, 10)
        state = model.initial_state(2)
        logits, state = model.step(torch.tensor([0, 1]), state)
        assert logits.shape == (2, 10), mode
        state = model.observe_prediction(state, logits, torch.tensor([1, 2]))


def test_compat_charcorpus() -> None:
    c = CharCorpus.from_text("hello world! " * 5)
    c.reset_cursors(2, seed=1)
    x, y = c.next_batch(2, torch.device("cpu"))
    assert x.shape == (2,)


def main() -> None:
    tests = [
        test_mirror_stream_only,
        test_eligibility_and_credit_move,
        test_gradients_flow,
        test_chunk_stream,
        test_attention_optional,
        test_readout_modes,
        test_compat_charcorpus,
        test_online_loss_drops,
        test_ablations_hurt_after_training,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed}/{len(tests)} failed")
        sys.exit(1)
    print(f"\nall {len(tests)} passed")


if __name__ == "__main__":
    main()
