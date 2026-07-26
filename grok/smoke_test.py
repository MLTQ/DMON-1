"""Fast smoke: wiring correctness + loss moves on a tiny synthetic stream."""

from __future__ import annotations

import math
import sys

import torch
from torch.nn import functional as F

from .config import TrainConfig
from .corpus import CharCorpus
from .model import StreamingCreature


def _small_cfg(**kwargs) -> TrainConfig:
    base = dict(
        n_cells=40,
        n_input=4,
        n_output=4,
        n_mirror=8,
        n_dendrites=6,
        hidden=32,
        steps_per_token=2,
        batch_size=1,
        truncate_every=32,
        use_attention=True,
        seed=0,
    )
    base.update(kwargs)
    return TrainConfig(**base)  # type: ignore[arg-type]


def test_mirror_is_stream_only() -> None:
    cfg = _small_cfg()
    corpus = CharCorpus.from_text("abcdefg" * 20)
    model = StreamingCreature(cfg, corpus.vocab_size)
    state = model.initial_state(1)
    x = torch.tensor([corpus.stoi["a"]])
    _, state2 = model.step(x, state)
    prev_slot_local = (state2.mirror_cursor - 1) % cfg.n_mirror
    for i, mi in enumerate(model.mirror_idx.tolist()):
        if i == prev_slot_local:
            assert state2.h[0, mi].abs().sum() > 0, "written mirror empty"
        else:
            assert state2.h[0, mi].abs().sum() == 0, "unwritten mirror polluted"


def test_state_persists_without_reset() -> None:
    cfg = _small_cfg()
    corpus = CharCorpus.from_text("hello world! " * 10)
    model = StreamingCreature(cfg, corpus.vocab_size)
    state = model.initial_state(1)
    for t in corpus.encode("hello"):
        _, state = model.step(t.view(1), state)
    assert state.h.abs().sum() > 0


def test_gradients_reach_dendrites_and_rule() -> None:
    cfg = _small_cfg()
    corpus = CharCorpus.from_text("abcd" * 30)
    model = StreamingCreature(cfg, corpus.vocab_size)
    state = model.initial_state(1)
    # Warm state: from pure zeros, query(h)=0 so query.weight gets no grad.
    for i in range(4):
        _, state = model.step(torch.tensor([i % 4]), state)
        state = state.detach()
    logits, _ = model.step(torch.tensor([0]), state)
    loss = F.cross_entropy(logits, torch.tensor([1]))
    loss.backward()
    assert model.graph.weights.grad is not None
    assert model.graph.weights.grad.abs().sum() > 0
    assert model.rule.gru.weight_ih.grad is not None
    assert model.rule.gru.weight_ih.grad.abs().sum() > 0
    assert model.graph.query is not None
    assert model.graph.query.weight.grad is not None
    assert model.graph.query.weight.grad.abs().sum() > 0


def test_attention_optional() -> None:
    cfg = _small_cfg(use_attention=False)
    model = StreamingCreature(cfg, 10)
    state = model.initial_state(2)
    logits, state2 = model.step(torch.tensor([0, 1]), state)
    assert logits.shape == (2, 10)
    assert state2.h.shape[0] == 2


def test_multistream_batch() -> None:
    cfg = _small_cfg(batch_size=4)
    corpus = CharCorpus.from_text("abcdefghijklmnopqrstuvwxyz" * 40)
    corpus.reset_cursors(4, seed=1)
    x, y = corpus.next_batch(4, torch.device("cpu"))
    assert x.shape == (4,)
    assert y.shape == (4,)
    model = StreamingCreature(cfg, corpus.vocab_size)
    state = model.initial_state(4)
    logits, state = model.step(x, state)
    assert logits.shape == (4, corpus.vocab_size)


def test_online_loss_drops() -> None:
    text = ("abc" * 200) + ("xyz" * 200)
    corpus = CharCorpus.from_text(text)
    cfg = _small_cfg(lr=5e-3, steps=400, n_dendrites=6, steps_per_token=3)
    torch.manual_seed(0)
    model = StreamingCreature(cfg, corpus.vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    state = model.initial_state(1)
    model.train()
    opt.zero_grad(set_to_none=True)
    corpus.reset_cursors(1, seed=0)

    def run_chunk(n: int) -> float:
        nonlocal state
        total = 0.0
        window: torch.Tensor | None = None
        wn = 0
        for i in range(n):
            x, y = corpus.next_batch(1, torch.device("cpu"))
            logits, state = model.step(x, state)
            loss = F.cross_entropy(logits, y)
            window = loss if window is None else window + loss
            wn += 1
            total += float(loss.item())
            if (i + 1) % cfg.truncate_every == 0:
                assert window is not None
                (window / wn).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                state = state.detach()
                window = None
                wn = 0
        if wn and window is not None:
            (window / wn).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            state = state.detach()
        return total / n

    first = run_chunk(128)
    last = run_chunk(256)
    chance = math.log(corpus.vocab_size)
    print(f"loss first={first:.4f} last={last:.4f} chance={chance:.4f}")
    assert last < first, f"loss did not drop: {first} → {last}"
    assert last < chance * 0.95, f"still near chance: {last} vs {chance}"


def main() -> None:
    tests = [
        test_mirror_is_stream_only,
        test_state_persists_without_reset,
        test_gradients_reach_dendrites_and_rule,
        test_attention_optional,
        test_multistream_batch,
        test_online_loss_drops,
    ]
    failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {exc}")
    if failed:
        print(f"\n{failed}/{len(tests)} failed")
        sys.exit(1)
    print(f"\nall {len(tests)} passed")


if __name__ == "__main__":
    main()
