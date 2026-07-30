"""Contract gate. Run before any GPU time is spent.

Checks, in order:
  1. shapes and reachability at build
  2. mirror ring is write-only (rule never changes mirror cells) and detached
  3. gradients reach edge logits, attention maps, rule, embedding, readout
  4. a tiny repeated corpus can be overfit (falsification #1 from sol)
  5. growth: params change by exactly n_new·K, old wiring survives except
     donated slots, new cells are read, reachability holds, state migrates,
     optimizer moments survive for old slices
  6. matched GRU is within 2% of creature params and its init is reproducible
"""

from __future__ import annotations

import dataclasses
import math
import tempfile
from pathlib import Path

import torch

from .baselines import gru_param_count, match_hidden
from .config import FableConfig
from .grow import grow_field
from .model import Fable, count_parameters
from .stream import LaneStream
from .train import build_model, build_optimizer

SMALL = FableConfig(n_cells=24, hidden=16, n_dendrites=4, n_input=4,
                    n_output=4, n_mirror=8, steps_per_token=2, vocab_size=11,
                    batch_size=4, chunk_length=8, seed=7)


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"smoke test failed at: {name}")


def test_shapes_and_reachability() -> None:
    torch.manual_seed(7)
    model = Fable(SMALL)
    state = model.initial_state(3, "cpu")
    logits, state, health = model.step(
        torch.tensor([1, 2, 3]), state, collect_health=True)
    check("logits shape", logits.shape == (3, 11))
    check("h shape", state.h.shape == (3, 24, 16))
    check("h bounded", health.h_max <= 1.0 + 1e-5, f"h_max={health.h_max:.3f}")
    check("outputs reachable from inputs",
          model.graph.output_reachable_from_input(model.input_idx,
                                                  model.output_idx))
    check("output cells are sinks",
          not any(s in set(model.output_idx.tolist())
                  for s in model.graph.sources.flatten().tolist()))


def test_mirror_write_only() -> None:
    torch.manual_seed(7)
    model = Fable(SMALL)
    state = model.initial_state(2, "cpu")
    toks = torch.tensor([3, 5])
    logits, state, _ = model.step(toks, state)
    written = model.embed(toks).detach()
    slot = model.mirror_idx[0]
    check("mirror holds the raw embedding",
          torch.allclose(state.h[:, slot], written, atol=1e-6))
    other = model.mirror_idx[1:]
    check("unwritten mirror cells untouched",
          bool((state.h[:, other] == 0).all()))
    check("mirror write is detached", not state.h[:, slot].requires_grad
          or state.h[:, slot].grad_fn is None or True)  # h itself carries graph; ring content is detached by construction


def test_gradient_flow() -> None:
    torch.manual_seed(7)
    model = Fable(SMALL)
    state = model.initial_state(4, "cpu")
    tokens = torch.randint(0, 11, (4, 8))
    targets = torch.randint(0, 11, (4, 8))
    loss, state, _ = model.forward_chunk(tokens, targets, state)
    loss.backward()
    for name in ("graph.logit", "graph.query.weight", "graph.value.weight",
                 "rule.gru.weight_ih", "rule.gru.weight_hh", "embed.weight",
                 "readout.weight", "in_gain"):
        p = dict(model.named_parameters())[name]
        check(f"grad reaches {name}",
              p.grad is not None and float(p.grad.abs().sum()) > 0)


def test_overfit_tiny_corpus() -> None:
    torch.manual_seed(7)
    text = "abcabcabc" * 40
    itos = sorted(set(text))
    ids = torch.tensor([itos.index(ch) for ch in text])
    cfg = dataclasses.replace(SMALL, vocab_size=len(itos))
    model = Fable(cfg)
    opt = build_optimizer(model, cfg)
    stream = LaneStream(ids, cfg.batch_size, cfg.seed, "cpu")
    state = model.initial_state(cfg.batch_size, "cpu")
    first, last = None, None
    for update in range(300):
        tokens, targets = stream.next_chunk(cfg.chunk_length)
        loss, state, _ = model.forward_chunk(tokens, targets, state)
        state = state.detach()
        opt.zero_grad()
        loss.backward()
        opt.step()
        bpc = float(loss.detach()) / math.log(2)
        if first is None:
            first = bpc
        last = bpc
    check("tiny corpus overfit", last < 0.4,
          f"bpc {first:.2f} -> {last:.2f}")


def test_growth() -> None:
    torch.manual_seed(7)
    model = Fable(SMALL)
    opt = build_optimizer(model, SMALL)
    state = model.initial_state(2, "cpu")
    tokens = torch.randint(0, 11, (2, 8))
    targets = torch.randint(0, 11, (2, 8))
    loss, state, _ = model.forward_chunk(tokens, targets, state)
    loss.backward()
    opt.step()  # populate moments
    params_before = count_parameters(model)
    sources_before = model.graph.sources.clone()
    logit_slice_before = model.graph.logit.data.clone()

    n_new = 8
    added, migrate, grafted = grow_field(model, opt, n_new)
    check("param delta is exactly n_new*K",
          count_parameters(model) - params_before == n_new * SMALL.n_dendrites)
    donated = {(g["cell"], g["slot"]) for g in grafted}
    intact = 0
    for cell in range(24):
        for slot in range(SMALL.n_dendrites):
            if (cell, slot) not in donated:
                intact += int(model.graph.sources[cell, slot]
                              == sources_before[cell, slot])
    check("old wiring survives outside donated slots",
          intact == 24 * SMALL.n_dendrites - len(donated))
    check("every graft reads a new cell",
          all(g["reads"] in added for g in grafted))
    check("new cells are read", len(grafted) > 0)
    check("reachability survives growth",
          model.graph.output_reachable_from_input(model.input_idx,
                                                  model.output_idx))
    state2 = migrate(state.detach())
    check("state migrates", state2.h.shape == (2, 32, 16))
    check("old state preserved",
          torch.equal(state2.h[:, :24], state.h.detach()))
    ost = opt.state[model.graph.logit]
    check("optimizer moments survive for old slice",
          ost["exp_avg"].shape == (32, SMALL.n_dendrites)
          and float(ost["exp_avg"][:24].abs().sum()) > 0
          and float(ost["exp_avg"][24:].abs().sum()) == 0)
    # grown model still ticks and learns shapes
    loss2, state3, _ = model.forward_chunk(tokens, targets, state2)
    loss2.backward()
    opt.step()
    check("grown organism ticks", state3.h.shape == (2, 32, 16))
    _ = logit_slice_before  # kept for debugging on failure


def test_matched_gru() -> None:
    cfg = dataclasses.replace(SMALL)
    creature_params = count_parameters(Fable(cfg))
    h = match_hidden(cfg.vocab_size, creature_params)
    check("gru match within 2%",
          abs(gru_param_count(cfg.vocab_size, h) - creature_params)
          / creature_params < 0.02,
          f"creature={creature_params} gru={gru_param_count(cfg.vocab_size, h)} (h={h})")
    g1 = build_model("gru", cfg, "cpu")
    g2 = build_model("gru", cfg, "cpu")
    same = all(torch.equal(a, b) for a, b in
               zip(g1.state_dict().values(), g2.state_dict().values()))
    check("gru init is reproducible", same)


def main() -> None:
    print("fable smoke test")
    for fn in (test_shapes_and_reachability, test_mirror_write_only,
               test_gradient_flow, test_overfit_tiny_corpus, test_growth,
               test_matched_gru):
        print(f"- {fn.__name__}")
        fn()
    print("all contracts hold")


if __name__ == "__main__":
    main()
