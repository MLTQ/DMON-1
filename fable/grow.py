"""Runtime growth: extend the field mid-stream, without touching the rule.

The shared rule is why growth is cheap: adding cells adds per-edge logits and
nothing else — no rule parameters, no readout parameters (the output bank is
fixed), no embedding change. The graft is output-silent at t=0 by
construction: new cells start at h=0, so their value-projected messages are
zero until their state develops.

Wiring at the graft:
  - each new cell samples K sources from the whole enlarged non-output pool
    (old inputs/mirrors/internals + other new cells)
  - each old mutable cell donates its weakest dendrite slot (never an output
    cell's forced sensory slot 0) to a new cell, round-robin, logit reset to
    0 — without this, nothing would ever read a new cell and the added tissue
    would be unreachable by construction

Optimizer surgery preserves Adam moments for every surviving parameter and
for the surviving slice of the resized logit tensor, so the graft is not a
disguised optimizer restart (that would confound F1's recruitment claim).

CLI runs one F1 arm: small (80 fixed), grown (80→128 at u2000), born (128).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn as nn

from .config import FableConfig
from .evaluate import _prequential
from .model import Fable, OrganismState, count_parameters
from .stream import load_corpus
from .train import config_from_args, add_config_args, run

GROW_AT = 2000
SMALL_CELLS = 80
BIG_CELLS = 128


def grow_field(model: Fable, opt: torch.optim.AdamW, n_new: int):
    """Extend the field by `n_new` internal cells in place.

    Returns (new_cell_ids, state_migrator).
    """
    device = model.mutable_idx.device
    old_n = model.n_cells
    k = model.graph.k
    gen = torch.Generator().manual_seed(model.cfg.seed + old_n)
    new_ids = torch.arange(old_n, old_n + n_new)

    pool = torch.cat([model.input_idx.cpu(), model.mirror_idx.cpu(),
                      model.internal_idx.cpu(), new_ids])
    new_rows = torch.empty(n_new, k, dtype=torch.long)
    for i in range(n_new):
        perm = pool[torch.randperm(len(pool), generator=gen)]
        new_rows[i] = perm[:k]

    sources = torch.cat([model.graph.sources.cpu(), new_rows]).to(device)

    old_logit = model.graph.logit
    new_logit_rows = torch.empty(n_new, k).uniform_(-0.05, 0.05, generator=gen)
    logit_data = torch.cat([old_logit.data.cpu(), new_logit_rows]).to(device)

    grafted = []
    donors = torch.cat([model.input_idx, model.internal_idx, model.output_idx])
    output_set = set(model.output_idx.tolist())
    for i, cell in enumerate(donors.tolist()):
        row_logit = logit_data[cell].abs().clone()
        if cell in output_set:
            row_logit[0] = float("inf")  # never donate the forced sensory slot
        slot = int(row_logit.argmin())
        target_new = int(new_ids[i % n_new])
        sources[cell, slot] = target_new
        logit_data[cell, slot] = 0.0
        grafted.append({"cell": cell, "slot": slot, "reads": target_new})

    new_logit = nn.Parameter(logit_data)
    model.graph.sources = sources  # plain assignment updates the registered buffer
    for group in opt.param_groups:
        group["params"] = [new_logit if p is old_logit else p
                           for p in group["params"]]
    if old_logit in opt.state:
        old_state = opt.state.pop(old_logit)
        opt.state[new_logit] = {
            "step": old_state["step"],
            "exp_avg": torch.cat(
                [old_state["exp_avg"], torch.zeros(n_new, k, device=device)]),
            "exp_avg_sq": torch.cat(
                [old_state["exp_avg_sq"], torch.zeros(n_new, k, device=device)]),
        }
    model.graph.logit = new_logit
    model.graph.n_cells = old_n + n_new

    model.internal_idx = torch.cat([model.internal_idx, new_ids.to(device)])
    model._rebuild_mutable()
    model.cfg = model.cfg.scaled(n_cells=old_n + n_new)

    def migrate(state: OrganismState) -> OrganismState:
        pad = torch.zeros(state.h.shape[0], n_new, state.h.shape[2],
                          device=state.h.device)
        return OrganismState(torch.cat([state.h, pad], dim=1),
                             state.mirror_cursor)

    return new_ids.tolist(), migrate, grafted


def recruitment_probe(checkpoint: Path, added: list[int], device: str) -> dict:
    """BPC with the added cells zeroed+frozen vs untouched, from a saved
    grown checkpoint. If freezing is free, the tissue was never recruited."""
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = FableConfig(**saved["config"])
    model = Fable(cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    corpus = load_corpus()
    frozen = torch.tensor(added, dtype=torch.long, device=device)
    normal = _prequential(model, corpus.holdout_ids, cfg.eval_warmup_tokens,
                          cfg.eval_tokens, device)
    ablated = _prequential(model, corpus.holdout_ids, cfg.eval_warmup_tokens,
                           cfg.eval_tokens, device, frozen_idx=frozen)
    return {"normal_bpc": normal["bits_per_character"],
            "frozen_added_bpc": ablated["bits_per_character"],
            "recruitment_delta_bpc":
                ablated["bits_per_character"] - normal["bits_per_character"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one F1 growth arm")
    ap.add_argument("--arm", choices=("small", "grown", "born"), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grow-at", type=int, default=GROW_AT)
    ap.add_argument("--probe-at", type=int, default=GROW_AT + 500)
    add_config_args(ap)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config_from_args(args)
    cfg = cfg.scaled(n_cells=BIG_CELLS if args.arm == "born" else SMALL_CELLS)

    growth_record: dict = {"arm": args.arm}
    added_cells: list[int] = []

    def on_update(update, model, opt, state):
        if args.arm != "grown":
            return None
        if update == args.grow_at + 1:
            added, migrate, grafted = grow_field(
                model, opt, BIG_CELLS - SMALL_CELLS)
            added_cells.extend(added)
            growth_record.update(
                grow_at=update, added_cells=added, grafts=grafted,
                params_after=count_parameters(model))
            print(f"[grown] grafted {len(added)} cells at u{update}; "
                  f"params={growth_record['params_after']}", flush=True)
            return migrate(state)
        if update == args.probe_at + 1 and added_cells:
            torch.save({"model": model.state_dict(),
                        "config": dataclasses.asdict(model.cfg)},
                       out_dir / "post_graft.pt")
        return None

    run("creature", cfg, out_dir, args.device, on_update=on_update)

    # Tissue probe for every arm: freeze (does the tissue work) alongside
    # shuffle (is it differentiated). Reporting only one of these produced a
    # wrong F1 decision earlier today — see fable/probe.md.
    from .probe import probe as tissue_probe
    growth_record["tissue"] = tissue_probe(out_dir / "creature.pt", args.device)
    print(json.dumps(growth_record["tissue"], indent=1), flush=True)

    if args.arm == "grown":
        growth_record["probe_final"] = recruitment_probe(
            out_dir / "creature.pt", added_cells, args.device)
        if (out_dir / "post_graft.pt").exists():
            growth_record["probe_post_graft"] = recruitment_probe(
                out_dir / "post_graft.pt", added_cells, args.device)
        print(json.dumps({k: growth_record[k] for k in
                          ("probe_final", "probe_post_graft")
                          if k in growth_record}, indent=1))
    (out_dir / "growth.json").write_text(json.dumps(growth_record, indent=1))


if __name__ == "__main__":
    main()
