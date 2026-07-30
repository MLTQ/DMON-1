"""Tissue probes: is a cell population doing work, and is it differentiated?

Two questions that a single ablation cannot separate, and conflating them
produced a wrong F1 decision on 2026-07-30:

  freeze(P)  — hold population P at zero every micro-step. Measures whether
               the tissue does WORK. Cost = what the rest of the organism
               cannot recover without it.
  shuffle(P) — permute P's state across cells every token. Measures whether
               the tissue is DIFFERENTIATED: whether *which* cell holds
               which state carries information.

A mean-field population — many cells carrying interchangeable state — is
permutation-invariant (shuffle ~ 0) while contributing heavily (freeze
large). F0 measured exactly that: shuffle +0.036..+0.066, freeze
+0.280..+0.831. Reporting only shuffle read as "inert tissue" and was wrong.

The differentiation question is not idle: an undifferentiated population
should scale sublinearly (more of an interchangeable thing averages), while
a differentiated one can specialize. That is the prediction F1 tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import FableConfig
from .evaluate import _prequential
from .model import Fable, OrganismState
from .stream import load_corpus


def load_creature(checkpoint: Path, device: str) -> tuple[Fable, FableConfig]:
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = FableConfig(**saved["config"])
    model = Fable(cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, cfg


def probe(checkpoint: Path, device: str, holdout=None) -> dict:
    model, cfg = load_creature(checkpoint, device)
    ids = holdout if holdout is not None else load_corpus().holdout_ids
    w, s = cfg.eval_warmup_tokens, cfg.eval_tokens

    def bpc(frozen=None, mutate=None):
        return _prequential(model, ids, w, s, device,
                            frozen_idx=frozen, mutate=mutate)["bits_per_character"]

    internal, mirror = model.internal_idx, model.mirror_idx
    perm = internal[torch.randperm(
        len(internal), generator=torch.Generator().manual_seed(1729))]

    def shuffle_internal(state: OrganismState) -> OrganismState:
        return OrganismState(state.h.index_copy(1, internal, state.h[:, perm]),
                             state.mirror_cursor)

    normal = bpc()
    out = {
        "checkpoint": str(checkpoint),
        "n_cells": model.n_cells,
        "n_internal": len(internal),
        "n_mirror": len(mirror),
        "normal_bpc": normal,
        "freeze_internal_bpc": bpc(frozen=internal),
        "freeze_mirror_bpc": bpc(frozen=mirror),
        "freeze_both_bpc": bpc(frozen=torch.cat([internal, mirror])),
        "shuffle_internal_bpc": bpc(mutate=shuffle_internal),
    }
    for key in ("freeze_internal", "freeze_mirror", "freeze_both",
                "shuffle_internal"):
        out[f"{key}_delta"] = out[f"{key}_bpc"] - normal
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe tissue work vs differentiation")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = []
    print("%-34s %8s %9s %9s %9s" % (
        "checkpoint", "normal", "frzInt", "frzMir", "shufInt"))
    for ck in args.checkpoints:
        r = probe(Path(ck), args.device)
        results.append(r)
        print("%-34s %8.4f %+9.3f %+9.3f %+9.3f" % (
            Path(ck).parent.name + "/" + Path(ck).name, r["normal_bpc"],
            r["freeze_internal_delta"], r["freeze_mirror_delta"],
            r["shuffle_internal_delta"]))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
