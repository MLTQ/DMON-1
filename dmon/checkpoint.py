"""Checkpoint I/O for trained rules.

Separated from both `train_m0` and `substrate` so that anything wanting to *load* a
rule — the probe, a renderer, a future contingency analysis — does not have to import
a training loop, and the physics module stays free of file formats.

A checkpoint stores the config alongside the weights on purpose. A rule is only
meaningful with respect to the ecology it was trained in, and a `state_dict` on its own
silently loses that. Loading a rule under different physics is a real experiment
(cross-evaluation) but it must be a deliberate one.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from .substrate import Substrate, SubstrateConfig

FORMAT = 1


def save(path: str | Path, sub: Substrate, meta: dict[str, Any] | None = None) -> Path:
    """Write weights + the physics they were trained under."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "cfg": asdict(sub.cfg),
            "state": sub.state_dict(),
            "meta": meta or {},
        },
        path,
    )
    return path


def load(path: str | Path, device: str | torch.device = "cpu") -> tuple[Substrate, dict]:
    """Rebuild a rule and its ecology. Returns `(substrate, meta)`.

    Unknown config keys are dropped with a warning rather than raising, so a
    checkpoint stays loadable after the config gains or loses a field. Missing keys
    fall back to current defaults — which is a silent change of physics, so it is
    reported too.
    """
    blob = torch.load(Path(path), map_location=device, weights_only=False)

    known = {f.name for f in fields(SubstrateConfig)}
    stored = blob.get("cfg", {})
    unknown = set(stored) - known
    missing = known - set(stored)
    if unknown:
        print(f"[checkpoint] ignoring config keys not in this SubstrateConfig: {sorted(unknown)}")
    if missing:
        print(f"[checkpoint] config keys absent from checkpoint, using defaults: {sorted(missing)}")

    cfg = SubstrateConfig(**{k: v for k, v in stored.items() if k in known})
    sub = Substrate(cfg).to(device)
    sub.load_state_dict(blob["state"])
    sub.eval()
    return sub, blob.get("meta", {})
