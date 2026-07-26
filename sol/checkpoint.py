"""Versioned, atomic checkpoints for the continuous SOL training process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .train import ContinuousTrainer

SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    trainer: ContinuousTrainer,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save a complete trainer snapshot and experiment metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "trainer": trainer.state_dict(),
            "metadata": dict(metadata or {}),
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    text: str,
    device: torch.device | str = "cpu",
) -> tuple[ContinuousTrainer, dict[str, Any]]:
    """Load and validate a complete trainer snapshot."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = int(payload.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported SOL checkpoint schema {version}; expected {SCHEMA_VERSION}"
        )
    trainer = ContinuousTrainer.from_state_dict(payload["trainer"], text, device)
    return trainer, dict(payload.get("metadata", {}))
