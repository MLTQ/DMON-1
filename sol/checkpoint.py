"""Versioned, atomic checkpoints for the continuous SOL training process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model import FieldState, SolConfig, SparseAxonField
from .stream import CharacterVocabulary
from .train import ContinuousTrainer

SCHEMA_VERSION = 1


@dataclass
class LoadedOrganism:
    """Inference-ready model, vocabulary, persistent state, and provenance."""

    model: SparseAxonField
    vocabulary: CharacterVocabulary
    state: FieldState
    updates: int
    metadata: dict[str, Any]


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


def load_organism(
    path: str | Path,
    device: torch.device | str = "cpu",
    lane: int = 0,
) -> LoadedOrganism:
    """Load one persistent stream lane without requiring the training corpus."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = int(payload.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported SOL checkpoint schema {version}; expected {SCHEMA_VERSION}"
        )
    trainer_payload = payload["trainer"]
    batch_size = int(trainer_payload["batch_size"])
    if not 0 <= lane < batch_size:
        raise ValueError(f"lane must be in [0, {batch_size})")
    device = torch.device(device)
    model = SparseAxonField(
        SolConfig(**trainer_payload["model_config"])
    ).to(device)
    model.load_state_dict(trainer_payload["model"])
    model.eval()
    state = model.state_from_snapshot(
        trainer_payload["field_state"],
        batch_size,
        device,
        lane=lane,
    )
    return LoadedOrganism(
        model=model,
        vocabulary=CharacterVocabulary(
            tuple(trainer_payload["vocabulary"])
        ),
        state=state,
        updates=int(trainer_payload["updates"]),
        metadata=dict(payload.get("metadata", {})),
    )
