"""Continuous multi-lane character stream with resumable cursors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_CORPUS = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "tinyshakespeare"
    / "input.txt"
)


@dataclass
class Corpus:
    text: str
    stoi: dict[str, int]
    itos: list[str]
    train_ids: torch.Tensor
    holdout_ids: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


def load_corpus(
    path: Path | str = DEFAULT_CORPUS, holdout_fraction: float = 0.10
) -> Corpus:
    text = Path(path).read_text()
    itos = sorted(set(text))
    stoi = {character: index for index, character in enumerate(itos)}
    ids = torch.tensor([stoi[character] for character in text], dtype=torch.long)
    split = int(len(ids) * (1.0 - holdout_fraction))
    return Corpus(text, stoi, itos, ids[:split], ids[split:])


class LaneStream:
    def __init__(self, ids: torch.Tensor, lanes: int, seed: int, device: str):
        self.ids = ids.to(device)
        self.length = len(ids)
        self.device = device
        generator = torch.Generator().manual_seed(seed)
        spacing = max(self.length // lanes, 1)
        base = torch.arange(lanes, dtype=torch.long) * spacing
        jitter = torch.randint(0, spacing, (lanes,), generator=generator)
        self.cursors = ((base + jitter) % self.length).to(device)

    def next_chunk(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(length, device=self.device)
        positions = (self.cursors[:, None] + offsets[None]) % self.length
        inputs = self.ids[positions]
        targets = self.ids[(positions + 1) % self.length]
        self.cursors = (self.cursors + length) % self.length
        return inputs, targets

    def state_dict(self) -> dict:
        return {"cursors": self.cursors.detach().cpu()}

    def load_state_dict(self, payload: dict) -> None:
        cursors = payload["cursors"].to(self.device)
        if cursors.shape != self.cursors.shape:
            raise ValueError("stream cursor shape does not match lane count")
        self.cursors = cursors
