"""Continuous multi-lane character stream over a contiguous 90/10 split.

The organism never sees an episode boundary: each lane is an independent cursor
into the training region that wraps modulo the region, emitting (current, next)
character pairs forever. Evaluation reads the held-out tail the same way, once,
prequentially.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare" / "input.txt"


@dataclass
class Corpus:
    text: str
    stoi: dict
    itos: list
    train_ids: torch.Tensor   # long [n_train]
    holdout_ids: torch.Tensor # long [n_holdout]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


def load_corpus(path: Path | str = DEFAULT_CORPUS, holdout_fraction: float = 0.10) -> Corpus:
    text = Path(path).read_text()
    itos = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(itos)}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(len(ids) * (1.0 - holdout_fraction))
    return Corpus(text=text, stoi=stoi, itos=itos,
                  train_ids=ids[:split], holdout_ids=ids[split:])


class LaneStream:
    """`n_lanes` independent wrapping cursors over one id tensor.

    `next_chunk(L)` returns (inputs, targets), each long [n_lanes, L], and
    advances every cursor by L. Lanes start evenly spaced so a batch never
    contains near-duplicate context at initialization.
    """

    def __init__(self, ids: torch.Tensor, n_lanes: int, seed: int, device: str):
        self.ids = ids.to(device)
        self.n = len(ids)
        g = torch.Generator().manual_seed(seed)
        base = torch.arange(n_lanes, dtype=torch.long) * (self.n // max(n_lanes, 1))
        jitter = torch.randint(0, max(self.n // max(n_lanes, 1), 1), (n_lanes,), generator=g)
        self.cursors = ((base + jitter) % self.n).to(device)
        self.device = device

    def next_chunk(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(length, device=self.device)
        idx = (self.cursors.unsqueeze(1) + offsets.unsqueeze(0)) % self.n
        inputs = self.ids[idx]
        targets = self.ids[(idx + 1) % self.n]
        self.cursors = (self.cursors + length) % self.n
        return inputs, targets
