"""The stream. Characters arriving one at a time, forever.

Deliberately not a dataset with epochs. `CharStream` is an endless iterator over
independent read positions — one per batch lane — because the creature has no episodes
and the data layer should not quietly reintroduce them by ending.

Held-out evaluation uses a disjoint tail of the corpus, and the reader never crosses
into it, so bits-per-character on the eval stream is a real number.
"""

from __future__ import annotations

from pathlib import Path

import torch

DEFAULT_CORPUS = Path.home() / "Code/petridish/data/tinyshakespeare/input.txt"


class CharStream:
    """Endless character stream with `batch` independent read heads."""

    def __init__(
        self,
        path: str | Path = DEFAULT_CORPUS,
        batch: int = 8,
        device: str | torch.device = "cpu",
        holdout: float = 0.1,
        split: str = "train",
        seed: int = 0,
    ):
        text = Path(path).read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab = len(self.chars)

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        cut = int(len(data) * (1.0 - holdout))
        self.data = (data[:cut] if split == "train" else data[cut:]).to(device)

        g = torch.Generator(device="cpu").manual_seed(seed)
        # Independent positions per lane. Lanes are separate readers of one corpus, not
        # separate episodes — nothing here ever signals an end.
        self.pos = torch.randint(0, len(self.data) - 1, (batch,), generator=g).to(device)
        self.batch = batch
        self.device = device

    def next(self) -> torch.Tensor:
        """The character now arriving, shape (batch,)."""
        tok = self.data[self.pos]
        self.pos = (self.pos + 1) % (len(self.data) - 1)
        return tok

    def peek(self) -> torch.Tensor:
        """The character that will arrive next — the prediction target."""
        return self.data[(self.pos) % (len(self.data) - 1)]

    def __len__(self) -> int:
        return len(self.data)


class History:
    """Rolling window of recent tokens, most recent first.

    This is what gets written into the mirror cells. Kept outside the lattice so that
    "what the stream has sent" and "what the substrate currently holds" stay separable —
    if they were the same object, a bug that let the network alter its own history would
    be invisible.
    """

    def __init__(self, length: int, batch: int, device, pad: int = 0):
        self.buf = torch.full((batch, length), pad, dtype=torch.long, device=device)

    def push(self, token: torch.Tensor):
        self.buf = torch.cat([token[:, None], self.buf[:, :-1]], dim=1)

    @property
    def tokens(self) -> torch.Tensor:
        return self.buf
