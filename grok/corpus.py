"""Character-level corpus and endless multi-stream for online training."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def ensure_shakespeare(data_dir: str | Path) -> Path:
    """Download Tiny Shakespeare once into data_dir/input.txt.

    Falls back to a sibling petridish cache when SSL/network fails, so local
    development is not blocked by certificate quirks.
    """

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "input.txt"
    if path.exists() and path.stat().st_size > 0:
        return path

    candidates = [
        Path.home() / "Code/petridish/data/tinyshakespeare/input.txt",
        Path(__file__).resolve().parents[2] / "petridish/data/tinyshakespeare/input.txt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            path.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
            return path

    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"could not obtain Tiny Shakespeare at {path}. "
            f"Place input.txt there manually. Underlying error: {exc}"
        ) from exc
    return path


class CharCorpus:
    """Byte-level char vocabulary; independent cursors per batch stream."""

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("corpus text is empty")
        self.text = text
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self._cursors: torch.Tensor | None = None

    @classmethod
    def from_shakespeare(cls, data_dir: str | Path = "data/tinyshakespeare") -> "CharCorpus":
        path = ensure_shakespeare(data_dir)
        return cls(path.read_text(encoding="utf-8"))

    @classmethod
    def from_text(cls, text: str) -> "CharCorpus":
        return cls(text)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)

    def reset_cursors(self, batch_size: int, *, seed: int = 0) -> None:
        """Place batch streams at staggered offsets so they see different text."""

        n = max(len(self.data) - 1, 1)
        g = torch.Generator().manual_seed(seed)
        if batch_size == 1:
            self._cursors = torch.zeros(1, dtype=torch.long)
        else:
            self._cursors = torch.randint(0, n, (batch_size,), generator=g)

    def next_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) each shape [B]: current char and next-char target.

        Each batch row has its own cursor. Organism state is *not* reset here —
        only the text pointers advance. Continuous streams, parallelized.
        """

        n = len(self.data)
        if self._cursors is None or self._cursors.numel() != batch_size:
            self.reset_cursors(batch_size)

        assert self._cursors is not None
        # Work on CPU indices then move tokens to device.
        cur = self._cursors % (n - 1)
        x = self.data[cur]
        y = self.data[cur + 1]
        self._cursors = (cur + 1) % (n - 1)
        return x.to(device), y.to(device)

    def window(self, start: int, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed window for evaluation: x[t] predicts y[t]=x[t+1]."""

        n = len(self.data)
        idx = torch.arange(start, start + length) % (n - 1)
        x = self.data[idx].to(device)
        y = self.data[idx + 1].to(device)
        return x, y
