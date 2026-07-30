"""Character vocabulary and multi-lane continuous stream (SOL pattern)."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def ensure_shakespeare(data_dir: str | Path) -> Path:
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
            f"could not obtain Tiny Shakespeare at {path}: {exc}"
        ) from exc
    return path


@dataclass(frozen=True)
class CharacterVocabulary:
    characters: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharacterVocabulary":
        chars = tuple(sorted(set(text)))
        if len(chars) < 2:
            raise ValueError("text must contain at least two distinct characters")
        return cls(chars)

    def __len__(self) -> int:
        return len(self.characters)

    def encode(self, text: str, device: torch.device | str | None = None) -> torch.Tensor:
        lookup = {c: i for i, c in enumerate(self.characters)}
        missing = sorted(set(text) - set(lookup))
        if missing:
            raise ValueError(f"characters outside vocabulary: {missing}")
        return torch.tensor([lookup[c] for c in text], device=device, dtype=torch.long)

    def decode(self, tokens: torch.Tensor | list[int]) -> str:
        values = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        return "".join(self.characters[int(i)] for i in values)


class ContinuousCharStream:
    """Staggered lanes produce adjacent truncated windows without episode resets."""

    def __init__(
        self,
        text: str,
        vocabulary: CharacterVocabulary,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        encoded = vocabulary.encode(text, device)
        if encoded.numel() < batch_size * 2:
            raise ValueError("text too short for batch size")
        self.data = encoded
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.starts = (
            torch.arange(batch_size, device=self.device)
            * (encoded.numel() // batch_size)
        )
        self.position = 0

    def next(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if length < 1:
            raise ValueError("length must be positive")
        offsets = torch.arange(length, device=self.device)
        positions = self.starts[:, None] + self.position + offsets[None, :]
        n = self.data.numel()
        inputs = self.data[positions % n]
        targets = self.data[(positions + 1) % n]
        self.position = (self.position + length) % n
        return inputs, targets


class CharCorpus:
    """Compatibility facade for older smoke tests and simple cursors."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.vocab = CharacterVocabulary.from_text(text)
        self.stoi = {c: i for i, c in enumerate(self.vocab.characters)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.vocab)
        self.data = self.vocab.encode(text)
        self._cursors: torch.Tensor | None = None

    @classmethod
    def from_shakespeare(cls, data_dir: str | Path = "data/tinyshakespeare") -> "CharCorpus":
        return cls(ensure_shakespeare(data_dir).read_text(encoding="utf-8"))

    @classmethod
    def from_text(cls, text: str) -> "CharCorpus":
        return cls(text)

    def encode(self, s: str) -> torch.Tensor:
        return self.vocab.encode(s)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        return self.vocab.decode(ids)

    def reset_cursors(self, batch_size: int, *, seed: int = 0) -> None:
        n = max(len(self.data) - 1, 1)
        g = torch.Generator().manual_seed(seed)
        self._cursors = (
            torch.zeros(1, dtype=torch.long)
            if batch_size == 1
            else torch.randint(0, n, (batch_size,), generator=g)
        )

    def next_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.data)
        if self._cursors is None or self._cursors.numel() != batch_size:
            self.reset_cursors(batch_size)
        assert self._cursors is not None
        cur = self._cursors % (n - 1)
        x = self.data[cur]
        y = self.data[cur + 1]
        self._cursors = (cur + 1) % (n - 1)
        return x.to(device), y.to(device)

    def window(
        self, start: int, length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.data)
        idx = torch.arange(start, start + length) % (n - 1)
        return self.data[idx].to(device), self.data[idx + 1].to(device)

    def train_text(self, holdout_frac: float = 0.1) -> str:
        cut = int(len(self.text) * (1.0 - holdout_frac))
        return self.text[: max(cut, 2)]

    def holdout_text(self, holdout_frac: float = 0.1) -> str:
        cut = int(len(self.text) * (1.0 - holdout_frac))
        return self.text[cut:]
