"""Character vocabulary and stateful continuous-corpus batching."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CharacterVocabulary:
    """A deterministic character-to-index mapping."""

    characters: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharacterVocabulary":
        if len(set(text)) < 2:
            raise ValueError("text must contain at least two distinct characters")
        return cls(tuple(sorted(set(text))))

    def __len__(self) -> int:
        return len(self.characters)

    def encode(self, text: str, device: torch.device | str | None = None) -> torch.Tensor:
        lookup = {character: index for index, character in enumerate(self.characters)}
        missing = sorted(set(text) - set(lookup))
        if missing:
            raise ValueError(f"text contains characters outside vocabulary: {missing}")
        return torch.tensor([lookup[character] for character in text], device=device)

    def decode(self, tokens: torch.Tensor | list[int]) -> str:
        values = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        return "".join(self.characters[int(index)] for index in values)


class ContinuousCharStream:
    """Produces adjacent truncated windows from persistent, staggered corpus lanes."""

    def __init__(
        self,
        text: str,
        vocabulary: CharacterVocabulary,
        batch_size: int,
        device: torch.device | str = "cpu",
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        encoded = vocabulary.encode(text, device)
        if encoded.numel() < batch_size * 2:
            raise ValueError("text is too short for the requested batch size")
        self.data = encoded
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.starts = (
            torch.arange(batch_size, device=self.device)
            * (encoded.numel() // batch_size)
        )
        self.position = 0

    def next(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return inputs and their immediate successor targets, then advance."""

        if length < 1:
            raise ValueError("length must be positive")
        offsets = torch.arange(length, device=self.device)
        positions = (self.starts[:, None] + self.position + offsets[None, :])
        inputs = self.data[positions % self.data.numel()]
        targets = self.data[(positions + 1) % self.data.numel()]
        self.position = (self.position + length) % self.data.numel()
        return inputs, targets

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = int(state["position"]) % self.data.numel()
