"""Hyperparameters for the streaming creature and trainer."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    """S0 defaults: smoke-friendly, scalable toward nanoGPT-class capacity."""

    # Field geometry
    n_cells: int = 128
    n_input: int = 16
    n_output: int = 16
    n_mirror: int = 32
    n_dendrites: int = 12
    hidden: int = 128
    steps_per_token: int = 4
    use_attention: bool = True

    # Stream / credit
    truncate_every: int = 64  # tokens of BPTT before detach
    batch_size: int = 16
    lr: float = 3e-3
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # Data
    seq_len_report: int = 256
    seed: int = 7

    # Run
    steps: int = 5000
    log_every: int = 100
    eval_every: int = 1000
    eval_tokens: int = 512
    device: str = "cpu"
    data_dir: str = "data/tinyshakespeare"

    def validate(self) -> None:
        n_ports = self.n_input + self.n_output + self.n_mirror
        if self.n_cells < n_ports:
            raise ValueError(
                f"n_cells={self.n_cells} must cover input+output+mirror={n_ports}"
            )
        if self.n_dendrites < 1:
            raise ValueError("n_dendrites must be >= 1")
        if self.steps_per_token < 1:
            raise ValueError("steps_per_token must be >= 1")
        if self.truncate_every < 1:
            raise ValueError("truncate_every must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def n_internal(self) -> int:
        return self.n_cells - self.n_input - self.n_output - self.n_mirror
