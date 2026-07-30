"""Hyperparameters for the grok streaming organism."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    """Geometry, credit, and trainer knobs. Defaults target a real S0 attempt."""

    # Field geometry
    n_cells: int = 64
    n_input: int = 8
    n_output: int = 8
    n_mirror: int = 16
    n_dendrites: int = 8
    hidden: int = 64
    steps_per_token: int = 3
    use_attention: bool = True
    # mean: pool output cells; concat: flatten then project; attn: learnable pool
    readout_mode: str = "mean"

    # Event memory / delayed credit (SOL-promoted path)
    eligibility_decay: float = 0.92
    edge_eligibility_decay: float = 0.96
    reward_gain: float = 0.25
    reward_baseline_decay: float = 0.99
    output_error_credit_gain: float = 0.5
    output_error_credit_decay: float = 0.80
    fast_plasticity_gain: float = 0.04
    fast_weight_decay: float = 0.995
    fast_weight_limit: float = 0.25

    # Structure (default off — SOL: only after capability)
    structural_probe_gain: float = 0.0
    structural_phase_every: int = 25
    structural_confirmations: int = 2
    structural_margin: float = 0.01
    structural_require_global_fitness: bool = True

    # Stream / optimizer
    batch_size: int = 16
    chunk_length: int = 32
    lr: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 7

    # Run
    updates: int = 2000
    log_every: int = 50
    eval_every: int = 250
    eval_tokens: int = 2048
    eval_warmup: int = 256
    device: str = "cpu"
    data_dir: str = "data/tinyshakespeare"

    # Back-compat aliases used by older smoke tests
    @property
    def steps(self) -> int:
        return self.updates

    @property
    def truncate_every(self) -> int:
        return self.chunk_length

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
        if self.batch_size < 1 or self.chunk_length < 1:
            raise ValueError("batch_size and chunk_length must be >= 1")
        for name in (
            "eligibility_decay",
            "edge_eligibility_decay",
            "reward_baseline_decay",
            "output_error_credit_decay",
            "fast_weight_decay",
        ):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.fast_weight_limit <= 0:
            raise ValueError("fast_weight_limit must be positive")
        if self.structural_probe_gain < 0:
            raise ValueError("structural_probe_gain must be nonnegative")
        if self.readout_mode not in ("mean", "concat", "attn"):
            raise ValueError("readout_mode must be mean, concat, or attn")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def n_internal(self) -> int:
        return self.n_cells - self.n_input - self.n_output - self.n_mirror
