"""Checkpointable configuration for the SOL2 organism and trainer."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass


@dataclass
class Sol2Config:
    # Tissue geometry. Memory cells are stream-written and do not run a rule.
    n_input: int = 8
    n_memory: int = 16
    n_compute: int = 64
    n_relay: int = 16
    n_output: int = 8
    # Slow tissue (M1c): a small weakly-contractive population appended after
    # relay so existing anatomy keeps exact indices. n_slow=0 is a bitwise
    # no-op on the kernel — no rule, no rows, no layout change.
    n_slow: int = 0
    slow_alpha_min: float = 0.001
    slow_alpha_max: float = 0.05
    slow_initial_alpha: float = 0.02
    hidden: int = 96
    n_dendrites: int = 12
    initial_active_dendrites: int = 8
    steps_per_token: int = 3
    organ_queries: int = 4

    # Experimental factors for S0-R.
    bounded_operators: bool = True
    cell_identity: bool = True
    operator_bound: float = 1.0
    value_gain: float = 0.85
    attention_temperature: float = 0.7
    edge_logit_limit: float = 4.0
    identity_gain: float = 0.5
    identity_bias: float = 0.10
    identity_time: float = 0.25
    cell_adapter_rank: int = 0
    cell_adapter_gain: float = 0.5
    relay_output_attention: bool = False
    relay_output_gain: float = 1.0
    relay_output_initial_gate: float = 0.0

    # Stream and optimization.
    vocab_size: int = 0
    batch_size: int = 12
    chunk_length: int = 32
    updates: int = 24_000
    lr: float = 1e-3
    lr_min: float = 1e-3
    warmup_updates: int = 200
    schedule: str = "constant"  # constant | cosine
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    reject_grad_norm: float = 10_000.0

    # Evaluation and reproducibility.
    eval_every: int = 500
    log_every: int = 100
    eval_warmup_tokens: int = 256
    eval_tokens: int = 2_048
    transformer_seq_len: int = 128
    seed: int = 7
    device: str = "cpu"

    @property
    def n_cells(self) -> int:
        return (
            self.n_input
            + self.n_memory
            + self.n_compute
            + self.n_relay
            + self.n_output
            + self.n_slow
        )

    @property
    def n_mutable(self) -> int:
        return self.n_cells - self.n_memory

    def validate(self) -> None:
        counts = (
            self.n_input,
            self.n_memory,
            self.n_compute,
            self.n_relay,
            self.n_output,
        )
        if any(n < 1 for n in counts):
            raise ValueError("every tissue must contain at least one cell")
        if self.hidden < 4:
            raise ValueError("hidden must be at least 4")
        if not 1 <= self.initial_active_dendrites <= self.n_dendrites:
            raise ValueError("initial active dendrites must be in [1, n_dendrites]")
        if self.steps_per_token < 1 or self.chunk_length < 1:
            raise ValueError("steps_per_token and chunk_length must be positive")
        if self.organ_queries < 1:
            raise ValueError("organ_queries must be positive")
        if self.cell_adapter_rank < 0 or self.cell_adapter_gain < 0:
            raise ValueError("cell adapter rank and gain cannot be negative")
        if self.relay_output_gain <= 0:
            raise ValueError("relay output gain must be positive")
        if not 0 <= self.relay_output_initial_gate < 1:
            raise ValueError("relay output initial gate must be in [0, 1)")
        if self.operator_bound <= 0 or self.attention_temperature <= 0:
            raise ValueError("operator bound and attention temperature must be positive")
        if self.schedule not in {"constant", "cosine"}:
            raise ValueError("schedule must be 'constant' or 'cosine'")
        if self.n_slow < 0:
            raise ValueError("n_slow cannot be negative")
        if self.n_slow > 0:
            if not 0 < self.slow_alpha_min < self.slow_alpha_max <= 1.0:
                raise ValueError("slow alpha bounds must satisfy 0 < min < max <= 1")
            if not (
                self.slow_alpha_min <= self.slow_initial_alpha <= self.slow_alpha_max
            ):
                raise ValueError("slow initial alpha must lie inside its bounds")
        if self.vocab_size < 0:
            raise ValueError("vocab_size cannot be negative")

    def scaled(self, **overrides) -> "Sol2Config":
        cfg = dataclasses.replace(self, **overrides)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(payload: dict) -> "Sol2Config":
        cfg = Sol2Config(**payload)
        cfg.validate()
        return cfg

    @staticmethod
    def from_json(text: str) -> "Sol2Config":
        return Sol2Config.from_dict(json.loads(text))
