"""Checkpointable configuration for the fable2 Broca-graft treatment."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

from sol2.config import Sol2Config


@dataclass
class Fable2Config:
    # Organism geometry. Memory tissue must span the longest exposure so the FIFO
    # holds the whole passage, not its tail.
    n_input: int = 8
    n_memory: int = 256
    n_compute: int = 96
    n_relay: int = 32
    n_output: int = 16
    hidden: int = 96
    n_dendrites: int = 12
    initial_active_dendrites: int = 8
    steps_per_token: int = 3
    organ_queries: int = 4
    cell_identity: bool = True
    cell_adapter_rank: int = 4
    # M1c slow tissue: 0 keeps the M0/G-line anatomy bitwise unchanged.
    n_slow: int = 0
    slow_alpha_min: float = 0.001
    slow_alpha_max: float = 0.05
    slow_initial_alpha: float = 0.02
    seed: int = 7

    # Broca graft. Depth fractions index frozen transformer blocks (0, 1] and are
    # resolved against the real layer count at attach; injections are additive
    # cross-attention residuals, bounded by control_gain componentwise.
    depth_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    n_control_tokens: int = 4
    control_rank: int = 8
    control_gain: float = 1.0
    trunk_width: int = 128

    # Training. Paired causal contrast is the objective; the task term is optional
    # and defaults off so the contrast cannot be diluted silently.
    updates: int = 300
    lr: float = 1e-3
    organ_lr_multiplier: float = 1.0
    grad_clip: float = 1.0
    reject_grad_norm: float = 10_000.0
    contrast_margin: float = 0.1
    task_weight: float = 0.0
    eval_every: int = 25
    checkpoint_every: int = 25
    permutation_seed: int = 101
    max_memory_tokens: int = 256
    max_question_tokens: int = 256

    def organism_config(self, *, device: str) -> Sol2Config:
        return Sol2Config(
            n_input=self.n_input,
            n_memory=self.n_memory,
            n_compute=self.n_compute,
            n_relay=self.n_relay,
            n_output=self.n_output,
            hidden=self.hidden,
            n_dendrites=self.n_dendrites,
            initial_active_dendrites=self.initial_active_dendrites,
            steps_per_token=self.steps_per_token,
            organ_queries=self.organ_queries,
            cell_identity=self.cell_identity,
            cell_adapter_rank=self.cell_adapter_rank,
            n_slow=self.n_slow,
            slow_alpha_min=self.slow_alpha_min,
            slow_alpha_max=self.slow_alpha_max,
            slow_initial_alpha=self.slow_initial_alpha,
            # The organism's own token organ is unused here; the Broca organ senses
            # frozen features directly. Two keeps the embedding nondegenerate.
            vocab_size=2,
            seed=self.seed,
            device=device,
        )

    def validate(self) -> None:
        if not self.depth_fractions:
            raise ValueError("at least one injection depth is required")
        if len(set(self.depth_fractions)) != len(self.depth_fractions):
            raise ValueError("injection depth fractions must be distinct")
        if any(not 0.0 < fraction <= 1.0 for fraction in self.depth_fractions):
            raise ValueError("depth fractions must lie in (0, 1]")
        if self.n_control_tokens < 1 or self.control_rank < 1 or self.trunk_width < 1:
            raise ValueError("control tokens, rank, and trunk width must be positive")
        if self.control_gain <= 0:
            raise ValueError("control_gain must be positive")
        if self.updates < 1 or self.eval_every < 1 or self.checkpoint_every < 1:
            raise ValueError("update and interval counts must be positive")
        if self.lr <= 0 or self.organ_lr_multiplier <= 0 or self.grad_clip <= 0:
            raise ValueError("learning rates and clip must be positive")
        if self.contrast_margin <= 0:
            raise ValueError("contrast margin must be positive")
        if self.task_weight < 0:
            raise ValueError("task_weight cannot be negative")
        if self.max_memory_tokens > self.n_memory:
            raise ValueError(
                "memory tissue must span the longest exposure: raise n_memory or "
                "lower max_memory_tokens"
            )
        self.organism_config(device="cpu").validate()

    def scaled(self, **overrides) -> "Fable2Config":
        cfg = dataclasses.replace(self, **overrides)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["depth_fractions"] = list(self.depth_fractions)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(payload: dict) -> "Fable2Config":
        payload = dict(payload)
        payload["depth_fractions"] = tuple(payload["depth_fractions"])
        cfg = Fable2Config(**payload)
        cfg.validate()
        return cfg

    @staticmethod
    def from_json(text: str) -> "Fable2Config":
        return Fable2Config.from_dict(json.loads(text))
