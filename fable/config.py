"""Fable configuration: one dataclass, no flag graveyard.

Everything sol/grok exposed as a mechanism toggle is either always-on here
(because it earned status) or absent (because it did not). The only knobs are
geometry, optimization, and evaluation cadence.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass


@dataclass
class FableConfig:
    # Geometry
    n_cells: int = 128
    hidden: int = 128
    n_dendrites: int = 12          # incoming edges per cell (K)
    n_input: int = 16              # sensory cells (stream-driven)
    n_output: int = 16             # readout cells
    n_mirror: int = 32             # mirror ring: recent stimulus, write-only, stop-grad
    steps_per_token: int = 4       # substrate ticks per character
    # Cell rule on trial via F3 (fable/experiments/f3-liquid-cell.md):
    # "gru" (incumbent) or "liquid" (CfC-style bounded input-dependent tau).
    # This field exists WITH a preregistration, per config.md's rule.
    cell_rule: str = "gru"
    # Per-cell expression vectors on trial via F8
    # (fable/experiments/f8-expression.md): zero-init per-cell affine on the
    # shared rule's message input — parametric cell identity ("gene
    # expression" over a shared genome). False = today's 12-scalars-per-cell
    # organism, whose measured shuffle delta is ~0.02 because cells are
    # interchangeable by construction.
    expression: bool = False
    # Associative memory organ on trial via F9a
    # (fable/experiments/f9-memory-organ.md): ring-write/content-read memory
    # whose reads drive a dedicated memory-port cell group; output-gated to
    # zero at init.
    memory: bool = False

    # Stream
    vocab_size: int = 0            # filled from corpus
    batch_size: int = 12           # parallel lanes
    chunk_length: int = 32         # tokens per BPTT window
    holdout_fraction: float = 0.10 # contiguous tail split

    # Optimization (identical schedule for creature and GRU — sol S12's lesson,
    # applied to *both* arms, which sol never did)
    lr: float = 3e-3
    lr_min: float = 3e-4
    warmup_updates: int = 200
    updates: int = 8000            # total; also the cosine horizon
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Transformer control only. It has no persistent state, so its context is
    # whatever sits in the window. Trained and evaluated at exactly this
    # length; batch is rescaled so tokens/update matches the creature's
    # batch_size × chunk_length.
    transformer_seq_len: int = 128

    # Evaluation
    eval_every: int = 500
    log_every: int = 100
    eval_tokens: int = 2048
    eval_warmup_tokens: int = 256

    # Misc
    seed: int = 7
    device: str = "cpu"

    def scaled(self, **overrides) -> "FableConfig":
        return dataclasses.replace(self, **overrides)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "FableConfig":
        return FableConfig(**json.loads(text))
