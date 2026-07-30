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
