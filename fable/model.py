"""The fable organism: ports, mirror ring, tick loop, concat readout.

Cell blocks (by index buffer, never by layout assumption — growth appends
cells, so contiguity of blocks is not an invariant):
  input    — driven by the current token's embedding at micro-step 0
  mirror   — FIFO ring of recent raw token embeddings; write-only from the
             stream, detached (the BYOL-collapse guard from the DMON
             architecture docs), never updated by the rule, and excluded from
             rule compute entirely (grok computed then discarded them — 25%
             waste, debt #2)
  internal — free tissue; the only block growth extends
  output   — read by the concat readout; pure sinks (no cell reads them)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cell import SharedRule
from .config import FableConfig
from .graph import DendriteGraph


@dataclass
class OrganismState:
    h: torch.Tensor      # [B, N, H]
    mirror_cursor: int

    def detach(self) -> "OrganismState":
        return OrganismState(self.h.detach(), self.mirror_cursor)


@dataclass
class StepHealth:
    h_max: float
    msg_rms: float        # PRE-clamp message RMS — the number that diagnoses scale drift
    logit_absmax: float


class MessageClamp(nn.Module):
    """Scale messages down to unit RMS when they exceed it; identity below.

    Deliberately not RMSNorm: full normalization multiplies small signals —
    and their gradients — by 1/rms, and that amplification compounds across
    the T×spt (=128) sequential micro-steps of a chunk's backward pass. The
    first F0 launch died exactly that way (gradients permanently inf from
    ~u875 with RMSNorm in this slot). The clamp only ever damps.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # eps inside the sqrt: at x == 0 (every message on the first
        # micro-step) sqrt' is inf and clamp's zero-gradient branch turns
        # 0 * inf into NaN for every upstream parameter
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-12).sqrt()
        return x / rms.clamp(min=1.0)


class Fable(nn.Module):
    def __init__(self, cfg: FableConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "fill vocab_size from the corpus first"
        n_ports = cfg.n_input + cfg.n_output + cfg.n_mirror
        assert n_ports <= cfg.n_cells, f"ports ({n_ports}) exceed field ({cfg.n_cells})"
        self.cfg = cfg
        n, hid = cfg.n_cells, cfg.hidden

        idx = torch.arange(n)
        self.register_buffer("input_idx", idx[: cfg.n_input].clone())
        self.register_buffer("mirror_idx", idx[cfg.n_input: cfg.n_input + cfg.n_mirror].clone())
        self.register_buffer("internal_idx", idx[cfg.n_input + cfg.n_mirror: n - cfg.n_output].clone())
        self.register_buffer("output_idx", idx[n - cfg.n_output:].clone())
        self._rebuild_mutable()

        self.embed = nn.Embedding(cfg.vocab_size, hid)
        nn.init.normal_(self.embed.weight, std=0.16)
        # Sensory frontend: shared embedding, per-input-cell affine identity.
        # grok's input_proj was a linear-on-linear frontend eating 31-60% of
        # the parameter budget (debt #21); this is 2·I·H params instead.
        self.in_gain = nn.Parameter(torch.ones(cfg.n_input, hid))
        self.in_bias = nn.Parameter(torch.zeros(cfg.n_input, hid))

        self.graph = DendriteGraph(n, hid, cfg.n_dendrites,
                                   self.input_idx, self.mirror_idx,
                                   self.internal_idx, self.output_idx,
                                   seed=cfg.seed)
        self.rule = SharedRule(hid)
        self.msg_clamp = MessageClamp()

        self.out_norm = nn.LayerNorm(cfg.n_output * hid)
        self.readout = nn.Linear(cfg.n_output * hid, cfg.vocab_size)

    def _rebuild_mutable(self) -> None:
        mutable = torch.cat([self.input_idx, self.internal_idx, self.output_idx])
        if hasattr(self, "mutable_idx"):
            self.mutable_idx = mutable.to(self.mutable_idx.device)
        else:
            self.register_buffer("mutable_idx", mutable)

    @property
    def n_cells(self) -> int:
        return len(self.input_idx) + len(self.mirror_idx) + len(self.internal_idx) + len(self.output_idx)

    def initial_state(self, batch: int, device: str | torch.device) -> OrganismState:
        return OrganismState(
            h=torch.zeros(batch, self.n_cells, self.cfg.hidden, device=device),
            mirror_cursor=0)

    def step(self, tokens: torch.Tensor, state: OrganismState,
             frozen_idx: torch.Tensor | None = None,
             collect_health: bool = False,
             ) -> tuple[torch.Tensor, OrganismState, StepHealth | None]:
        """One token for every lane: mirror write, spt micro-steps, readout.

        tokens: [B] long. Returns (logits [B, V], state, health or None).
        """
        cfg = self.cfg
        h = state.h
        emb = self.embed(tokens)                                    # [B, H]

        slot = self.mirror_idx[state.mirror_cursor % len(self.mirror_idx)]
        h = h.index_copy(1, slot.reshape(1), emb.detach().unsqueeze(1))

        n_in = len(self.input_idx)
        drive_in = emb.unsqueeze(1) * self.in_gain + self.in_bias   # [B, I, H]

        msg = None
        raw_msg = None
        for micro in range(cfg.steps_per_token):
            raw_msg = self.graph.aggregate(h, self.mutable_idx)
            msg = self.msg_clamp(raw_msg)
            drive = torch.zeros_like(msg)
            if micro == 0:
                drive = drive.index_copy(
                    1, torch.arange(n_in, device=drive.device), drive_in)
            h_mut = h[:, self.mutable_idx]
            h_new = self.rule(h_mut, msg, drive)
            h = h.index_copy(1, self.mutable_idx, h_new)
            if frozen_idx is not None and len(frozen_idx) > 0:
                zeros = torch.zeros(h.shape[0], len(frozen_idx), h.shape[2],
                                    device=h.device, dtype=h.dtype)
                h = h.index_copy(1, frozen_idx, zeros)

        out = h[:, self.output_idx]                                 # [B, O, H]
        logits = self.readout(self.out_norm(out.reshape(out.shape[0], -1)))

        health = None
        if collect_health:
            with torch.no_grad():
                health = StepHealth(
                    h_max=float(h.abs().max()),
                    msg_rms=float(raw_msg.pow(2).mean().sqrt()),
                    logit_absmax=float(logits.abs().max()))
        return logits, OrganismState(h, state.mirror_cursor + 1), health

    def forward_chunk(self, tokens: torch.Tensor, targets: torch.Tensor,
                      state: OrganismState,
                      ) -> tuple[torch.Tensor, OrganismState, StepHealth | None]:
        """tokens/targets: [B, T]. Returns (mean CE loss, state, last health)."""
        losses = []
        health = None
        last = tokens.shape[1] - 1
        for t in range(tokens.shape[1]):
            logits, state, health = self.step(
                tokens[:, t], state, collect_health=(t == last))
            losses.append(F.cross_entropy(logits, targets[:, t]))
        return torch.stack(losses).mean(), state, health


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
