"""Streaming creature: directed-dendrite field with mirror memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .cell import SharedGRURule
from .config import TrainConfig
from .graph import DendriteGraph


@dataclass(slots=True)
class CreatureState:
    """Persistent electrical state carried across stream tokens."""

    h: torch.Tensor  # [B, N, H]
    mirror_cursor: int

    def detach(self) -> "CreatureState":
        return CreatureState(self.h.detach(), self.mirror_cursor)

    def clone_detach(self) -> "CreatureState":
        return CreatureState(self.h.detach().clone(), self.mirror_cursor)


class StreamingCreature(nn.Module):
    """Character-level streaming network on a dendrite graph.

    Port layout (contiguous blocks for clarity):
      [0, n_input)              — input cells (stream-written + mutable)
      [n_input, +n_mirror)      — mirror cells (stream-written ring ONLY)
      [.., +n_internal)         — internal computation
      [.., +n_output)           — output cells (readout)

    Mirror contract (architecture §2):
      - Stream writes recent stimulus embeddings into the mirror ring.
      - The shared rule cannot overwrite mirror state (masked update).
      - Dendrites may read mirrors, so past events remain addressable when
        later loss arrives — continuous reward needs a memory of the event.
    """

    def __init__(self, config: TrainConfig, vocab_size: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vocab_size = vocab_size
        h = config.hidden
        n = config.n_cells

        self.register_buffer("input_idx", torch.arange(0, config.n_input), persistent=False)
        self.register_buffer(
            "mirror_idx",
            torch.arange(config.n_input, config.n_input + config.n_mirror),
            persistent=False,
        )
        self.register_buffer(
            "internal_idx",
            torch.arange(
                config.n_input + config.n_mirror,
                config.n_input + config.n_mirror + config.n_internal,
            ),
            persistent=False,
        )
        self.register_buffer(
            "output_idx", torch.arange(n - config.n_output, n), persistent=False
        )

        mutable = torch.ones(n, dtype=torch.bool)
        mutable[self.mirror_idx] = False
        self.register_buffer("mutable_mask", mutable, persistent=True)

        self.embed = nn.Embedding(vocab_size, h)
        self.graph = DendriteGraph(
            n_cells=n,
            n_dendrites=config.n_dendrites,
            hidden=h,
            input_idx=self.input_idx,
            output_idx=self.output_idx,
            mirror_idx=self.mirror_idx,
            internal_idx=self.internal_idx,
            seed=config.seed,
            use_attention=config.use_attention,
        )
        self.rule = SharedGRURule(h)
        # Distributed input: spread char vector across input ports.
        self.input_proj = nn.Linear(h, config.n_input * h)
        # Readout: mean-pool output cells then project (scales better than concat).
        self.out_norm = nn.LayerNorm(h)
        self.readout = nn.Linear(h, vocab_size)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> CreatureState:
        if device is None:
            device = self.embed.weight.device
        h = torch.zeros(
            batch_size, self.config.n_cells, self.config.hidden, device=device
        )
        return CreatureState(h=h, mirror_cursor=0)

    def step(
        self,
        token_ids: torch.Tensor,
        state: CreatureState,
    ) -> tuple[torch.Tensor, CreatureState]:
        """Consume one token per batch row; return next-token logits and new state.

        token_ids: [B] long
        logits: [B, vocab]
        """

        b = token_ids.shape[0]
        h = state.h
        if h.shape[0] != b:
            raise ValueError(f"state batch {h.shape[0]} != tokens batch {b}")

        emb = self.embed(token_ids)  # [B, H]
        drive_full = torch.zeros_like(h)
        spread = self.input_proj(emb).view(b, self.config.n_input, self.config.hidden)
        drive_full = drive_full.clone()
        drive_full[:, self.input_idx, :] = spread

        # --- Mirror write (stream only, no rule) ---
        mirror_write = emb.detach()
        h = h.clone()
        slot = self.mirror_idx[state.mirror_cursor % self.config.n_mirror]
        h[:, slot, :] = mirror_write
        next_cursor = (state.mirror_cursor + 1) % self.config.n_mirror

        # --- Recurrent microsteps over the dendrite graph ---
        for micro in range(self.config.steps_per_token):
            messages = self.graph.aggregate(h)
            drive = drive_full if micro == 0 else torch.zeros_like(drive_full)
            proposed = self.rule(h, messages, drive)
            mask = self.mutable_mask.view(1, -1, 1)
            h = torch.where(mask, proposed, h)

        out_states = h[:, self.output_idx, :].mean(dim=1)
        logits = self.readout(self.out_norm(out_states))
        return logits, CreatureState(h=h, mirror_cursor=next_cursor)

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        state: CreatureState | None = None,
        *,
        truncate_every: int | None = None,
    ) -> tuple[torch.Tensor, CreatureState, torch.Tensor]:
        """Run a contiguous sequence; return logits [B,T,V], final state, mean CE."""

        if tokens.dim() != 2:
            raise ValueError("tokens must be [B, T]")
        b, t = tokens.shape
        device = tokens.device
        if state is None:
            state = self.initial_state(b, device)

        logits_list = []
        losses = []
        for i in range(t - 1):
            logits, state = self.step(tokens[:, i], state)
            logits_list.append(logits)
            losses.append(F.cross_entropy(logits, tokens[:, i + 1]))
            if truncate_every is not None and (i + 1) % truncate_every == 0:
                state = state.detach()
        stacked = torch.stack(logits_list, dim=1)
        loss = torch.stack(losses).mean()
        return stacked, state, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
