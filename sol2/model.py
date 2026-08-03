"""Typed persistent SOL2 organism with bounded dynamics and private cell identity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import Sol2Config
from .graph import DirectedConnectome
from .state import OrganismState
from .tissues import TissueBank


@dataclass
class StepHealth:
    hidden_absmax: float
    message_rms: float
    message_absmax: float
    logit_absmax: float
    alpha_mean: dict[str, float]
    operator_norm_max: float


class Sol2(nn.Module):
    TISSUES = ("input", "compute", "relay", "output")

    def __init__(self, cfg: Sol2Config):
        super().__init__()
        cfg.validate()
        if cfg.vocab_size <= 0:
            raise ValueError("fill vocab_size from the corpus before building SOL2")
        self.cfg = cfg
        self._build_index_buffers()

        hidden = cfg.hidden
        self.embedding = nn.Embedding(cfg.vocab_size, hidden)
        nn.init.normal_(self.embedding.weight, std=0.16)
        self.sensory_gain = nn.Parameter(torch.zeros(cfg.n_input, hidden))
        self.sensory_bias = nn.Parameter(torch.zeros(cfg.n_input, hidden))

        source_pool = torch.cat(
            [self.input_idx, self.memory_idx, self.compute_idx, self.relay_idx]
        )
        self.graph = DirectedConnectome(
            cfg.n_cells,
            hidden,
            cfg.n_dendrites,
            cfg.initial_active_dendrites,
            source_pool,
            self.input_idx,
            self.output_idx,
            seed=cfg.seed,
            bounded_operators=cfg.bounded_operators,
            operator_bound=cfg.operator_bound,
            value_gain=cfg.value_gain,
            attention_temperature=cfg.attention_temperature,
            edge_logit_limit=cfg.edge_logit_limit,
        )
        self.tissues = TissueBank(
            hidden,
            bounded_operators=cfg.bounded_operators,
            operator_bound=cfg.operator_bound,
        )

        if cfg.cell_identity:
            self.cell_gain = nn.Parameter(torch.zeros(cfg.n_cells, hidden))
            self.cell_bias = nn.Parameter(torch.zeros(cfg.n_cells, hidden))
        else:
            self.cell_gain = None
            self.cell_bias = None

        output_width = cfg.n_output * hidden
        self.output_norm = nn.LayerNorm(output_width)
        self.readout = nn.Linear(output_width, cfg.vocab_size)

    def _build_index_buffers(self) -> None:
        cfg = self.cfg
        cursor = 0
        ranges: dict[str, torch.Tensor] = {}
        for name, count in (
            ("input", cfg.n_input),
            ("memory", cfg.n_memory),
            ("compute", cfg.n_compute),
            ("output", cfg.n_output),
            # Relay tissue is last so transport growth can append cells without
            # moving any existing organ or making checkpoint layout implicit.
            ("relay", cfg.n_relay),
        ):
            ranges[name] = torch.arange(cursor, cursor + count, dtype=torch.long)
            cursor += count
        for name, value in ranges.items():
            attr = f"{name}_idx"
            if hasattr(self, attr):
                setattr(self, attr, value.to(getattr(self, attr).device))
            else:
                self.register_buffer(attr, value)
        self._rebuild_mutable()

    def _rebuild_mutable(self) -> None:
        mutable = torch.cat(
            [self.input_idx, self.compute_idx, self.relay_idx, self.output_idx]
        )
        if hasattr(self, "mutable_idx"):
            self.mutable_idx = mutable.to(self.mutable_idx.device)
        else:
            self.register_buffer("mutable_idx", mutable)

        cursor = 0
        for name in self.TISSUES:
            count = len(getattr(self, f"{name}_idx"))
            value = torch.arange(cursor, cursor + count, dtype=torch.long)
            attr = f"{name}_pos"
            if hasattr(self, attr):
                setattr(self, attr, value.to(getattr(self, attr).device))
            else:
                self.register_buffer(attr, value)
            cursor += count

    @property
    def n_cells(self) -> int:
        return int(
            len(self.input_idx)
            + len(self.memory_idx)
            + len(self.compute_idx)
            + len(self.relay_idx)
            + len(self.output_idx)
        )

    @property
    def internal_idx(self) -> torch.Tensor:
        return torch.cat([self.compute_idx, self.relay_idx])

    def tissue_indices(self, name: str) -> torch.Tensor:
        if name not in (*self.TISSUES, "memory", "internal"):
            raise KeyError(f"unknown tissue {name!r}")
        return self.internal_idx if name == "internal" else getattr(self, f"{name}_idx")

    def initial_state(self, batch: int, device: str | torch.device) -> OrganismState:
        return OrganismState(
            hidden=torch.zeros(batch, self.n_cells, self.cfg.hidden, device=device),
            memory_cursor=0,
            weight_version=0,
        )

    def _apply_identity(
        self, message: torch.Tensor, mutable_cells: torch.Tensor
    ) -> torch.Tensor:
        if self.cell_gain is None:
            return message
        gain = 1.0 + self.cfg.identity_gain * torch.tanh(
            self.cell_gain[mutable_cells]
        ).unsqueeze(0)
        bias = self.cfg.identity_bias * torch.tanh(
            self.cell_bias[mutable_cells]
        ).unsqueeze(0)
        return message * gain + bias

    def step(
        self,
        tokens: torch.Tensor,
        state: OrganismState,
        *,
        frozen_idx: torch.Tensor | None = None,
        collect_health: bool = False,
    ) -> tuple[torch.Tensor, OrganismState, StepHealth | None]:
        cfg = self.cfg
        hidden = state.hidden
        if hidden.shape[1:] != (self.n_cells, cfg.hidden):
            raise ValueError("state shape does not match the current organism")

        # Embedding is bounded before entering persistent state. This is a kernel
        # invariant, not one arm of the graph-operator factorial.
        embedded = torch.tanh(self.embedding(tokens))
        memory_slot = self.memory_idx[state.memory_cursor % len(self.memory_idx)]
        hidden = hidden.index_copy(
            1, memory_slot.reshape(1), embedded.detach().unsqueeze(1)
        )

        sensory_gain = 1.0 + 0.5 * torch.tanh(self.sensory_gain)
        sensory_bias = 0.1 * torch.tanh(self.sensory_bias)
        sensory_drive = embedded.unsqueeze(1) * sensory_gain + sensory_bias

        raw_message = None
        message = None
        alpha_means: dict[str, float] = {}
        for micro in range(cfg.steps_per_token):
            raw_message = self.graph.aggregate(hidden, self.mutable_idx)
            message = self._apply_identity(raw_message, self.mutable_idx)
            drive = torch.zeros_like(message)
            if micro == 0:
                drive = drive.index_copy(1, self.input_pos, sensory_drive)

            for name in self.TISSUES:
                cells = getattr(self, f"{name}_idx")
                positions = getattr(self, f"{name}_pos")
                updated, alpha = self.tissues(
                    name,
                    hidden[:, cells],
                    message[:, positions],
                    drive[:, positions],
                )
                hidden = hidden.index_copy(1, cells, updated)
                if collect_health:
                    alpha_means[name] = float(alpha.detach().mean())

            if frozen_idx is not None and len(frozen_idx):
                zeros = hidden.new_zeros(hidden.shape[0], len(frozen_idx), hidden.shape[2])
                hidden = hidden.index_copy(1, frozen_idx, zeros)

        output = hidden[:, self.output_idx].reshape(hidden.shape[0], -1)
        logits = self.readout(self.output_norm(output))

        health = None
        if collect_health:
            norms = {
                **{f"graph.{k}": v for k, v in self.graph.operator_norms().items()},
                **{f"tissue.{k}": v for k, v in self.tissues.operator_norms().items()},
            }
            health = StepHealth(
                hidden_absmax=float(hidden.detach().abs().max()),
                message_rms=float(raw_message.detach().pow(2).mean().sqrt()),
                message_absmax=float(message.detach().abs().max()),
                logit_absmax=float(logits.detach().abs().max()),
                alpha_mean=alpha_means,
                operator_norm_max=max(norms.values()),
            )

        return (
            logits,
            OrganismState(
                hidden=hidden,
                memory_cursor=state.memory_cursor + 1,
                weight_version=state.weight_version,
            ),
            health,
        )

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        state: OrganismState,
        *,
        collect_health: bool = False,
    ) -> tuple[torch.Tensor, OrganismState, StepHealth | None]:
        losses = []
        health = None
        for position in range(tokens.shape[1]):
            logits, state, health = self.step(
                tokens[:, position],
                state,
                collect_health=collect_health and position == tokens.shape[1] - 1,
            )
            losses.append(F.cross_entropy(logits, targets[:, position]))
        return torch.stack(losses).mean(), state, health


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
