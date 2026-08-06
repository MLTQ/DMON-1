"""Typed persistent SOL2 organism with bounded dynamics and private cell identity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import Sol2Config
from .graph import DirectedConnectome
from .organs import AttentiveOutputOrgan, SensorEffectorOrgan
from .relay_output_transport import RelayOutputAttention
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
    organ_attention_entropy: float
    organ_attention_max: float
    organ_effective_cells: float


@dataclass
class StepTrace:
    """Opt-in live observations from the exact transition graph."""

    recalled: torch.Tensor | None = None
    recall_memory: torch.Tensor | None = None


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
            self.internal_idx,
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
            self.cell_gain = nn.Parameter(torch.zeros(cfg.n_mutable, hidden))
            self.cell_bias = nn.Parameter(torch.zeros(cfg.n_mutable, hidden))
        else:
            self.cell_gain = None
            self.cell_bias = None

        if cfg.cell_adapter_rank > 0:
            rank = cfg.cell_adapter_rank
            self.cell_adapter_down = nn.Parameter(
                torch.empty(cfg.n_mutable, 3 * hidden, rank)
            )
            # Keep the zero-up adapter silent without perturbing matched organ
            # initialization through consumption of the global RNG stream.
            adapter_generator = torch.Generator().manual_seed(cfg.seed + 42_000)
            nn.init.normal_(
                self.cell_adapter_down,
                std=(3 * hidden) ** -0.5,
                generator=adapter_generator,
            )
            self.cell_adapter_up = nn.Parameter(
                torch.zeros(cfg.n_mutable, rank, hidden)
            )
        else:
            self.cell_adapter_down = None
            self.cell_adapter_up = None

        self.output_organ = AttentiveOutputOrgan(
            hidden,
            cfg.vocab_size,
            cfg.organ_queries,
            bounded_operators=cfg.bounded_operators,
            operator_bound=cfg.operator_bound,
            value_gain=cfg.value_gain,
            attention_temperature=cfg.attention_temperature,
        )
        self.relay_output_transport = None
        if cfg.relay_output_attention:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed + 73_000)
                self.relay_output_transport = RelayOutputAttention(
                    cfg.n_output,
                    hidden,
                    bounded_operators=cfg.bounded_operators,
                    operator_bound=cfg.operator_bound,
                    value_gain=cfg.value_gain,
                    attention_temperature=cfg.attention_temperature,
                    transport_gain=cfg.relay_output_gain,
                    initial_gate=cfg.relay_output_initial_gate,
                )
        # Organ A keeps the legacy parameter names so acquisition checkpoints load
        # strictly. Newly attached ports live here and are always selected explicitly.
        self.attached_organs = nn.ModuleDict()

    @staticmethod
    def _validate_attached_organ_name(name: str) -> None:
        if not name or name == "A" or "." in name:
            raise ValueError("attached organ name must be non-empty, non-'A', and dot-free")

    def attach_organ(self, name: str, *, seed: int) -> SensorEffectorOrgan:
        """Create a deterministic independent sensor/effector bundle."""

        self._validate_attached_organ_name(name)
        if name in self.attached_organs:
            raise ValueError(f"organ {name!r} is already attached")
        device = self.embedding.weight.device
        cuda_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            organ = SensorEffectorOrgan(
                len(self.input_idx),
                self.cfg.hidden,
                self.cfg.vocab_size,
                self.cfg.organ_queries,
                bounded_operators=self.cfg.bounded_operators,
                operator_bound=self.cfg.operator_bound,
                value_gain=self.cfg.value_gain,
                attention_temperature=self.cfg.attention_temperature,
            ).to(device)
        self.attached_organs[name] = organ
        return organ

    def attach_organ_module(self, name: str, organ: nn.Module) -> nn.Module:
        """Attach a specialized sensor/effector that obeys the organ protocol."""

        self._validate_attached_organ_name(name)
        if name in self.attached_organs:
            raise ValueError(f"organ {name!r} is already attached")
        missing = [
            method
            for method in ("sense", "read", "operator_norms")
            if not callable(getattr(organ, method, None))
        ]
        if missing:
            raise TypeError(f"organ is missing required methods: {', '.join(missing)}")
        organ.to(self.embedding.weight.device)
        self.attached_organs[name] = organ
        return organ

    def detach_organ(self, name: str) -> nn.Module:
        """Remove and return an attached bundle without changing its parameters."""

        if name == "A":
            raise ValueError("the checkpoint-compatible primary organ cannot be detached")
        if name not in self.attached_organs:
            raise KeyError(f"organ {name!r} is not attached")
        return self.attached_organs.pop(name)

    def reattach_organ(self, name: str, organ: nn.Module) -> None:
        """Restore the exact previously detached module object."""

        self.attach_organ_module(name, organ)

    def organ_parameters(self, name: str):
        """Iterate the parameters belonging only to one physical interface."""

        if name == "A":
            yield from self.embedding.parameters()
            yield self.sensory_gain
            yield self.sensory_bias
            yield from self.output_organ.parameters()
            return
        if name not in self.attached_organs:
            raise KeyError(f"organ {name!r} is not attached")
        yield from self.attached_organs[name].parameters()

    def _sense(
        self, tokens: torch.Tensor, organ_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if organ_name == "A":
            embedded = torch.tanh(self.embedding(tokens))
            sensory_gain = 1.0 + 0.5 * torch.tanh(self.sensory_gain)
            sensory_bias = 0.1 * torch.tanh(self.sensory_bias)
            return embedded, embedded.unsqueeze(1) * sensory_gain + sensory_bias
        if organ_name not in self.attached_organs:
            raise KeyError(f"organ {organ_name!r} is not attached")
        return self.attached_organs[organ_name].sense(tokens)

    def _read_output(self, hidden: torch.Tensor, organ_name: str, collect_health: bool):
        output_cells = hidden[:, self.output_idx]
        if organ_name == "A":
            return self.output_organ(output_cells, collect_health=collect_health)
        if organ_name not in self.attached_organs:
            raise KeyError(f"organ {organ_name!r} is not attached")
        return self.attached_organs[organ_name].read(
            output_cells, collect_health=collect_health
        )

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

        # Private-expression rows follow physical mutable-cell order. Memory owns no
        # learned rule and therefore receives no dead identity parameter. Relay is
        # physically last, so new relay expression rows append without moving organs.
        expression_cells = torch.cat(
            [self.input_idx, self.compute_idx, self.output_idx, self.relay_idx]
        )
        expression_pos = torch.full(
            (self.n_cells,), -1, dtype=torch.long, device=expression_cells.device
        )
        expression_pos[expression_cells] = torch.arange(
            len(expression_cells), device=expression_cells.device
        )
        for name, value in (
            ("expression_cells", expression_cells),
            ("expression_pos", expression_pos),
        ):
            if hasattr(self, name):
                setattr(self, name, value.to(getattr(self, name).device))
            else:
                self.register_buffer(name, value)

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

    def _private_expression(
        self, cells: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.cell_gain is None:
            return None, None
        positions = self.expression_pos[cells]
        if bool((positions < 0).any()):
            raise ValueError("stream-written memory has no private expression")
        gain = torch.tanh(self.cell_gain[positions]).unsqueeze(0)
        bias = torch.tanh(self.cell_bias[positions]).unsqueeze(0)
        target_shift = self.cfg.identity_gain * gain + self.cfg.identity_bias * bias
        alpha_shift = self.cfg.identity_time * (gain - bias)
        return target_shift, alpha_shift

    def _private_adapter(
        self, cells: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.cell_adapter_down is None:
            return None, None
        positions = self.expression_pos[cells]
        if bool((positions < 0).any()):
            raise ValueError("stream-written memory has no private transition adapter")
        return self.cell_adapter_down[positions], self.cell_adapter_up[positions]

    def step(
        self,
        tokens: torch.Tensor,
        state: OrganismState,
        *,
        organ_name: str = "A",
        frozen_idx: torch.Tensor | None = None,
        write_memory: bool = True,
        collect_health: bool = False,
        trace: StepTrace | None = None,
    ) -> tuple[torch.Tensor, OrganismState, StepHealth | None]:
        cfg = self.cfg
        hidden = state.hidden
        if hidden.shape[1:] != (self.n_cells, cfg.hidden):
            raise ValueError("state shape does not match the current organism")
        if trace is not None:
            trace.recalled = None
            trace.recall_memory = None

        # Output tissue is a per-token interface, not an autonomous recurrent model.
        # It may integrate over microsteps, but it cannot carry a private sequence
        # model across token boundaries and bypass internal tissue.
        output_zeros = hidden.new_zeros(
            hidden.shape[0], len(self.output_idx), hidden.shape[2]
        )
        hidden = hidden.index_copy(1, self.output_idx, output_zeros)

        if frozen_idx is not None and len(frozen_idx):
            frozen_zeros = hidden.new_zeros(
                hidden.shape[0], len(frozen_idx), hidden.shape[2]
            )
            hidden = hidden.index_copy(1, frozen_idx, frozen_zeros)

        # Embedding is bounded before entering persistent state. This is a kernel
        # invariant, not one arm of the graph-operator factorial.
        embedded, sensory_drive = self._sense(tokens, organ_name)
        if write_memory:
            memory_slot = self.memory_idx[state.memory_cursor % len(self.memory_idx)]
            hidden = hidden.index_copy(
                1, memory_slot.reshape(1), embedded.unsqueeze(1)
            )
        elif organ_name != "A":
            recall = getattr(self.attached_organs[organ_name], "recall", None)
            if callable(recall):
                recall_memory, recall_count = self._chronological_memory(
                    hidden, state.memory_cursor
                )
                recalled, _ = recall(
                    embedded,
                    recall_memory,
                    valid_count=recall_count,
                    collect_health=False,
                )
                if trace is not None:
                    trace.recalled = recalled
                    trace.recall_memory = recall_memory
                sensory_drive = sensory_drive + recalled.unsqueeze(1)

        raw_message = None
        message = None
        alpha_means: dict[str, float] = {}
        for micro in range(cfg.steps_per_token):
            raw_message = self.graph.aggregate(hidden, self.mutable_idx)
            message = raw_message
            if self.relay_output_transport is not None:
                transported = self.relay_output_transport(
                    hidden[:, self.relay_idx], hidden[:, self.output_idx]
                )
                message = message.index_copy(
                    1,
                    self.output_pos,
                    message[:, self.output_pos] + transported,
                )
            drive = torch.zeros_like(message)
            if micro == 0:
                drive = drive.index_copy(1, self.input_pos, sensory_drive)

            for name in self.TISSUES:
                cells = getattr(self, f"{name}_idx")
                positions = getattr(self, f"{name}_pos")
                target_shift, alpha_shift = self._private_expression(cells)
                adapter_down, adapter_up = self._private_adapter(cells)
                updated, alpha = self.tissues(
                    name,
                    hidden[:, cells],
                    message[:, positions],
                    drive[:, positions],
                    target_shift=target_shift,
                    alpha_shift=alpha_shift,
                    adapter_down=adapter_down,
                    adapter_up=adapter_up,
                    adapter_gain=cfg.cell_adapter_gain,
                )
                hidden = hidden.index_copy(1, cells, updated)
                if collect_health:
                    alpha_means[name] = float(alpha.detach().mean())

            if frozen_idx is not None and len(frozen_idx):
                zeros = hidden.new_zeros(hidden.shape[0], len(frozen_idx), hidden.shape[2])
                hidden = hidden.index_copy(1, frozen_idx, zeros)

        logits, organ_health = self._read_output(hidden, organ_name, collect_health)

        health = None
        if collect_health:
            organ_norms = (
                self.output_organ.operator_norms()
                if organ_name == "A"
                else self.attached_organs[organ_name].operator_norms()
            )
            norms = {
                **{f"graph.{k}": v for k, v in self.graph.operator_norms().items()},
                **{f"tissue.{k}": v for k, v in self.tissues.operator_norms().items()},
                **{f"organ.{organ_name}.{k}": v for k, v in organ_norms.items()},
            }
            if self.relay_output_transport is not None:
                norms.update(
                    {
                        f"relay_output.{k}": v
                        for k, v in self.relay_output_transport.operator_norms().items()
                    }
                )
            assert organ_health is not None
            health = StepHealth(
                hidden_absmax=float(hidden.detach().abs().max()),
                message_rms=float(raw_message.detach().pow(2).mean().sqrt()),
                message_absmax=float(message.detach().abs().max()),
                logit_absmax=float(logits.detach().abs().max()),
                alpha_mean=alpha_means,
                operator_norm_max=max(norms.values()),
                organ_attention_entropy=organ_health.attention_entropy,
                organ_attention_max=organ_health.attention_max,
                organ_effective_cells=organ_health.effective_cells,
            )

        return (
            logits,
            OrganismState(
                hidden=hidden,
                memory_cursor=state.memory_cursor + int(write_memory),
                weight_version=state.weight_version,
            ),
            health,
        )

    def _chronological_memory(
        self, hidden: torch.Tensor, memory_cursor: int
    ) -> tuple[torch.Tensor, int]:
        """Return valid FIFO slots oldest-to-newest across circular wraparound."""

        capacity = len(self.memory_idx)
        count = min(max(int(memory_cursor), 0), capacity)
        memory = hidden[:, self.memory_idx]
        if count < capacity:
            return memory[:, :count], count
        start = memory_cursor % capacity
        order = (torch.arange(capacity, device=hidden.device) + start) % capacity
        return memory.index_select(1, order), count

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        state: OrganismState,
        *,
        organ_name: str = "A",
        collect_health: bool = False,
    ) -> tuple[torch.Tensor, OrganismState, StepHealth | None]:
        losses = []
        health = None
        for position in range(tokens.shape[1]):
            logits, state, health = self.step(
                tokens[:, position],
                state,
                organ_name=organ_name,
                collect_health=collect_health and position == tokens.shape[1] - 1,
            )
            losses.append(F.cross_entropy(logits, targets[:, position]))
        return torch.stack(losses).mean(), state, health


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
