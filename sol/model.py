"""Persistent character-stream neural field with sparse directed communication.

This is NCA-like in the sense that every cell applies one shared recurrent rule and
retains private state. It deliberately rejects convolutional neighbor exchange:
targets own a small number of explicit dendrite slots, each naming one source cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SolConfig:
    """Architecture and metabolic constants for one character organ."""

    vocab_size: int
    cells: int = 32
    channels: int = 32
    dendrites: int = 4
    sensory_cells: int = 4
    output_cells: int = 4
    message_steps: int = 3
    topology_seed: int = 0

    eligibility_decay: float = 0.92
    edge_eligibility_decay: float = 0.96
    sensory_trace_decay: float = 0.90
    stimulation_decay: float = 0.72
    reward_gain: float = 0.25
    reward_baseline_decay: float = 0.99
    fast_weight_decay: float = 0.995
    fast_plasticity_gain: float = 0.04
    fast_weight_limit: float = 0.25
    structural_probe_gain: float = 0.0

    energy_start: float = 0.60
    basal_cost: float = 0.002
    activity_cost: float = 0.003
    stimulation_gain: float = 0.025

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if self.cells < 4:
            raise ValueError("cells must be at least four")
        if not 1 <= self.dendrites <= self.cells:
            raise ValueError("dendrites must be in [1, cells]")
        if not 1 <= self.sensory_cells < self.cells:
            raise ValueError("sensory_cells must be in [1, cells)")
        if not 1 <= self.output_cells < self.cells:
            raise ValueError("output_cells must be in [1, cells)")
        if self.sensory_cells + self.output_cells > self.cells:
            raise ValueError("sensory and output populations must not overlap")
        if self.message_steps < 1:
            raise ValueError("message_steps must be positive")
        for name in (
            "eligibility_decay",
            "edge_eligibility_decay",
            "sensory_trace_decay",
            "stimulation_decay",
            "reward_baseline_decay",
            "fast_weight_decay",
        ):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if (
            self.reward_gain < 0
            or self.fast_plasticity_gain < 0
            or self.structural_probe_gain < 0
        ):
            raise ValueError(
                "reward, fast plasticity, and probe gains must be nonnegative"
            )
        if self.fast_weight_limit <= 0:
            raise ValueError("fast_weight_limit must be positive")
        if self.structural_probe_gain > 0 and self.dendrites >= self.cells:
            raise ValueError(
                "structural probes require at least one non-edge source"
            )


@dataclass
class FieldState:
    """All fast state that persists while a stream is alive."""

    hidden: torch.Tensor
    energy: torch.Tensor
    stimulation: torch.Tensor
    eligibility: torch.Tensor
    edge_eligibility: torch.Tensor
    probe_eligibility: torch.Tensor
    fast_weight: torch.Tensor
    sensory_trace: torch.Tensor
    reward: torch.Tensor
    reward_baseline: torch.Tensor

    def detached(self) -> "FieldState":
        """Cut the reverse-mode graph without resetting the organism."""

        return FieldState(
            hidden=self.hidden.detach(),
            energy=self.energy.detach(),
            stimulation=self.stimulation.detach(),
            eligibility=self.eligibility.detach(),
            edge_eligibility=self.edge_eligibility.detach(),
            probe_eligibility=self.probe_eligibility.detach(),
            fast_weight=self.fast_weight.detach(),
            sensory_trace=self.sensory_trace.detach(),
            reward=self.reward.detach(),
            reward_baseline=self.reward_baseline.detach(),
        )

    def clone(self) -> "FieldState":
        """Copy state for counterfactual probes."""

        return FieldState(
            hidden=self.hidden.clone(),
            energy=self.energy.clone(),
            stimulation=self.stimulation.clone(),
            eligibility=self.eligibility.clone(),
            edge_eligibility=self.edge_eligibility.clone(),
            probe_eligibility=self.probe_eligibility.clone(),
            fast_weight=self.fast_weight.clone(),
            sensory_trace=self.sensory_trace.clone(),
            reward=self.reward.clone(),
            reward_baseline=self.reward_baseline.clone(),
        )


@dataclass
class FieldTrace:
    """Differentiable evidence retained from one streamed sequence."""

    hidden: list[torch.Tensor]
    edge_flow: list[torch.Tensor]
    novelty: list[torch.Tensor]
    mean_energy: list[torch.Tensor]
    mean_fast_weight: list[torch.Tensor]
    mean_edge_eligibility: list[torch.Tensor]
    fast_weight_saturation: list[torch.Tensor]
    mean_probe_eligibility: list[torch.Tensor]
    mean_probe_flow: list[torch.Tensor]
    structural_edge_evidence: list[torch.Tensor]
    structural_probe_evidence: list[torch.Tensor]
    probe_effect: list[torch.Tensor]
    prequential_reward: list[torch.Tensor]

    def cell_credit(self) -> torch.Tensor:
        """Return mean absolute loss gradient per token and cell after backward."""

        rows = []
        for value in self.hidden:
            if value.grad is None:
                raise RuntimeError("backward must run before cell_credit()")
            rows.append(value.grad.abs().mean(dim=(0, 2)))
        return torch.stack(rows)

    def probe_fitness(self) -> torch.Tensor:
        """Estimate each probe's first-order effect on global sequence loss."""

        rows = []
        for hidden, effect in zip(self.hidden, self.probe_effect, strict=True):
            if hidden.grad is None:
                raise RuntimeError("backward must run before probe_fitness()")
            rows.append(
                -(hidden.grad.detach() * effect).mean(dim=(0, 2))
            )
        return torch.stack(rows)


def _directed_sources(cfg: SolConfig) -> torch.Tensor:
    """Build a deterministic sparse small-world graph.

    Local predecessor edges preserve spatial continuity; skip and random edges provide
    axons rather than a convolutional light cone. Every output cell receives one
    sensory axon so the organ is causally connected before topology growth exists.
    """

    generator = torch.Generator().manual_seed(cfg.topology_seed)
    stride = max(2, round(math.sqrt(cfg.cells)))
    rows: list[list[int]] = []
    output_start = cfg.cells - cfg.output_cells

    for target in range(cfg.cells):
        candidates = [
            target,
            (target - 1) % cfg.cells,
            (target - stride) % cfg.cells,
        ]
        if target >= output_start:
            candidates.append((target - output_start) % cfg.sensory_cells)

        while len(candidates) < cfg.dendrites:
            candidates.append(
                int(torch.randint(cfg.cells, (1,), generator=generator).item())
            )

        unique: list[int] = []
        for source in candidates:
            if source not in unique:
                unique.append(source)
        while len(unique) < cfg.dendrites:
            source = int(torch.randint(cfg.cells, (1,), generator=generator).item())
            if source not in unique:
                unique.append(source)
        rows.append(unique[: cfg.dendrites])

    return torch.tensor(rows, dtype=torch.long)


def _initial_probe_sources(sources: torch.Tensor) -> torch.Tensor:
    """Choose one deterministic non-edge source for each target."""

    cells, dendrites = sources.shape
    if dendrites >= cells:
        return torch.zeros(cells, dtype=torch.long)
    candidates: list[int] = []
    for target, row in enumerate(sources.tolist()):
        occupied = {int(source) for source in row}
        for offset in range(1, cells + 1):
            candidate = (target + offset) % cells
            if candidate not in occupied:
                candidates.append(candidate)
                break
    return torch.tensor(candidates, dtype=torch.long)


class SparseAxonField(nn.Module):
    """A homogeneous recurrent cell rule running on a directed sparse graph."""

    def __init__(self, cfg: SolConfig):
        super().__init__()
        self.cfg = cfg
        c, n, k = cfg.channels, cfg.cells, cfg.dendrites

        initial_sources = _directed_sources(cfg)
        self.register_buffer("sources", initial_sources)
        self.register_buffer(
            "probe_sources", _initial_probe_sources(initial_sources)
        )
        self.register_buffer(
            "structural_edge_credit", torch.zeros(n, k)
        )
        self.register_buffer(
            "structural_probe_credit", torch.zeros(n)
        )
        self.register_buffer(
            "structural_probe_fitness", torch.zeros(n)
        )
        self.register_buffer(
            "structural_probe_confirmations",
            torch.zeros(n, dtype=torch.long),
        )
        self.register_buffer(
            "structural_edge_age", torch.zeros(n, k, dtype=torch.long)
        )
        self.register_buffer(
            "total_rewires", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer("sensory_indices", torch.arange(cfg.sensory_cells))
        self.register_buffer(
            "output_indices", torch.arange(n - cfg.output_cells, n)
        )

        roles = torch.zeros(n, 3)
        roles[:, 0] = 1.0
        roles[self.sensory_indices] = torch.tensor([0.0, 1.0, 0.0])
        roles[self.output_indices] = torch.tensor([0.0, 0.0, 1.0])
        self.register_buffer("roles", roles)

        self.token_embedding = nn.Embedding(cfg.vocab_size, c)
        self.sensory_projection = nn.Linear(c, c, bias=False)
        self.cell_identity = nn.Parameter(torch.empty(n, c))
        self.role_projection = nn.Linear(3, c, bias=False)

        self.message_query = nn.Linear(c, c, bias=False)
        self.message_key = nn.Linear(c, c, bias=False)
        self.message_value = nn.Linear(c, c, bias=False)
        self.emit_gate = nn.Linear(c, 1)
        self.edge_weight = nn.Parameter(torch.empty(n, k))
        self.edge_bias = nn.Parameter(torch.zeros(n, k))

        # stimulus, incoming message, reward-tagged eligibility, persistent identity,
        # and current metabolic energy all enter the same shared recurrent rule.
        self.cell_rule = nn.GRUCell(c * 4 + 1, c)
        self.output_norm = nn.LayerNorm(cfg.output_cells * c)
        self.output_readout = nn.Linear(cfg.output_cells * c, cfg.vocab_size)

        nn.init.normal_(self.token_embedding.weight, std=0.16)
        nn.init.normal_(self.cell_identity, std=0.08)
        nn.init.normal_(self.edge_weight, std=0.12)
        nn.init.orthogonal_(self.message_value.weight)
        nn.init.constant_(self.emit_gate.bias, 0.5)

    def initial_state(
        self,
        batch: int,
        device: torch.device | str | None = None,
    ) -> FieldState:
        """Create a resting organism. This is not called at chunk boundaries."""

        if batch < 1:
            raise ValueError("batch must be positive")
        device = device or self.cell_identity.device
        context = self._cell_context().to(device)
        hidden = torch.tanh(context).unsqueeze(0).expand(batch, -1, -1).clone()
        return FieldState(
            hidden=hidden,
            energy=torch.full(
                (batch, self.cfg.cells), self.cfg.energy_start, device=device
            ),
            stimulation=torch.zeros(batch, self.cfg.cells, device=device),
            eligibility=torch.zeros(
                batch, self.cfg.cells, self.cfg.channels, device=device
            ),
            edge_eligibility=torch.zeros(
                batch, self.cfg.cells, self.cfg.dendrites, device=device
            ),
            probe_eligibility=torch.zeros(
                batch, self.cfg.cells, device=device
            ),
            fast_weight=torch.zeros(
                batch, self.cfg.cells, self.cfg.dendrites, device=device
            ),
            sensory_trace=torch.zeros(batch, self.cfg.channels, device=device),
            reward=torch.zeros(batch, device=device),
            reward_baseline=torch.full(
                (batch,), math.log(self.cfg.vocab_size), device=device
            ),
        )

    def state_from_snapshot(
        self,
        payload: Mapping[str, Any],
        batch: int,
        device: torch.device | str,
        lane: int | None = None,
    ) -> FieldState:
        """Restore additive fast state while accepting pre-plasticity checkpoints."""

        device = torch.device(device)
        fallback = self.initial_state(batch, device)
        restored: dict[str, torch.Tensor] = {}
        for field in fields(FieldState):
            value = payload.get(field.name)
            tensor = (
                getattr(fallback, field.name)
                if value is None
                else value.to(device)
            )
            restored[field.name] = (
                tensor if lane is None else tensor[lane : lane + 1]
            )
        return FieldState(**restored)

    def load_compatible_state_dict(
        self,
        payload: Mapping[str, torch.Tensor],
    ) -> None:
        """Load model weights while upgrading pre-structure checkpoints."""

        restored = dict(payload)
        additive = (
            "probe_sources",
            "structural_edge_credit",
            "structural_probe_credit",
            "structural_probe_fitness",
            "structural_probe_confirmations",
            "structural_edge_age",
            "total_rewires",
        )
        missing = {name for name in additive if name not in restored}
        current = self.state_dict()
        for name in missing:
            restored[name] = current[name]
        self.load_state_dict(restored)
        if "probe_sources" in missing:
            self.probe_sources.copy_(_initial_probe_sources(self.sources))

    def _cell_context(self) -> torch.Tensor:
        return self.cell_identity + self.role_projection(self.roles)

    def birth_sources(self) -> torch.Tensor:
        """Reconstruct the deterministic pre-growth connectome."""

        return _directed_sources(self.cfg).to(self.sources.device)

    def _messages(
        self,
        hidden: torch.Tensor,
        stimulation: torch.Tensor,
        fast_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read each target's named dendrites and return message, flow, stimulation."""

        source_hidden = hidden[:, self.sources]  # (B, target, dendrite, channel)
        target_query = self.message_query(hidden).unsqueeze(2)
        source_key = self.message_key(source_hidden)
        score = (
            (target_query * source_key).sum(dim=-1) / math.sqrt(self.cfg.channels)
            + self.edge_bias
        )
        attention = torch.softmax(score, dim=2)
        emission = torch.sigmoid(self.emit_gate(source_hidden)).squeeze(-1)
        signed_weight = torch.tanh(self.edge_weight.unsqueeze(0) + fast_weight)
        coefficient = attention * emission * signed_weight
        value = self.message_value(source_hidden)
        incoming = (coefficient.unsqueeze(-1) * value).sum(dim=2)

        flow = coefficient.abs()
        source_stimulation = stimulation[:, self.sources]
        propagated = (flow * source_stimulation).sum(dim=2)
        return incoming, flow, propagated

    def _probe_message(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one weak exploratory candidate message per target."""

        source_hidden = hidden[:, self.probe_sources]
        if self.cfg.structural_probe_gain == 0:
            return (
                torch.zeros_like(hidden),
                hidden.new_zeros(hidden.shape[:2]),
                source_hidden,
            )
        emission = torch.sigmoid(self.emit_gate(source_hidden)).squeeze(-1)
        flow = self.cfg.structural_probe_gain * emission
        message = flow.unsqueeze(-1) * self.message_value(source_hidden)
        return message, flow, source_hidden

    def _updated_fast_weight(
        self,
        fast_weight: torch.Tensor,
        edge_eligibility: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """Apply reward locally through a smooth homeostatic efficacy bound."""

        cfg = self.cfg
        drive = (
            cfg.fast_weight_decay * fast_weight
            + cfg.fast_plasticity_gain
            * reward[:, None, None]
            * edge_eligibility
        )
        return cfg.fast_weight_limit * torch.tanh(
            drive / cfg.fast_weight_limit
        )

    def observe_surprise(
        self,
        state: FieldState,
        surprise: torch.Tensor,
    ) -> FieldState:
        """Convert observed error to signed reward around recent expectation."""

        if surprise.shape != state.reward_baseline.shape:
            raise ValueError("surprise must match the state batch")
        detached = surprise.detach()
        reward = torch.tanh(state.reward_baseline - detached)
        baseline = (
            self.cfg.reward_baseline_decay * state.reward_baseline
            + (1 - self.cfg.reward_baseline_decay) * detached
        )
        return replace(
            state,
            reward=reward,
            reward_baseline=baseline,
        )

    def tick(
        self,
        state: FieldState,
        token: torch.Tensor | None,
        reward: torch.Tensor | None = None,
        retain_credit: bool = False,
        allow_fast_plasticity: bool = True,
    ) -> tuple[torch.Tensor, FieldState, dict[str, torch.Tensor]]:
        """Consume one optional character and emit one next-character distribution.

        `token=None` is a genuine unstimulated interval: cell dynamics continue, but
        external novelty is zero and energy therefore decays.
        """

        cfg = self.cfg
        batch = state.hidden.shape[0]
        reward = state.reward if reward is None else reward
        if reward.shape != (batch,):
            raise ValueError(f"reward must have shape ({batch},)")

        external = state.hidden.new_zeros(batch, cfg.cells, cfg.channels)
        external_stimulation = state.energy.new_zeros(batch, cfg.cells)
        sensory_trace = state.sensory_trace

        if token is None:
            novelty = state.energy.new_zeros(batch)
        else:
            if token.shape != (batch,):
                raise ValueError(f"token must have shape ({batch},)")
            embedded = self.token_embedding(token)
            novelty = (
                (embedded - state.sensory_trace).square().mean(dim=1) + 1e-8
            ).sqrt()
            projected = self.sensory_projection(embedded)
            external[:, self.sensory_indices] = projected.unsqueeze(1)
            external_stimulation[:, self.sensory_indices] = novelty.unsqueeze(1)
            sensory_trace = (
                cfg.sensory_trace_decay * state.sensory_trace
                + (1 - cfg.sensory_trace_decay) * embedded
            )

        hidden = state.hidden
        energy = state.energy
        stimulation = state.stimulation
        eligibility = state.eligibility
        edge_eligibility = state.edge_eligibility
        probe_eligibility = state.probe_eligibility
        structural_edge_evidence = (
            reward[:, None, None] * edge_eligibility
        ).mean(dim=0)
        structural_probe_evidence = (
            reward[:, None] * probe_eligibility
        ).mean(dim=0)
        fast_weight = (
            self._updated_fast_weight(
                state.fast_weight,
                state.edge_eligibility,
                reward,
            )
            if allow_fast_plasticity
            else torch.zeros_like(state.fast_weight)
        )
        context = self._cell_context().unsqueeze(0).expand(batch, -1, -1)
        mean_flow = state.energy.new_zeros(cfg.cells, cfg.dendrites)
        mean_probe_flow = state.energy.new_zeros(cfg.cells)
        final_probe_effect = torch.zeros_like(hidden)

        for _ in range(cfg.message_steps):
            source_hidden = hidden[:, self.sources]
            incoming, flow, propagated = self._messages(
                hidden, stimulation, fast_weight
            )
            probe_message, probe_flow, probe_source_hidden = (
                self._probe_message(hidden)
            )
            incoming_with_probe = incoming + probe_message
            stimulation = (
                cfg.stimulation_decay * propagated + external_stimulation
            ).clamp(0.0, 1.0)
            reward_drive = (
                cfg.reward_gain * reward[:, None, None] * eligibility
            )
            rule_input = torch.cat(
                (
                    external,
                    incoming_with_probe,
                    reward_drive,
                    context,
                    energy.unsqueeze(-1),
                ),
                dim=2,
            )
            updated = self.cell_rule(
                rule_input.flatten(0, 1), hidden.flatten(0, 1)
            ).reshape_as(hidden)
            if cfg.structural_probe_gain > 0:
                with torch.no_grad():
                    counterfactual_input = torch.cat(
                        (
                            external,
                            incoming,
                            reward_drive,
                            context,
                            energy.unsqueeze(-1),
                        ),
                        dim=2,
                    )
                    without_probe = self.cell_rule(
                        counterfactual_input.flatten(0, 1),
                        hidden.detach().flatten(0, 1),
                    ).reshape_as(hidden)
                    probe_effect = updated.detach() - without_probe
                    probe_tag = torch.tanh(
                        (
                            probe_source_hidden.detach()
                            * probe_effect
                        ).mean(dim=2)
                        * math.sqrt(cfg.channels)
                    )
                    final_probe_effect = probe_effect
            else:
                probe_tag = torch.zeros_like(probe_eligibility)
                final_probe_effect = torch.zeros_like(hidden)

            eligibility = (
                cfg.eligibility_decay * eligibility
                + (1 - cfg.eligibility_decay) * torch.tanh(external + incoming)
            )
            target_change = updated - hidden
            edge_tag = torch.tanh(
                (source_hidden * target_change.unsqueeze(2)).mean(dim=3)
                * math.sqrt(cfg.channels)
            )
            edge_tag = (
                edge_tag - edge_tag.mean(dim=2, keepdim=True)
            ).clamp(-1.0, 1.0)
            edge_eligibility = (
                cfg.edge_eligibility_decay * edge_eligibility
                + (1 - cfg.edge_eligibility_decay) * edge_tag
            )
            probe_eligibility = (
                cfg.edge_eligibility_decay * probe_eligibility
                + (1 - cfg.edge_eligibility_decay) * probe_tag
            )
            activity = (updated - hidden).abs().mean(dim=2)
            energy = (
                energy
                - cfg.basal_cost
                - cfg.activity_cost * activity
                + cfg.stimulation_gain * stimulation
            ).clamp(0.0, 1.0)
            hidden = updated
            mean_flow = mean_flow + flow.mean(dim=0) / cfg.message_steps
            mean_probe_flow = (
                mean_probe_flow
                + probe_flow.mean(dim=0) / cfg.message_steps
            )

        if retain_credit and hidden.requires_grad:
            hidden.retain_grad()

        output = hidden[:, self.output_indices].reshape(batch, -1)
        logits = self.output_readout(self.output_norm(output))
        next_state = FieldState(
            hidden=hidden,
            energy=energy,
            stimulation=stimulation,
            eligibility=eligibility,
            edge_eligibility=edge_eligibility,
            probe_eligibility=probe_eligibility,
            fast_weight=fast_weight,
            sensory_trace=sensory_trace,
            reward=torch.zeros_like(reward),
            reward_baseline=state.reward_baseline,
        )
        diagnostics = {
            "edge_flow": mean_flow,
            "novelty": novelty,
            "mean_energy": energy.mean(dim=1),
            "mean_fast_weight": fast_weight.abs().mean(dim=(1, 2)),
            "mean_edge_eligibility": edge_eligibility.abs().mean(
                dim=(1, 2)
            ),
            "mean_probe_eligibility": probe_eligibility.abs().mean(dim=1),
            "probe_flow": mean_probe_flow,
            "structural_edge_evidence": structural_edge_evidence,
            "structural_probe_evidence": structural_probe_evidence,
            "probe_effect": final_probe_effect,
            "total_rewires": self.total_rewires.detach().clone(),
            "fast_weight_saturation": (
                fast_weight.abs() >= 0.95 * cfg.fast_weight_limit
            )
            .to(fast_weight.dtype)
            .mean(dim=(1, 2)),
        }
        return logits, next_state, diagnostics

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        state: FieldState | None = None,
        targets: torch.Tensor | None = None,
        retain_credit: bool = False,
    ) -> tuple[torch.Tensor, FieldState, FieldTrace]:
        """Stream `(batch, time)` tokens through one uninterrupted organism state.

        When targets are supplied, each next-character score becomes a bounded scalar
        reward for the following tick. This is separate from gradient learning: exact
        BPTT supplies parameter credit inside the window, while reward-gated persistent
        eligibility carries event-specific influence across truncation boundaries.
        """

        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, time)")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("targets must match tokens")
        batch, length = tokens.shape
        if state is None:
            state = self.initial_state(batch, tokens.device)
        if state.hidden.shape[0] != batch:
            raise ValueError("state batch does not match tokens")

        logits: list[torch.Tensor] = []
        trace = FieldTrace(
            [], [], [], [], [], [], [], [], [], [], [], [], []
        )
        reward = state.reward

        for index in range(length):
            current_logits, state, diag = self.tick(
                state, tokens[:, index], reward, retain_credit=retain_credit
            )
            logits.append(current_logits)
            trace.hidden.append(state.hidden)
            trace.edge_flow.append(diag["edge_flow"])
            trace.novelty.append(diag["novelty"])
            trace.mean_energy.append(diag["mean_energy"])
            trace.mean_fast_weight.append(diag["mean_fast_weight"])
            trace.mean_edge_eligibility.append(
                diag["mean_edge_eligibility"]
            )
            trace.fast_weight_saturation.append(
                diag["fast_weight_saturation"]
            )
            trace.mean_probe_eligibility.append(
                diag["mean_probe_eligibility"]
            )
            trace.mean_probe_flow.append(diag["probe_flow"])
            trace.structural_edge_evidence.append(
                diag["structural_edge_evidence"]
            )
            trace.structural_probe_evidence.append(
                diag["structural_probe_evidence"]
            )
            trace.probe_effect.append(diag["probe_effect"])

            if targets is not None:
                surprise = F.cross_entropy(
                    current_logits, targets[:, index], reduction="none"
                )
                state = self.observe_surprise(state, surprise)
                reward = state.reward
            trace.prequential_reward.append(reward.detach())

        return torch.stack(logits, dim=1), state, trace
