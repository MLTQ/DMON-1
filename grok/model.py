"""Streaming organism: dendrites, mirrors, eligibility, reverse vector credit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .cell import SharedGRURule
from .config import TrainConfig
from .graph import DendriteGraph


@dataclass(slots=True)
class CreatureState:
    """Persistent electrical + credit state across stream tokens and BPTT cuts."""

    h: torch.Tensor  # [B, N, H]
    eligibility: torch.Tensor  # [B, N, H]
    edge_eligibility: torch.Tensor  # [B, N, K]
    fast_weight: torch.Tensor  # [B, N, K]
    output_error_credit: torch.Tensor  # [B, N, H]
    reward: torch.Tensor  # [B]
    reward_baseline: torch.Tensor  # [B]
    mirror_cursor: int
    # Structural probe telemetry (detached)
    probe_credit: torch.Tensor | None = None  # [N]
    probe_fitness: torch.Tensor | None = None  # [N]

    def detach(self) -> "CreatureState":
        return CreatureState(
            h=self.h.detach(),
            eligibility=self.eligibility.detach(),
            edge_eligibility=self.edge_eligibility.detach(),
            fast_weight=self.fast_weight.detach(),
            output_error_credit=self.output_error_credit.detach(),
            reward=self.reward.detach(),
            reward_baseline=self.reward_baseline.detach(),
            mirror_cursor=self.mirror_cursor,
            probe_credit=(
                None if self.probe_credit is None else self.probe_credit.detach()
            ),
            probe_fitness=(
                None if self.probe_fitness is None else self.probe_fitness.detach()
            ),
        )


class StreamingCreature(nn.Module):
    """Character organ on a sparse directed field.

    Port layout:
      [0, n_input)           — sensory (stream + mutable)
      [n_input, +n_mirror)   — mirrors (stream-written only)
      internals              — computation
      last n_output          — readout bank

    Credit path (SOL findings):
      - eligibility remembers recent participation
      - delayed signed reward (vs surprise baseline) meets eligibility
      - decoder-shaped reverse credit travels sourceward through real dendrites
      - fast edge efficacy updates when reward meets edge eligibility
    """

    def __init__(self, config: TrainConfig, vocab_size: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vocab_size = vocab_size
        h = config.hidden
        n = config.n_cells
        k = config.n_dendrites

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
        self.input_proj = nn.Linear(h, config.n_input * h, bias=False)
        self.graph = DendriteGraph(
            n_cells=n,
            n_dendrites=k,
            hidden=h,
            input_idx=self.input_idx,
            output_idx=self.output_idx,
            mirror_idx=self.mirror_idx,
            internal_idx=self.internal_idx,
            seed=config.seed,
            use_attention=config.use_attention,
        )
        self.rule = SharedGRURule(h)
        self.readout_mode = config.readout_mode
        if self.readout_mode == "mean":
            self.out_norm = nn.LayerNorm(h)
            self.readout = nn.Linear(h, vocab_size)
            self.out_query = None
        elif self.readout_mode == "concat":
            self.out_norm = nn.LayerNorm(config.n_output * h)
            self.readout = nn.Linear(config.n_output * h, vocab_size)
            self.out_query = None
        else:  # attn
            self.out_norm = nn.LayerNorm(h)
            self.out_query = nn.Parameter(torch.randn(h) * 0.02)
            self.readout = nn.Linear(h, vocab_size)

        # Structural probe sources (one non-edge candidate per target)
        probes = self._initial_probes(n, self.graph.sources, config.seed + 1)
        self.register_buffer("probe_sources", probes, persistent=True)
        self.register_buffer(
            "probe_confirmations", torch.zeros(n, dtype=torch.long), persistent=True
        )
        self.register_buffer("total_rewires", torch.zeros((), dtype=torch.long))

        nn.init.normal_(self.embed.weight, std=0.16)

    @staticmethod
    def _initial_probes(n: int, sources: torch.Tensor, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        probes = torch.zeros(n, dtype=torch.long)
        for t in range(n):
            used = set(sources[t].tolist()) | {t}
            candidates = [i for i in range(n) if i not in used]
            if not candidates:
                candidates = [i for i in range(n) if i != t]
            j = int(torch.randint(0, len(candidates), (1,), generator=g))
            probes[t] = candidates[j]
        return probes

    def initial_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> CreatureState:
        if device is None:
            device = self.embed.weight.device
        n, h, k = self.config.n_cells, self.config.hidden, self.config.n_dendrites
        return CreatureState(
            h=torch.zeros(batch_size, n, h, device=device),
            eligibility=torch.zeros(batch_size, n, h, device=device),
            edge_eligibility=torch.zeros(batch_size, n, k, device=device),
            fast_weight=torch.zeros(batch_size, n, k, device=device),
            output_error_credit=torch.zeros(batch_size, n, h, device=device),
            reward=torch.zeros(batch_size, device=device),
            reward_baseline=torch.full(
                (batch_size,), math.log(self.vocab_size), device=device
            ),
            mirror_cursor=0,
            probe_credit=torch.zeros(n, device=device),
            probe_fitness=torch.zeros(n, device=device),
        )

    def _update_fast_weight(
        self, fast: torch.Tensor, edge_elig: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        drive = (
            cfg.fast_weight_decay * fast
            + cfg.fast_plasticity_gain * reward[:, None, None] * edge_elig
        )
        # Competitive within each target's fan.
        drive = drive - drive.mean(dim=2, keepdim=True)
        limit = cfg.fast_weight_limit
        return limit * torch.tanh(drive / limit)

    def observe_prediction(
        self, state: CreatureState, logits: torch.Tensor, target: torch.Tensor
    ) -> CreatureState:
        """Set delayed scalar reward and launch decoder-shaped reverse credit."""

        surprise = F.cross_entropy(logits, target, reduction="none").detach()
        reward = torch.tanh(state.reward_baseline - surprise)
        baseline = (
            self.config.reward_baseline_decay * state.reward_baseline
            + (1 - self.config.reward_baseline_decay) * surprise
        )
        credit = state.output_error_credit
        if self.config.output_error_credit_gain > 0:
            with torch.no_grad():
                correction = F.one_hot(target, self.vocab_size).to(logits.dtype)
                correction = correction - torch.softmax(logits.detach(), dim=-1)
                # Map vocab residual into hidden via readout transpose.
                raw = F.linear(correction, self.readout.weight.t())
                credit = credit.clone()
                if self.readout_mode == "concat":
                    # raw: [B, n_output * H] → per-output slices
                    raw = raw.view(-1, self.config.n_output, self.config.hidden)
                    raw = raw / math.sqrt(self.config.n_output)
                    credit[:, self.output_idx, :] = torch.tanh(
                        credit[:, self.output_idx, :] + raw
                    )
                else:
                    out_corr = raw / math.sqrt(self.config.n_output)
                    credit[:, self.output_idx, :] = torch.tanh(
                        credit[:, self.output_idx, :] + out_corr.unsqueeze(1)
                    )
        else:
            credit = torch.zeros_like(credit)
        return CreatureState(
            h=state.h,
            eligibility=state.eligibility,
            edge_eligibility=state.edge_eligibility,
            fast_weight=state.fast_weight,
            output_error_credit=credit,
            reward=reward,
            reward_baseline=baseline,
            mirror_cursor=state.mirror_cursor,
            probe_credit=state.probe_credit,
            probe_fitness=state.probe_fitness,
        )

    def step(
        self,
        token_ids: torch.Tensor,
        state: CreatureState,
    ) -> tuple[torch.Tensor, CreatureState]:
        """Consume one token per batch row; return logits and updated state."""

        cfg = self.config
        b = token_ids.shape[0]
        h = state.h
        if h.shape[0] != b:
            raise ValueError(f"state batch {h.shape[0]} != tokens {b}")

        emb = self.embed(token_ids)
        drive_full = torch.zeros_like(h)
        spread = self.input_proj(emb).view(b, cfg.n_input, cfg.hidden)
        drive_full = drive_full.clone()
        drive_full[:, self.input_idx, :] = spread

        # Mirror write: stream only, detached content.
        h = h.clone()
        slot = self.mirror_idx[state.mirror_cursor % cfg.n_mirror]
        h[:, slot, :] = emb.detach()
        next_cursor = (state.mirror_cursor + 1) % cfg.n_mirror

        # Apply delayed reward to fast weights before this tick's dynamics.
        fast = self._update_fast_weight(
            state.fast_weight, state.edge_eligibility, state.reward
        )

        eligibility = state.eligibility
        edge_elig = state.edge_eligibility
        out_credit = state.output_error_credit
        last_coeff = None
        probe_credit_acc = (
            state.probe_credit
            if state.probe_credit is not None
            else h.new_zeros(cfg.n_cells)
        )
        probe_fit_acc = (
            state.probe_fitness
            if state.probe_fitness is not None
            else h.new_zeros(cfg.n_cells)
        )

        for micro in range(cfg.steps_per_token):
            messages, coeff = self.graph.aggregate(h, fast)
            last_coeff = coeff

            # Optional weak structural probe message (non-differentiable evidence).
            if cfg.structural_probe_gain > 0:
                probe_src = h[:, self.probe_sources, :]  # [B, N, H]
                probe_msg = cfg.structural_probe_gain * probe_src
                messages = messages + probe_msg

            credit_drive = (
                cfg.reward_gain * state.reward[:, None, None] * eligibility
                + cfg.output_error_credit_gain * out_credit * eligibility
            )
            drive = drive_full if micro == 0 else torch.zeros_like(drive_full)
            proposed = self.rule(h, messages, drive, credit_drive)
            mask = self.mutable_mask.view(1, -1, 1)
            updated = torch.where(mask, proposed, h)

            # Eligibility tags (detached from graph for stability of memory).
            with torch.no_grad():
                participation = torch.tanh(drive_full + messages).detach()
                eligibility = (
                    cfg.eligibility_decay * eligibility
                    + (1 - cfg.eligibility_decay) * participation
                )
                # Edge tags: source state aligns with target change.
                src_h = h[:, self.graph.sources, :].detach()
                delta = (updated - h).detach().unsqueeze(2)
                edge_tag = torch.tanh(
                    (src_h * delta).mean(dim=3) * math.sqrt(cfg.hidden)
                )
                edge_tag = edge_tag - edge_tag.mean(dim=2, keepdim=True)
                edge_elig = (
                    cfg.edge_eligibility_decay * edge_elig
                    + (1 - cfg.edge_eligibility_decay) * edge_tag
                )
                if cfg.structural_probe_gain > 0:
                    # Local probe credit: reward × probe·delta alignment.
                    psrc = h[:, self.probe_sources, :].detach()
                    ptag = torch.tanh(
                        (psrc * (updated - h).detach()).mean(dim=2)
                        * math.sqrt(cfg.hidden)
                    )
                    probe_credit_acc = probe_credit_acc + (
                        state.reward[:, None] * ptag
                    ).mean(dim=0)

            h = updated

            # Transport reverse credit sourceward along the same coefficients.
            if cfg.output_error_credit_gain > 0 and last_coeff is not None:
                with torch.no_grad():
                    transported = self.graph.transport_vector_credit(
                        out_credit, last_coeff.detach()
                    )
                    out_credit = torch.tanh(
                        cfg.output_error_credit_decay * out_credit + transported
                    )

        out_bank = h[:, self.output_idx, :]  # [B, O, H]
        if self.readout_mode == "mean":
            pooled = out_bank.mean(dim=1)
        elif self.readout_mode == "concat":
            pooled = out_bank.reshape(b, -1)
        else:
            assert self.out_query is not None
            scores = torch.einsum("boh,h->bo", out_bank, self.out_query)
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.einsum("bo,boh->bh", weights, out_bank)
        logits = self.readout(self.out_norm(pooled))
        new_state = CreatureState(
            h=h,
            eligibility=eligibility,
            edge_eligibility=edge_elig,
            fast_weight=fast,
            output_error_credit=out_credit,
            reward=state.reward,
            reward_baseline=state.reward_baseline,
            mirror_cursor=next_cursor,
            probe_credit=probe_credit_acc,
            probe_fitness=probe_fit_acc,
        )
        return logits, new_state

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        state: CreatureState,
    ) -> tuple[torch.Tensor, CreatureState, torch.Tensor]:
        """Run one truncated BPTT window. tokens/targets: [B, T]."""

        if tokens.shape != targets.shape:
            raise ValueError("tokens and targets must match")
        losses = []
        for t in range(tokens.shape[1]):
            logits, state = self.step(tokens[:, t], state)
            losses.append(F.cross_entropy(logits, targets[:, t]))
            state = self.observe_prediction(state, logits, targets[:, t])
        loss = torch.stack(losses).mean()
        return logits, state, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
