"""Persistent closed loop between a frozen language organ and the SOL2 organism."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .language_backbone import FrozenLanguageBackbone
from .language_organs import ContinuousLanguageOrgan
from .model import Sol2, StepHealth, StepTrace
from .state import OrganismState


@dataclass
class LanguageStep:
    logits: torch.Tensor
    controls: torch.Tensor
    state: OrganismState
    health: StepHealth | None
    recalled: torch.Tensor | None
    recall_content_scores: torch.Tensor | None = None
    recall_scores: torch.Tensor | None = None
    recall_attention: torch.Tensor | None = None


class LivingLanguageSystem(nn.Module):
    """Let one continuing organism perceive and control a frozen language model."""

    def __init__(
        self,
        organism: Sol2,
        backbone: FrozenLanguageBackbone,
        *,
        organ_name: str = "language",
    ) -> None:
        super().__init__()
        if organ_name not in organism.attached_organs:
            raise KeyError(f"continuous organ {organ_name!r} is not attached")
        self.organism = organism
        # Registered when the implementation is an nn.Module, which makes device
        # movement reliable; frozen parameters are still excluded from training.
        self.backbone = backbone
        self.organ_name = organ_name
        self._validate_frozen_backbone()

    def _validate_frozen_backbone(self) -> None:
        parameters = getattr(self.backbone, "parameters", None)
        if parameters is not None and any(parameter.requires_grad for parameter in parameters()):
            raise ValueError("language-backbone parameters must all be frozen")

    def initial_state(self, batch: int, device: str | torch.device) -> OrganismState:
        return self.organism.initial_state(batch, device)

    def _evolve_feature_sequence(
        self,
        feature_sequence: torch.Tensor,
        state: OrganismState,
        *,
        control_scale: float,
        frozen_idx: torch.Tensor | None,
        write_memory: bool,
        collect_final_health: bool = False,
        capture_final_recall: bool = False,
    ) -> tuple[torch.Tensor, OrganismState, StepHealth | None, StepTrace | None]:
        controls = []
        next_state = state
        final_health = None
        final_trace = None
        for position in range(feature_sequence.shape[1]):
            features = feature_sequence[:, position].to(
                device=next_state.hidden.device,
                dtype=next_state.hidden.dtype,
            )
            trace = (
                StepTrace()
                if capture_final_recall
                and position == feature_sequence.shape[1] - 1
                else None
            )
            position_controls, next_state, health = self.organism.step(
                features,
                next_state,
                organ_name=self.organ_name,
                frozen_idx=frozen_idx,
                write_memory=write_memory,
                collect_health=(
                    collect_final_health and position == feature_sequence.shape[1] - 1
                ),
                trace=trace,
            )
            controls.append(position_controls * control_scale)
            if health is not None:
                final_health = health
            if trace is not None:
                final_trace = trace
        return torch.stack(controls, dim=1), next_state, final_health, final_trace

    def observe_sequence(
        self,
        input_ids: torch.Tensor,
        state: OrganismState,
        *,
        control_scale: float = 1.0,
        frozen_idx: torch.Tensor | None = None,
        write_memory: bool = True,
    ) -> tuple[OrganismState, torch.Tensor]:
        """Absorb a causal token sequence without allocating vocabulary logits."""

        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain a non-empty token sequence")
        feature_sequence = self.backbone.encode(input_ids)
        return self.observe_feature_sequence(
            feature_sequence,
            state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
            write_memory=write_memory,
        )

    def observe_feature_sequence(
        self,
        feature_sequence: torch.Tensor,
        state: OrganismState,
        *,
        control_scale: float = 1.0,
        frozen_idx: torch.Tensor | None = None,
        write_memory: bool = True,
    ) -> tuple[OrganismState, torch.Tensor]:
        """Absorb already-computed frozen features without vocabulary logits."""

        if feature_sequence.ndim != 3 or feature_sequence.shape[1] < 1:
            raise ValueError("features must have non-empty [batch, sequence, width] shape")
        controls, next_state, _, _ = self._evolve_feature_sequence(
            feature_sequence,
            state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
            write_memory=write_memory,
        )
        return next_state, controls

    def score_next_after_sequence(
        self,
        input_ids: torch.Tensor,
        state: OrganismState,
        *,
        control_scale: float = 1.0,
        frozen_idx: torch.Tensor | None = None,
        write_memory: bool = True,
        collect_health: bool = False,
        control_mode: str = "late_residual",
        control_reference: torch.Tensor | None = None,
        capture_final_recall: bool = False,
    ) -> LanguageStep:
        """Absorb a complete prompt and score only its next-token distribution."""

        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain a non-empty token sequence")
        feature_sequence = self.backbone.encode(input_ids)
        return self.score_next_from_features(
            feature_sequence,
            state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
            write_memory=write_memory,
            collect_health=collect_health,
            control_mode=control_mode,
            input_ids=input_ids,
            control_reference=control_reference,
            capture_final_recall=capture_final_recall,
        )

    def score_next_from_features(
        self,
        feature_sequence: torch.Tensor,
        state: OrganismState,
        *,
        control_scale: float = 1.0,
        frozen_idx: torch.Tensor | None = None,
        write_memory: bool = True,
        collect_health: bool = False,
        control_mode: str = "late_residual",
        input_ids: torch.Tensor | None = None,
        control_reference: torch.Tensor | None = None,
        capture_final_recall: bool = False,
    ) -> LanguageStep:
        """Evolve over cached features and score their final next-token distribution."""

        if feature_sequence.ndim != 3 or feature_sequence.shape[1] < 1:
            raise ValueError("features must have non-empty [batch, sequence, width] shape")
        controls, next_state, health, recall_trace = self._evolve_feature_sequence(
            feature_sequence,
            state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
            write_memory=write_memory,
            collect_final_health=collect_health,
            capture_final_recall=capture_final_recall,
        )
        effective_controls = controls[:, -1]
        if control_reference is not None:
            if control_reference.shape != effective_controls.shape:
                raise ValueError(
                    "control reference must match [batch, tokens, width] controls"
                )
            effective_controls = effective_controls - control_scale * (
                control_reference.detach().to(
                    device=effective_controls.device,
                    dtype=effective_controls.dtype,
                )
            )
        if control_mode == "late_residual":
            logits = self.backbone.controlled_logits_from_features(
                feature_sequence[:, -1:], effective_controls
            )[:, -1]
        elif control_mode == "prefix":
            if input_ids is None:
                raise ValueError("prefix control requires the original input_ids")
            if input_ids.shape[:2] != feature_sequence.shape[:2]:
                raise ValueError("prefix input_ids must match cached feature positions")
            logits = self.backbone.prefix_logits(input_ids, effective_controls)[:, -1]
        else:
            raise ValueError(f"unknown language control mode {control_mode!r}")
        return LanguageStep(
            logits=logits,
            controls=effective_controls,
            state=next_state,
            health=health,
            recalled=None if recall_trace is None else recall_trace.recalled,
            recall_content_scores=(
                None if recall_trace is None else recall_trace.recall_content_scores
            ),
            recall_scores=None if recall_trace is None else recall_trace.recall_scores,
            recall_attention=(
                None if recall_trace is None else recall_trace.recall_attention
            ),
        )

    def advance(
        self,
        context_ids: torch.Tensor,
        state: OrganismState,
        *,
        control_scale: float = 1.0,
        frozen_idx: torch.Tensor | None = None,
        collect_health: bool = False,
        control_mode: str = "late_residual",
    ) -> LanguageStep:
        """Observe the newest context token, evolve once, and predict the next token."""

        if context_ids.ndim != 2 or context_ids.shape[1] < 1:
            raise ValueError("context_ids must contain at least one token per batch")
        feature_sequence = self.backbone.encode(context_ids)
        features = feature_sequence[:, -1].to(
            device=state.hidden.device,
            dtype=state.hidden.dtype,
        )
        controls, next_state, health = self.organism.step(
            features,
            state,
            organ_name=self.organ_name,
            frozen_idx=frozen_idx,
            collect_health=collect_health,
        )
        controlled = controls * control_scale
        if control_mode == "late_residual":
            logits = self.backbone.controlled_logits_from_features(
                feature_sequence, controlled
            )[:, -1]
        elif control_mode == "prefix":
            logits = self.backbone.prefix_logits(context_ids, controlled)[:, -1]
        else:
            raise ValueError(f"unknown language control mode {control_mode!r}")
        return LanguageStep(
            logits=logits,
            controls=controlled,
            state=next_state,
            health=health,
            recalled=None,
        )

    def teacher_forced_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        state: OrganismState,
        *,
        loss_mask: torch.Tensor | None = None,
        control_scale: float = 1.0,
        frozen_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, OrganismState, torch.Tensor]:
        """Train on a token sequence while preserving cellular state across positions."""

        if input_ids.shape != target_ids.shape or input_ids.ndim != 2:
            raise ValueError("input_ids and target_ids must share [batch, sequence] shape")
        feature_sequence = self.backbone.encode(input_ids)
        control_sequence, next_state, _, _ = self._evolve_feature_sequence(
            feature_sequence,
            state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
            write_memory=True,
        )
        logits = self.backbone.controlled_logits_sequence_from_features(
            feature_sequence, control_sequence
        )
        token_loss = F.cross_entropy(
            logits.flatten(0, 1), target_ids.flatten(), reduction="none"
        ).view_as(target_ids)
        if loss_mask is None:
            loss = token_loss.mean()
        else:
            if loss_mask.shape != input_ids.shape:
                raise ValueError("loss_mask must share the input sequence shape")
            mask = loss_mask.to(device=token_loss.device, dtype=token_loss.dtype)
            loss = (token_loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, next_state, control_sequence

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        state: OrganismState,
        *,
        max_new_tokens: int,
        control_scale: float = 1.0,
        frozen_idx: torch.Tensor | None = None,
        control_mode: str = "late_residual",
    ) -> tuple[torch.Tensor, OrganismState]:
        """Greedily generate while feeding every prompt and generated token back."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        context = prompt_ids
        next_state = state
        step = None
        for position in range(prompt_ids.shape[1]):
            step = self.advance(
                context[:, : position + 1],
                next_state,
                control_scale=control_scale,
                frozen_idx=frozen_idx,
                control_mode=control_mode,
            )
            next_state = step.state
        for _ in range(max_new_tokens):
            if step is None:
                raise ValueError("prompt_ids must contain at least one token")
            token = step.logits.argmax(dim=-1, keepdim=True)
            context = torch.cat([context, token], dim=1)
            step = self.advance(
                context,
                next_state,
                control_scale=control_scale,
                frozen_idx=frozen_idx,
                control_mode=control_mode,
            )
            next_state = step.state
        return context, next_state


def graft_language_backbone(
    organism: Sol2,
    backbone: FrozenLanguageBackbone,
    *,
    organ_name: str = "language",
    n_control_tokens: int = 4,
    control_rank: int = 8,
    control_gain: float = 1.0,
    recall_gain: float = 0.25,
    coherent_recall: bool = False,
    recall_residual_gain: float = 0.1,
    recall_top_k: int = 0,
    recall_recency_bias: float = 0.0,
    seed: int = 0,
) -> LivingLanguageSystem:
    """Create and attach a deterministic continuous graft for one frozen backbone."""

    device = organism.embedding.weight.device
    cuda_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        organ = ContinuousLanguageOrgan(
            organism.cfg.n_input,
            organism.cfg.hidden,
            backbone.width,
            organism.cfg.organ_queries,
            n_control_tokens,
            control_rank,
            bounded_operators=organism.cfg.bounded_operators,
            operator_bound=organism.cfg.operator_bound,
            value_gain=organism.cfg.value_gain,
            attention_temperature=organism.cfg.attention_temperature,
            control_gain=control_gain,
            recall_gain=recall_gain,
            coherent_recall=coherent_recall,
            recall_residual_gain=recall_residual_gain,
            recall_top_k=recall_top_k,
            recall_recency_bias=recall_recency_bias,
        )
    organism.attach_organ_module(organ_name, organ)
    return LivingLanguageSystem(organism, backbone, organ_name=organ_name)
