"""Closed loop between one persistent organism and a multi-depth frozen backbone."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from sol2.model import Sol2
from sol2.state import OrganismState
from sol2.wiki_memory import LABELS

from .backbone import resolve_depths
from .config import Fable2Config
from .organ import BrocaOrgan


@dataclass
class EpisodeScore:
    """One scored question. Label scoring is always fp32 (C1 review instrument fix)."""

    label_logits: torch.Tensor
    loss: torch.Tensor
    correct: bool
    predicted_label: str
    correct_label: str
    label_log_probs: dict[str, float]
    state: OrganismState | None
    control_rms: float
    per_depth_control_rms: tuple[float, ...]
    internal_activity: float
    memory_activity: float


class BrocaSystem(nn.Module):
    """Let one continuing organism modulate a frozen LM at several depths."""

    def __init__(
        self,
        organism: Sol2,
        backbone: nn.Module,
        depths: tuple[int, ...],
        *,
        organ_name: str = "broca",
    ) -> None:
        super().__init__()
        if organ_name not in organism.attached_organs:
            raise KeyError(f"broca organ {organ_name!r} is not attached")
        organ = organism.attached_organs[organ_name]
        if organ.n_depths != len(depths):
            raise ValueError("organ depth count must match the resolved depth list")
        if len(set(depths)) != len(depths):
            raise ValueError("injection depths must be distinct")
        self.organism = organism
        self.backbone = backbone
        self.depths = tuple(int(depth) for depth in depths)
        self.organ_name = organ_name
        self._validate_frozen_backbone()

    def _validate_frozen_backbone(self) -> None:
        parameters = getattr(self.backbone, "parameters", None)
        if parameters is not None and any(p.requires_grad for p in parameters()):
            raise ValueError("language-backbone parameters must all be frozen")

    def initial_state(self, batch: int, device: str | torch.device) -> OrganismState:
        return self.organism.initial_state(batch, device)

    def _evolve(
        self,
        feature_sequence: torch.Tensor,
        state: OrganismState,
        *,
        write_memory: bool,
        frozen_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, OrganismState]:
        """Run the organism over frozen features; return final-step controls."""

        if feature_sequence.ndim != 3 or feature_sequence.shape[1] < 1:
            raise ValueError("features must have non-empty [batch, sequence, width] shape")
        next_state = state
        controls = None
        for position in range(feature_sequence.shape[1]):
            features = feature_sequence[:, position].to(
                device=next_state.hidden.device, dtype=next_state.hidden.dtype
            )
            controls, next_state, _ = self.organism.step(
                features,
                next_state,
                organ_name=self.organ_name,
                frozen_idx=frozen_idx,
                write_memory=write_memory,
            )
        assert controls is not None
        return controls, next_state

    def observe_feature_sequence(
        self,
        feature_sequence: torch.Tensor,
        state: OrganismState,
        *,
        frozen_idx: torch.Tensor | None = None,
    ) -> OrganismState:
        """Absorb an exposure (memory writes on) without touching the backbone."""

        _, next_state = self._evolve(
            feature_sequence, state, write_memory=True, frozen_idx=frozen_idx
        )
        return next_state

    def depth_controls(
        self,
        controls: torch.Tensor,
        *,
        lesion_depths: frozenset[int] = frozenset(),
    ) -> dict[int, torch.Tensor]:
        """Split stacked organ controls into the per-depth injection dictionary."""

        if controls.ndim != 4 or controls.shape[1] != len(self.depths):
            raise ValueError("controls must have [batch, depths, tokens, width] shape")
        unknown = lesion_depths - set(self.depths)
        if unknown:
            raise ValueError(f"cannot lesion unselected depths {sorted(unknown)}")
        banks: dict[int, torch.Tensor] = {}
        for position, depth in enumerate(self.depths):
            bank = controls[:, position]
            if depth in lesion_depths:
                bank = torch.zeros_like(bank).detach()
            banks[depth] = bank
        return banks

    def score_from_features(
        self,
        question_features: torch.Tensor,
        question_ids: torch.Tensor,
        state: OrganismState,
        labels: torch.Tensor,
        correct_index: int,
        *,
        bare: bool = False,
        inject: bool = True,
        lesion_depths: frozenset[int] = frozenset(),
        frozen_idx: torch.Tensor | None = None,
    ) -> EpisodeScore:
        """Evolve over a question and score its final answer distribution.

        `bare=True` is the organism-free floor (no hooks, no gradient graph).
        `inject=False` runs the organism but injects detached exact zeros, which the
        backbone contract makes bitwise-identical to the bare floor — asserted as an
        invariant by the CPU gate, never reported as an independent control arm.
        """

        if question_ids.shape[0] != 1:
            raise ValueError("episodes currently require one lifetime lane")
        device = question_ids.device
        if bare:
            logits = self.backbone.bare_logits(question_ids)[:, -1]
            scored_state = None
            control_rms = 0.0
            per_depth = tuple(0.0 for _ in self.depths)
            internal_activity = 0.0
            memory_activity = 0.0
        else:
            controls, scored_state = self._evolve(
                question_features, state, write_memory=False, frozen_idx=frozen_idx
            )
            if not inject:
                controls = torch.zeros_like(controls).detach()
            banks = self.depth_controls(controls, lesion_depths=lesion_depths)
            logits = self.backbone.injected_logits(question_ids, banks)[:, -1]
            control_rms = float(controls.detach().float().pow(2).mean().sqrt())
            per_depth = tuple(
                float(controls[:, position].detach().float().pow(2).mean().sqrt())
                for position in range(len(self.depths))
            )
            internal = scored_state.hidden[:, self.organism.internal_idx]
            memory = scored_state.hidden[:, self.organism.memory_idx]
            internal_activity = float(internal.detach().pow(2).mean().sqrt())
            memory_activity = float(memory.detach().pow(2).mean().sqrt())

        label_logits = logits.index_select(-1, labels.to(device)).float()
        target = torch.tensor([correct_index], device=device)
        loss = F.cross_entropy(label_logits, target)
        log_probs = label_logits.detach().log_softmax(dim=-1)[0]
        predicted = int(label_logits.detach().argmax(dim=-1))
        return EpisodeScore(
            label_logits=label_logits[0],
            loss=loss,
            correct=predicted == correct_index,
            predicted_label=LABELS[predicted],
            correct_label=LABELS[correct_index],
            label_log_probs={
                label: float(log_probs[index]) for index, label in enumerate(LABELS)
            },
            state=scored_state,
            control_rms=control_rms,
            per_depth_control_rms=per_depth,
            internal_activity=internal_activity,
            memory_activity=memory_activity,
        )


    def continuation_log_likelihood(
        self,
        full_ids: torch.Tensor,
        continuation_start: int,
        prompt_features: torch.Tensor,
        state: OrganismState,
        *,
        bare: bool = False,
        inject: bool = True,
        lesion_depths: frozenset[int] = frozenset(),
        frozen_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Mean fp32 per-token log-probability of a continuation span.

        The organism evolves over the *prompt* features only (the continuation
        is the measurement, never an observation), its final-state controls are
        injected over the full sequence, and the span's tokens are scored in
        fp32. Returns `(mean_log_prob_with_graph, control_rms)`.
        """

        if full_ids.shape[0] != 1:
            raise ValueError("episodes currently require one lifetime lane")
        if not 0 < continuation_start < full_ids.shape[1]:
            raise ValueError("continuation must be a non-empty suffix span")
        if bare:
            logits = self.backbone.bare_logits(full_ids)
            stats = {
                "control_rms": 0.0,
                "per_depth_control_rms": tuple(0.0 for _ in self.depths),
            }
        else:
            controls, _ = self._evolve(
                prompt_features, state, write_memory=False, frozen_idx=frozen_idx
            )
            if not inject:
                controls = torch.zeros_like(controls).detach()
            banks = self.depth_controls(controls, lesion_depths=lesion_depths)
            logits = self.backbone.injected_logits(full_ids, banks)
            stats = {
                "control_rms": float(controls.detach().float().pow(2).mean().sqrt()),
                "per_depth_control_rms": tuple(
                    float(controls[:, position].detach().float().pow(2).mean().sqrt())
                    for position in range(len(self.depths))
                ),
            }
        span_logits = logits[:, continuation_start - 1 : -1].float()
        span_targets = full_ids[:, continuation_start:]
        log_probs = span_logits.log_softmax(dim=-1)
        token_log_probs = log_probs.gather(-1, span_targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.mean(), stats


def build_broca_system(
    backbone: nn.Module,
    cfg: Fable2Config,
    *,
    device: str | torch.device = "cpu",
    organ_name: str = "broca",
) -> BrocaSystem:
    """Build a fresh organism, attach a deterministic Broca graft, wire the loop."""

    cfg.validate()
    depths = resolve_depths(cfg.depth_fractions, backbone.n_layers)
    torch.manual_seed(cfg.seed)
    organism = Sol2(cfg.organism_config(device=str(device))).to(device)
    cuda_devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(cfg.seed + 1)
        organ = BrocaOrgan(
            cfg.n_input,
            cfg.hidden,
            backbone.width,
            cfg.organ_queries,
            len(depths),
            cfg.n_control_tokens,
            cfg.control_rank,
            cfg.trunk_width,
            bounded_operators=organism.cfg.bounded_operators,
            operator_bound=organism.cfg.operator_bound,
            value_gain=organism.cfg.value_gain,
            attention_temperature=organism.cfg.attention_temperature,
            control_gain=cfg.control_gain,
        )
    organism.attach_organ_module(organ_name, organ)
    return BrocaSystem(organism, backbone, depths, organ_name=organ_name)
