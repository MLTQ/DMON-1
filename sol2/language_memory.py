"""Context-erasure curriculum that makes persistent organism state necessary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .living_language import LivingLanguageSystem
from .state import OrganismState


@dataclass(frozen=True)
class ContextErasureTask:
    mode_a_token: int = 1
    mode_b_token: int = 2
    distractor_token: int = 3
    query_token: int = 4
    response_a_token: int = 5
    response_b_token: int = 6
    exposure_steps: int = 3
    distractor_steps: int = 2

    def validate(self, vocab_size: int) -> None:
        tokens = (
            self.mode_a_token,
            self.mode_b_token,
            self.distractor_token,
            self.query_token,
            self.response_a_token,
            self.response_b_token,
        )
        if min(tokens) < 0 or max(tokens) >= vocab_size:
            raise ValueError("context-erasure task token lies outside the vocabulary")
        if self.mode_a_token == self.mode_b_token:
            raise ValueError("mode tokens must differ")
        if self.response_a_token == self.response_b_token:
            raise ValueError("response tokens must differ")
        if self.exposure_steps < 1 or self.distractor_steps < 0:
            raise ValueError("exposure_steps must be positive and distractor_steps nonnegative")


@dataclass
class ContextErasureResult:
    loss: torch.Tensor
    accuracy: float
    state: OrganismState
    control_rms: float
    internal_activity: float


@dataclass(frozen=True)
class LanguageMemoryTrainConfig:
    updates: int = 200
    batch_size: int = 16
    lr: float = 2e-3
    grad_clip: float = 1.0
    seed: int = 0
    persistent_lifetimes: bool = True

    def validate(self) -> None:
        if self.updates < 1 or self.batch_size < 2:
            raise ValueError("updates must be positive and batch_size at least two")
        if self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError("lr and grad_clip must be positive")


def balanced_modes(batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Return a shuffled, approximately balanced binary mode assignment."""

    modes = torch.arange(batch_size) % 2
    return modes[torch.randperm(batch_size, generator=generator)]


def run_context_erasure_episode(
    system: LivingLanguageSystem,
    state: OrganismState,
    modes: torch.Tensor,
    task: ContextErasureTask,
    *,
    control_scale: float = 1.0,
    reset_at_query: bool = False,
    frozen_idx: torch.Tensor | None = None,
) -> ContextErasureResult:
    """Teach a mode, erase visible context, then ask an identical query."""

    task.validate(system.backbone.vocab_size)
    if modes.ndim != 1 or modes.shape[0] != state.hidden.shape[0]:
        raise ValueError("modes must have one value per organism-state batch lane")
    device = state.hidden.device
    modes = modes.to(device=device, dtype=torch.long)
    mode_tokens = torch.where(
        modes == 0,
        torch.full_like(modes, task.mode_a_token),
        torch.full_like(modes, task.mode_b_token),
    )
    context = mode_tokens.unsqueeze(1)
    next_state = system.advance(
        context, state, control_scale=control_scale, frozen_idx=frozen_idx
    ).state
    for _ in range(task.exposure_steps - 1):
        repeated_mode = mode_tokens.unsqueeze(1)
        context = torch.cat([context, repeated_mode], dim=1)
        next_state = system.advance(
            context,
            next_state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
        ).state
    for _ in range(task.distractor_steps):
        distractor = torch.full(
            (len(modes), 1), task.distractor_token, dtype=torch.long, device=device
        )
        context = torch.cat([context, distractor], dim=1)
        next_state = system.advance(
            context,
            next_state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
        ).state

    if reset_at_query:
        next_state = system.initial_state(len(modes), device)
    erased_context = torch.full(
        (len(modes), 1), task.query_token, dtype=torch.long, device=device
    )
    query = system.advance(
        erased_context,
        next_state,
        control_scale=control_scale,
        frozen_idx=frozen_idx,
    )
    targets = torch.where(
        modes == 0,
        torch.full_like(modes, task.response_a_token),
        torch.full_like(modes, task.response_b_token),
    )
    loss = F.cross_entropy(query.logits, targets)
    accuracy = float((query.logits.argmax(dim=-1) == targets).float().mean())
    internal = query.state.hidden[:, system.organism.internal_idx]
    return ContextErasureResult(
        loss=loss,
        accuracy=accuracy,
        state=query.state,
        control_rms=float(query.controls.detach().pow(2).mean().sqrt()),
        internal_activity=float(internal.detach().pow(2).mean().sqrt()),
    )


def train_context_erasure(
    system: LivingLanguageSystem,
    task: ContextErasureTask,
    config: LanguageMemoryTrainConfig,
) -> list[dict[str, float]]:
    """Train persistent lifetime lanes on the context-erasure requirement."""

    config.validate()
    task.validate(system.backbone.vocab_size)
    device = next(system.organism.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in system.organism.parameters() if parameter.requires_grad),
        lr=config.lr,
        weight_decay=0.0,
    )
    state = system.initial_state(config.batch_size, device)
    history: list[dict[str, float]] = []
    for update in range(1, config.updates + 1):
        modes = balanced_modes(config.batch_size, generator).to(device)
        result = run_context_erasure_episode(system, state, modes, task)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            system.organism.parameters(), config.grad_clip
        )
        optimizer.step()
        history.append(
            {
                "update": float(update),
                "loss": float(result.loss.detach()),
                "accuracy": result.accuracy,
                "control_rms": result.control_rms,
                "internal_activity": result.internal_activity,
                "grad_norm": float(grad_norm),
            }
        )
        state = (
            result.state.clone_detached()
            if config.persistent_lifetimes
            else system.initial_state(config.batch_size, device)
        )
    return history


@torch.no_grad()
def evaluate_context_erasure_controls(
    system: LivingLanguageSystem,
    task: ContextErasureTask,
    *,
    batch_size: int = 32,
    seed: int = 10_000,
) -> dict[str, ContextErasureResult]:
    """Evaluate normal, zero-control, reset, and internal-lesion causal arms."""

    device = next(system.organism.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    modes = balanced_modes(batch_size, generator).to(device)
    initial = system.initial_state(batch_size, device)
    return {
        "normal": run_context_erasure_episode(system, initial, modes, task),
        "zero_control": run_context_erasure_episode(
            system, initial, modes, task, control_scale=0.0
        ),
        "reset": run_context_erasure_episode(
            system, initial, modes, task, reset_at_query=True
        ),
        "internal_lesion": run_context_erasure_episode(
            system,
            initial,
            modes,
            task,
            frozen_idx=system.organism.internal_idx,
        ),
    }
