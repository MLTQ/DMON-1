"""Shared schedule, optimizer, and update-health logic for every SOL2 arm."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import Sol2Config


def schedule_factor(cfg: Sol2Config, update: int) -> float:
    if update <= cfg.warmup_updates:
        return update / max(cfg.warmup_updates, 1)
    if cfg.schedule == "constant":
        return 1.0
    progress = min(
        (update - cfg.warmup_updates)
        / max(cfg.updates - cfg.warmup_updates, 1),
        1.0,
    )
    floor = cfg.lr_min / cfg.lr
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer: torch.optim.Optimizer, cfg: Sol2Config, update: int) -> float:
    learning_rate = cfg.lr * schedule_factor(cfg, update)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def build_optimizer(model: torch.nn.Module, cfg: Sol2Config) -> torch.optim.AdamW:
    no_decay_markers = (
        "bias",
        "norm",
        "edge_logit",
        "cell_gain",
        "cell_bias",
        "sensory_gain",
        "sensory_bias",
    )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if parameter.ndim < 2 or any(
            marker in name for marker in no_decay_markers
        ) else decay
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )


@dataclass
class UpdateHealth:
    accepted: bool
    total_norm: float
    module_norms: dict[str, float]
    cell_gradient_utilization: dict[str, float]


def gradient_health(model: torch.nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    by_module: dict[str, list[torch.Tensor]] = {}
    all_norms = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().norm()
        all_norms.append(norm)
        module = name.split(".", 1)[0]
        by_module.setdefault(module, []).append(norm)
    if not all_norms:
        raise RuntimeError("no gradients reached the model")
    total = torch.linalg.vector_norm(torch.stack(all_norms))
    modules = {
        name: float(torch.linalg.vector_norm(torch.stack(norms)))
        for name, norms in by_module.items()
    }
    return total, modules


def cell_gradient_utilization(model: torch.nn.Module) -> dict[str, float]:
    """Summarize useful and reserve identity rows without requiring full occupancy."""

    gain = getattr(model, "cell_gain", None)
    bias = getattr(model, "cell_bias", None)
    if gain is None or bias is None or gain.grad is None or bias.grad is None:
        return {}
    per_expression = (
        gain.grad.detach().square() + bias.grad.detach().square()
    ).mean(-1).sqrt()
    per_cell = per_expression.new_zeros(model.n_cells)
    per_cell[model.expression_cells] = per_expression
    internal = getattr(model, "internal_idx")
    internal_scale = per_cell[internal].max().clamp_min(1e-12)
    internal_weight = per_cell[internal]
    if float(internal_weight.sum()) <= 1e-12:
        effective_fraction = 0.0
    else:
        probability = internal_weight / internal_weight.sum()
        effective_fraction = float(
            (-(probability * probability.clamp_min(1e-12).log()).sum()).exp()
            / len(internal)
        )
    result = {
        "internal_nonzero_fraction": float((per_cell[internal] > 1e-12).float().mean()),
        "internal_material_fraction": float(
            (per_cell[internal] > internal_scale * 1e-2).float().mean()
        ),
        "internal_effective_fraction": effective_fraction,
    }
    result["internal_reserve_fraction"] = 1.0 - effective_fraction
    for name in getattr(model, "TISSUES"):
        cells = model.tissue_indices(name)
        scale = per_cell[cells].max().clamp_min(1e-12)
        result[f"{name}_material_fraction"] = float(
            (per_cell[cells] > scale * 1e-2).float().mean()
        )
        result[f"{name}_gradient_rms"] = float(
            per_cell[cells].square().mean().sqrt()
        )
    return result


def guarded_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Sol2Config,
) -> UpdateHealth:
    total, modules = gradient_health(model)
    utilization = cell_gradient_utilization(model)
    accepted = bool(torch.isfinite(total)) and float(total) < cfg.reject_grad_norm
    if accepted:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    return UpdateHealth(accepted, float(total), modules, utilization)
