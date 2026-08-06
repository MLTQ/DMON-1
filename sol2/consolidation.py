"""Causal-utility calibration and graded update protection for SOL2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import Sol2
from .procedural_benchmark import _detach_state, _step_episode
from .procedural_task import ProceduralTask, ProcedureRegime


@dataclass
class UtilityProfile:
    """Measured importance and the branch-specific plasticity it induces."""

    measured_cell: torch.Tensor
    measured_edge: torch.Tensor
    applied_cell: torch.Tensor
    applied_edge: torch.Tensor
    cell_plasticity: torch.Tensor
    edge_plasticity: torch.Tensor
    genome_plasticity: float

    def cpu_payload(self) -> dict:
        return {
            "measured_cell": self.measured_cell.detach().cpu(),
            "measured_edge": self.measured_edge.detach().cpu(),
            "applied_cell": self.applied_cell.detach().cpu(),
            "applied_edge": self.applied_edge.detach().cpu(),
            "cell_plasticity": self.cell_plasticity.detach().cpu(),
            "edge_plasticity": self.edge_plasticity.detach().cpu(),
            "genome_plasticity": self.genome_plasticity,
        }


def _rank(values: torch.Tensor) -> torch.Tensor:
    """Map a nonconstant vector to deterministic ranks in [0, 1]."""

    if values.numel() <= 1 or float(values.max() - values.min()) <= 1e-20:
        return torch.zeros_like(values)
    order = torch.argsort(values, stable=True)
    result = torch.empty_like(values)
    result[order] = torch.linspace(
        0.0, 1.0, values.numel(), device=values.device, dtype=values.dtype
    )
    return result


def _cell_tissue_ranks(model: Sol2, values: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(values)
    for tissue in ("input", "memory", "compute", "relay", "output"):
        cells = model.tissue_indices(tissue)
        result[cells] = _rank(values[cells])
    return result


def _edge_tissue_ranks(model: Sol2, values: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(values)
    active = model.graph.active
    for tissue in ("input", "memory", "compute", "relay", "output"):
        targets = model.tissue_indices(tissue)
        tissue_active = active[targets]
        ranked = _rank(values[targets][tissue_active])
        tissue_result = result[targets].clone()
        tissue_result[tissue_active] = ranked
        result[targets] = tissue_result
    return result


def _edge_to_cell_scores(model: Sol2, edge_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    active = model.graph.active
    active_score = edge_score * active
    incoming = active_score.sum(-1) / active.sum(-1).clamp_min(1)
    outgoing = edge_score.new_zeros(model.n_cells)
    outgoing.scatter_add_(0, model.graph.sources[active], active_score[active])
    outgoing_count = edge_score.new_zeros(model.n_cells)
    outgoing_count.scatter_add_(
        0,
        model.graph.sources[active],
        torch.ones_like(active_score[active]),
    )
    return incoming, outgoing / outgoing_count.clamp_min(1)


def calibrate_causal_utility(
    model: Sol2,
    state,
    task: ProceduralTask,
    regime: ProcedureRegime,
    cfg,
    *,
    batches: int,
    steps: int,
    generator_seed: int,
    directional_edges: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Estimate A utility from private-expression and anatomical gradient demand."""

    if batches < 1 or steps < 1:
        raise ValueError("utility calibration budgets must be positive")
    generator = torch.Generator().manual_seed(generator_seed)
    calibration_state = state.clone_detached()
    identity_score = torch.zeros(model.n_cells, device=cfg.device)
    adapter_down_score = torch.zeros(model.n_cells, device=cfg.device)
    adapter_up_score = torch.zeros(model.n_cells, device=cfg.device)
    edge_score = torch.zeros_like(model.graph.edge_logit)
    was_training = model.training
    model.eval()
    losses = []
    accuracies = []
    for _ in range(batches):
        batch = task.sample_batch(regime, cfg.batch_size, steps, generator, cfg.device)
        logits, next_state = _step_episode(model, batch.tokens, calibration_state)
        loss = F.cross_entropy(logits, batch.answer_tokens)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if model.cell_gain.grad is None or model.cell_bias.grad is None:
            raise RuntimeError("private cell expression received no calibration gradient")
        identity_private = (
            model.cell_gain.grad.detach().square()
            + model.cell_bias.grad.detach().square()
        ).mean(-1).sqrt()
        identity_score[model.expression_cells] += identity_private
        if model.cell_adapter_down is not None:
            if (
                model.cell_adapter_down.grad is None
                or model.cell_adapter_up.grad is None
            ):
                raise RuntimeError("private cell adapters received no calibration gradient")
            adapter_down_score[model.expression_cells] += (
                model.cell_adapter_down.grad.detach()
                .square()
                .mean(dim=(1, 2))
                .sqrt()
            )
            adapter_up_score[model.expression_cells] += (
                model.cell_adapter_up.grad.detach()
                .square()
                .mean(dim=(1, 2))
                .sqrt()
            )
        if model.graph.edge_logit.grad is None:
            raise RuntimeError("edge logits received no calibration gradient")
        edge_score += model.graph.edge_logit.grad.detach().abs()
        losses.append(float(loss.detach()))
        accuracies.append(
            float(logits.detach().argmax(-1).eq(batch.answer_tokens).float().mean())
        )
        calibration_state = _detach_state(next_state)
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    identity_score /= batches
    adapter_down_score /= batches
    adapter_up_score /= batches
    edge_score = edge_score / batches * model.graph.active
    incoming, outgoing = _edge_to_cell_scores(model, edge_score)
    private_ranks = [_cell_tissue_ranks(model, identity_score)]
    if model.cell_adapter_down is not None:
        private_ranks.extend(
            [
                _cell_tissue_ranks(model, adapter_down_score),
                _cell_tissue_ranks(model, adapter_up_score),
            ]
        )
    expression_rank = torch.stack(private_ranks).mean(0)
    incoming_rank = _cell_tissue_ranks(model, incoming)
    outgoing_rank = _cell_tissue_ranks(model, outgoing)
    cell_utility = _cell_tissue_ranks(
        model,
        0.50 * expression_rank + 0.25 * incoming_rank + 0.25 * outgoing_rank,
    )
    intrinsic_edge = _edge_tissue_ranks(model, edge_score)
    sources = model.graph.sources
    target_utility = cell_utility.unsqueeze(-1).expand_as(intrinsic_edge)
    if directional_edges:
        # An incoming connection belongs to the target's computation. A useful
        # source may fan out into both mature and reserve targets, so source utility
        # alone must not protect all of its outgoing anatomy.
        edge_utility = torch.maximum(intrinsic_edge, target_utility)
    else:
        source_utility = cell_utility[sources]
        edge_utility = torch.maximum(
            intrinsic_edge,
            torch.maximum(target_utility, source_utility),
        )
    edge_utility = edge_utility * model.graph.active
    diagnostics = {
        "batches": batches,
        "steps": steps,
        "mean_loss": sum(losses) / len(losses),
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "identity_gradient_mean": float(identity_score.mean()),
        "adapter_down_gradient_mean": float(adapter_down_score.mean()),
        "adapter_up_gradient_mean": float(adapter_up_score.mean()),
        "active_edge_gradient_mean": float(edge_score[model.graph.active].mean()),
        "directional_edges": directional_edges,
    }
    return cell_utility.detach(), edge_utility.detach(), diagnostics


def _shuffle_cells_within_tissue(
    model: Sol2, values: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    result = values.clone()
    for tissue in ("input", "memory", "compute", "relay", "output"):
        cells = model.tissue_indices(tissue)
        order = torch.randperm(len(cells), generator=generator, device="cpu").to(cells.device)
        result[cells] = values[cells[order]]
    return result


def _shuffle_edges_within_target_tissue(
    model: Sol2, values: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    result = torch.zeros_like(values)
    active = model.graph.active
    for tissue in ("input", "memory", "compute", "relay", "output"):
        targets = model.tissue_indices(tissue)
        mask = active[targets]
        selected = values[targets][mask]
        order = torch.randperm(selected.numel(), generator=generator, device="cpu").to(
            selected.device
        )
        tissue_result = result[targets].clone()
        tissue_result[mask] = selected[order]
        result[targets] = tissue_result
    return result


def make_utility_profile(
    model: Sol2,
    measured_cell: torch.Tensor,
    measured_edge: torch.Tensor,
    *,
    branch: str,
    threshold: float,
    temperature: float,
    minimum_plasticity: float,
    genome_plasticity: float,
    shuffle_seed: int,
) -> UtilityProfile:
    """Construct the plastic, measured, or distribution-matched shuffled treatment."""

    if branch not in {"plastic", "uniform", "consolidated", "shuffled"}:
        raise ValueError(f"unknown consolidation branch {branch!r}")
    if not 0.0 <= minimum_plasticity <= 1.0:
        raise ValueError("minimum plasticity must be in [0, 1]")
    if temperature <= 0.0 or not 0.0 <= genome_plasticity <= 1.0:
        raise ValueError("invalid consolidation parameters")
    applied_cell = measured_cell.clone()
    applied_edge = measured_edge.clone()
    if branch == "shuffled":
        generator = torch.Generator().manual_seed(shuffle_seed)
        applied_cell = _shuffle_cells_within_tissue(model, measured_cell, generator)
        # The measured edge score already includes intrinsic, source, and target
        # utility. Permuting that final score preserves the exact treatment-strength
        # distribution while breaking its placement on learned anatomy.
        applied_edge = _shuffle_edges_within_target_tissue(
            model, measured_edge, generator
        )

    if branch == "plastic":
        cell_plasticity = torch.ones_like(applied_cell)
        edge_plasticity = torch.ones_like(applied_edge)
        applied_genome = 1.0
    else:
        cell_protection = torch.sigmoid((applied_cell - threshold) / temperature)
        edge_protection = torch.sigmoid((applied_edge - threshold) / temperature)
        cell_plasticity = 1.0 - (1.0 - minimum_plasticity) * cell_protection
        edge_plasticity = 1.0 - (1.0 - minimum_plasticity) * edge_protection
        edge_plasticity = torch.where(
            model.graph.active, edge_plasticity, torch.ones_like(edge_plasticity)
        )
        if branch == "uniform":
            active = model.graph.active
            cell_mean = cell_plasticity.mean()
            edge_mean = edge_plasticity[active].mean()
            cell_plasticity = torch.full_like(cell_plasticity, cell_mean)
            edge_plasticity = torch.where(
                active,
                torch.full_like(edge_plasticity, edge_mean),
                torch.ones_like(edge_plasticity),
            )
        applied_genome = genome_plasticity
    return UtilityProfile(
        measured_cell=measured_cell,
        measured_edge=measured_edge,
        applied_cell=applied_cell,
        applied_edge=applied_edge,
        cell_plasticity=cell_plasticity,
        edge_plasticity=edge_plasticity,
        genome_plasticity=applied_genome,
    )


class ConsolidationPolicy:
    """Scale the realized AdamW parameter delta, including weight decay."""

    def __init__(self, model: Sol2, profile: UtilityProfile):
        self.model = model
        self.profile = profile

    def _mask(self, name: str, parameter: torch.Tensor) -> torch.Tensor | float | None:
        if name in {"cell_gain", "cell_bias"}:
            return self.profile.cell_plasticity[self.model.expression_cells].unsqueeze(-1)
        if name in {"cell_adapter_down", "cell_adapter_up"}:
            return self.profile.cell_plasticity[self.model.expression_cells].view(
                -1, 1, 1
            )
        if name == "graph.edge_logit":
            return self.profile.edge_plasticity
        if name.startswith(("graph.query.", "graph.key.", "graph.value.", "tissues.")):
            return self.profile.genome_plasticity
        return None

    @torch.no_grad()
    def capture_before_step(self) -> dict[str, torch.Tensor]:
        before = {}
        for name, parameter in self.model.named_parameters():
            mask = self._mask(name, parameter)
            if parameter.grad is not None and mask is not None:
                if isinstance(mask, float) and mask >= 1.0:
                    continue
                if isinstance(mask, torch.Tensor) and bool((mask >= 1.0).all()):
                    continue
                before[name] = parameter.detach().clone()
        return before

    @torch.no_grad()
    def apply_after_step(self, before: dict[str, torch.Tensor]) -> None:
        named = dict(self.model.named_parameters())
        for name, old in before.items():
            parameter = named[name]
            mask = self._mask(name, parameter)
            assert mask is not None
            parameter.copy_(old + (parameter - old) * mask)


def profile_summary(model: Sol2, profile: UtilityProfile) -> dict:
    result = {
        "genome_plasticity": profile.genome_plasticity,
        "tissues": {},
    }
    for tissue in ("input", "memory", "compute", "relay", "output"):
        cells = model.tissue_indices(tissue)
        result["tissues"][tissue] = {
            "cells": len(cells),
            "measured_utility_mean": float(profile.measured_cell[cells].mean()),
            "applied_utility_mean": float(profile.applied_cell[cells].mean()),
            "plasticity_mean": float(profile.cell_plasticity[cells].mean()),
            "plasticity_min": float(profile.cell_plasticity[cells].min()),
        }
    active = model.graph.active
    result["active_edge_plasticity_mean"] = float(profile.edge_plasticity[active].mean())
    result["active_edge_plasticity_min"] = float(profile.edge_plasticity[active].min())
    return result
