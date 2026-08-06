"""Cumulative checkpoint anchors and plasticity-pressure measurement for SOL2."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Sol2


@dataclass
class AnchorProfile:
    """Branch-specific protection strengths on mature cells, edges, and genome."""

    measured_cell: torch.Tensor
    measured_edge: torch.Tensor
    cell_protection: torch.Tensor
    edge_protection: torch.Tensor
    genome_protection: float
    anchor_rate: float

    def cpu_payload(self) -> dict:
        return {
            "measured_cell": self.measured_cell.detach().cpu(),
            "measured_edge": self.measured_edge.detach().cpu(),
            "cell_protection": self.cell_protection.detach().cpu(),
            "edge_protection": self.edge_protection.detach().cpu(),
            "genome_protection": self.genome_protection,
            "anchor_rate": self.anchor_rate,
        }


def make_anchor_profile(
    model: Sol2,
    measured_cell: torch.Tensor,
    measured_edge: torch.Tensor,
    *,
    branch: str,
    threshold: float,
    temperature: float,
    anchor_rate: float,
) -> AnchorProfile:
    """Build plastic, uniform-anchor, or measured-anchor protection strengths."""

    if branch not in {"plastic", "uniform_anchor", "measured_anchor", "developmental"}:
        raise ValueError(f"unknown anchor branch {branch!r}")
    if temperature <= 0.0 or anchor_rate < 0.0:
        raise ValueError("anchor temperature must be positive and rate nonnegative")
    if branch == "plastic":
        cell = torch.zeros_like(measured_cell)
        edge = torch.zeros_like(measured_edge)
        genome = 0.0
    else:
        cell = torch.sigmoid((measured_cell - threshold) / temperature)
        edge = torch.sigmoid((measured_edge - threshold) / temperature)
        edge = torch.where(model.graph.active, edge, torch.zeros_like(edge))
        if branch == "uniform_anchor":
            cell = torch.full_like(cell, cell.mean())
            edge_mean = edge[model.graph.active].mean()
            edge = torch.where(
                model.graph.active,
                torch.full_like(edge, edge_mean),
                torch.zeros_like(edge),
            )
        genome = 1.0
    return AnchorProfile(
        measured_cell=measured_cell,
        measured_edge=measured_edge,
        cell_protection=cell,
        edge_protection=edge,
        genome_protection=genome,
        anchor_rate=anchor_rate,
    )


def _is_substrate(name: str) -> bool:
    return not name.startswith(
        (
            "embedding.",
            "sensory_gain",
            "sensory_bias",
            "output_organ.",
            "attached_organs.",
        )
    )


class ProximalAnchorPolicy:
    """Restore mature parameters toward A after each accepted optimizer step."""

    def __init__(self, model: Sol2, profile: AnchorProfile):
        self.model = model
        self.profile = profile
        self.anchors = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if _is_substrate(name)
        }
        self.base_expression_rows = len(model.expression_cells)
        self.base_cells = model.n_cells

    def _row_strength(
        self, name: str, parameter: torch.Tensor
    ) -> tuple[torch.Tensor | float | None, int | None]:
        if name in {"cell_gain", "cell_bias"}:
            strength = self.profile.cell_protection[
                self.model.expression_cells[: self.base_expression_rows]
            ].unsqueeze(-1)
            return strength, self.base_expression_rows
        if name in {"cell_adapter_down", "cell_adapter_up"}:
            strength = self.profile.cell_protection[
                self.model.expression_cells[: self.base_expression_rows]
            ].view(-1, 1, 1)
            return strength, self.base_expression_rows
        if name == "graph.edge_logit":
            return self.profile.edge_protection, self.base_cells
        if name.startswith(("graph.query.", "graph.key.", "graph.value.", "tissues.")):
            return self.profile.genome_protection, None
        return None, None

    @torch.no_grad()
    def apply_proximal(self) -> None:
        """Apply the exact proximal operator of the quadratic anchor energy."""

        if self.profile.anchor_rate == 0.0:
            return
        named = dict(self.model.named_parameters())
        for name, anchor in self.anchors.items():
            parameter = named[name]
            strength, rows = self._row_strength(name, parameter)
            if strength is None:
                continue
            if isinstance(strength, float):
                if strength == 0.0:
                    continue
                parameter.copy_(
                    anchor
                    + (parameter - anchor)
                    / (1.0 + self.profile.anchor_rate * strength)
                )
                continue
            assert rows is not None
            current = parameter[:rows]
            parameter[:rows].copy_(
                anchor
                + (current - anchor)
                / (1.0 + self.profile.anchor_rate * strength)
            )

    @torch.no_grad()
    def gradient_pressure(self) -> dict[str, float]:
        """Measure how much mature-substrate gradient demand meets anchor stiffness."""

        anchored = torch.zeros((), device=self.model.mutable_idx.device)
        total = torch.zeros_like(anchored)
        named = dict(self.model.named_parameters())
        for name, anchor in self.anchors.items():
            parameter = named[name]
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            strength, rows = self._row_strength(name, parameter)
            if strength is None:
                continue
            total += gradient.square().sum()
            if isinstance(strength, float):
                anchored += strength * gradient.square().sum()
            else:
                assert rows is not None
                anchored += (gradient[:rows].square() * strength).sum()
        total_value = float(total)
        anchored_value = float(anchored)
        pressure = anchored_value / max(total_value, 1e-20)
        return {
            "anchored_gradient_squared": anchored_value,
            "substrate_gradient_squared": total_value,
            "accessible_gradient_squared": max(total_value - anchored_value, 0.0),
            "pressure": pressure,
        }

    @torch.no_grad()
    def anchor_energy(self) -> float:
        """Return protection-weighted mean squared mature displacement."""

        weighted = torch.zeros((), device=self.model.mutable_idx.device)
        weight = torch.zeros_like(weighted)
        named = dict(self.model.named_parameters())
        for name, anchor in self.anchors.items():
            parameter = named[name]
            strength, rows = self._row_strength(name, parameter)
            if strength is None:
                continue
            if isinstance(strength, float):
                weighted += strength * (parameter - anchor).square().sum()
                weight += strength * parameter.numel()
            else:
                assert rows is not None
                expanded = strength.expand_as(parameter[:rows])
                weighted += (expanded * (parameter[:rows] - anchor).square()).sum()
                weight += expanded.sum()
        return float(weighted / weight.clamp_min(1e-20))


def anchor_profile_summary(model: Sol2, profile: AnchorProfile) -> dict:
    result = {
        "anchor_rate": profile.anchor_rate,
        "genome_protection": profile.genome_protection,
        "tissues": {},
    }
    for tissue in ("input", "memory", "compute", "relay", "output"):
        cells = model.tissue_indices(tissue)
        mature = cells[cells < len(profile.cell_protection)]
        result["tissues"][tissue] = {
            "mature_cells": len(mature),
            "measured_utility_mean": float(profile.measured_cell[mature].mean()),
            "protection_mean": float(profile.cell_protection[mature].mean()),
            "protection_max": float(profile.cell_protection[mature].max()),
        }
    active = model.graph.active[: len(profile.edge_protection)]
    result["installed_edge_protection_mean"] = float(
        profile.edge_protection[active].mean()
    )
    return result
