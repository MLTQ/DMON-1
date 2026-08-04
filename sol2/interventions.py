"""Reversible parameter and anatomy interventions for causal evaluation."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .model import Sol2


@contextmanager
def shuffled_private_expression(model: Sol2, seed: int = 1731):
    """Permute private expression within internal tissue types, then restore it."""

    if model.cell_gain is None or model.cell_bias is None:
        yield
        return
    gain = model.cell_gain.detach().clone()
    bias = model.cell_bias.detach().clone()
    generator = torch.Generator().manual_seed(seed)
    try:
        with torch.no_grad():
            for name in ("compute", "relay"):
                cells = model.tissue_indices(name)
                positions = model.expression_pos[cells]
                permutation = torch.randperm(len(cells), generator=generator)
                source = positions.detach().cpu()[permutation].to(positions.device)
                model.cell_gain[positions] = gain[source]
                model.cell_bias[positions] = bias[source]
        yield
    finally:
        with torch.no_grad():
            model.cell_gain.copy_(gain)
            model.cell_bias.copy_(bias)


@contextmanager
def zero_private_adapters(model: Sol2, cells: torch.Tensor | None = None):
    """Temporarily remove private transition residuals from selected mutable cells."""

    if model.cell_adapter_up is None:
        yield
        return
    original = model.cell_adapter_up.detach().clone()
    positions = (
        torch.arange(len(original), device=original.device)
        if cells is None
        else model.expression_pos[cells]
    )
    try:
        with torch.no_grad():
            model.cell_adapter_up[positions] = 0.0
        yield
    finally:
        with torch.no_grad():
            model.cell_adapter_up.copy_(original)


@contextmanager
def disable_graft_edges(model: Sol2, grafts: list[dict]):
    """Temporarily make a recorded set of additive dendrites dormant."""

    original = model.graph.active.detach().clone()
    try:
        with torch.no_grad():
            for graft in grafts:
                model.graph.active[graft["target"], graft["slot"]] = False
        yield
    finally:
        with torch.no_grad():
            model.graph.active.copy_(original)


@contextmanager
def degree_preserving_rewire(model: Sol2, seed: int = 1733):
    """Shuffle installed sources while preserving target and source degrees."""

    original = model.graph.sources.detach().clone()
    generator = torch.Generator().manual_seed(seed)
    all_targets = torch.arange(model.n_cells, device=original.device)
    is_output = torch.zeros(model.n_cells, dtype=torch.bool, device=original.device)
    is_output[model.output_idx] = True
    groups = (all_targets[~is_output], model.output_idx)
    try:
        with torch.no_grad():
            for targets in groups:
                active = model.graph.active[targets]
                values = original[targets][active]
                permutation = torch.randperm(len(values), generator=generator).to(
                    values.device
                )
                rows = model.graph.sources[targets].clone()
                rows[active] = values[permutation]
                model.graph.sources[targets] = rows
        yield
    finally:
        with torch.no_grad():
            model.graph.sources.copy_(original)
