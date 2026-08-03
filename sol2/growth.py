"""Append-only relay-tissue growth with optimizer and state preservation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .model import Sol2
from .state import OrganismState


def _replace_resized_parameter(
    owner: nn.Module,
    name: str,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    old_rows: int,
) -> nn.Parameter:
    old = getattr(owner, name)
    new = nn.Parameter(data)
    for group in optimizer.param_groups:
        group["params"] = [new if parameter is old else parameter for parameter in group["params"]]
    if old in optimizer.state:
        old_state = optimizer.state.pop(old)
        new_state = {}
        for key, value in old_state.items():
            if torch.is_tensor(value) and value.shape == old.shape:
                padding = torch.zeros_like(data[old_rows:])
                new_state[key] = torch.cat([value, padding], dim=0)
            else:
                new_state[key] = value
        optimizer.state[new] = new_state
    setattr(owner, name, new)
    return new


def grow_relay_tissue(
    model: Sol2,
    optimizer: torch.optim.Optimizer,
    n_new: int,
) -> tuple[list[int], Callable[[OrganismState], OrganismState], list[dict]]:
    """Append relay cells and weakly connect each through a dormant slot.

    No installed edge is removed. Existing tensors and optimizer moments keep their
    rows exactly; new parameter rows begin with zero moments.
    """

    if n_new < 1:
        raise ValueError("n_new must be positive")
    device = model.mutable_idx.device
    old_n = model.n_cells
    new_ids = torch.arange(old_n, old_n + n_new, device=device)
    graph = model.graph
    generator = torch.Generator().manual_seed(model.cfg.seed + old_n + n_new)

    source_pool = torch.cat(
        [
            model.input_idx.cpu(),
            model.memory_idx.cpu(),
            model.compute_idx.cpu(),
            model.relay_idx.cpu(),
            new_ids.cpu(),
        ]
    )
    source_rows = torch.empty(n_new, graph.n_dendrites, dtype=torch.long)
    for offset, target in enumerate(new_ids.cpu().tolist()):
        pool = source_pool[source_pool != target]
        if len(pool) >= graph.n_dendrites:
            picks = pool[
                torch.randperm(len(pool), generator=generator)[: graph.n_dendrites]
            ]
        else:
            extra = pool[
                torch.randint(
                    len(pool),
                    (graph.n_dendrites - len(pool),),
                    generator=generator,
                )
            ]
            picks = torch.cat([pool, extra])
        picks[0] = model.input_idx.cpu()[offset % len(model.input_idx)]
        source_rows[offset] = picks

    new_active_rows = torch.zeros(n_new, graph.n_dendrites, dtype=torch.bool)
    new_active_rows[:, : model.cfg.initial_active_dendrites] = True
    sources = torch.cat([graph.sources.cpu(), source_rows], dim=0).to(device)
    active = torch.cat([graph.active.cpu(), new_active_rows], dim=0).to(device)

    # Every new cell is read by one existing target. Prefer output and relay tissue so
    # a new transport path can affect behavior, but use only dormant slots.
    donors = torch.cat([model.output_idx, model.relay_idx, model.compute_idx, model.input_idx])
    available = [
        (int(target), int(slot))
        for target in donors.tolist()
        for slot in torch.nonzero(~active[target], as_tuple=True)[0].tolist()
        if not (target in set(model.output_idx.tolist()) and slot == 0)
    ]
    if len(available) < n_new:
        raise RuntimeError("not enough dormant dendrite slots for silent relay growth")

    grafts = []
    for new_id, (target, slot) in zip(new_ids.tolist(), available):
        sources[target, slot] = new_id
        active[target, slot] = True
        grafts.append({"target": target, "slot": slot, "source": new_id})

    edge_rows = torch.empty(n_new, graph.n_dendrites).uniform_(
        -0.05, 0.05, generator=generator
    )
    edge_data = torch.cat([graph.edge_logit.detach().cpu(), edge_rows], dim=0).to(device)
    weak_raw = -3.0 if model.cfg.bounded_operators else -4.0
    for graft in grafts:
        edge_data[graft["target"], graft["slot"]] = weak_raw
    _replace_resized_parameter(graph, "edge_logit", edge_data, optimizer, old_n)
    graph.sources = sources
    graph.active = active
    graph.n_cells = old_n + n_new

    if model.cell_gain is not None:
        zero_rows = torch.zeros(n_new, model.cfg.hidden, device=device)
        gain_data = torch.cat([model.cell_gain.detach(), zero_rows], dim=0)
        bias_data = torch.cat([model.cell_bias.detach(), zero_rows.clone()], dim=0)
        _replace_resized_parameter(model, "cell_gain", gain_data, optimizer, old_n)
        _replace_resized_parameter(model, "cell_bias", bias_data, optimizer, old_n)

    model.relay_idx = torch.cat([model.relay_idx, new_ids])
    model._rebuild_mutable()
    model.cfg = model.cfg.scaled(n_relay=model.cfg.n_relay + n_new)

    required = torch.cat([model.output_idx, new_ids])
    if not graph.reachable(model.input_idx, required):
        raise RuntimeError("growth violated sensory reachability")
    if not graph.output_cells_are_sinks(model.output_idx):
        raise RuntimeError("growth made an output cell a message source")

    def migrate(state: OrganismState) -> OrganismState:
        padding = state.hidden.new_zeros(
            state.hidden.shape[0], n_new, state.hidden.shape[2]
        )
        return OrganismState(
            hidden=torch.cat([state.hidden, padding], dim=1),
            memory_cursor=state.memory_cursor,
            weight_version=state.weight_version,
        )

    return new_ids.tolist(), migrate, grafts
