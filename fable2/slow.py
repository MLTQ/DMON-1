"""M1c: graft an M0-anatomy checkpoint into a slow-tissue organism.

Slow tissue appends after relay, so every pre-existing tensor row keeps its
exact index — the graft is a prefix copy with an exact census, never a
strict=False load. New slow rows keep their fresh constructor initialization;
one dormant dendrite slot per compute and output cell is rewired onto a slow
source at the growth machinery's weak-logit convention.
"""

from __future__ import annotations

import torch

from .system import BrocaSystem

GRAFT_EDGE_RAW_LOGIT = -3.0


def graft_from_checkpoint(payload: dict, system: BrocaSystem) -> dict:
    """Copy an old-anatomy checkpoint into a slow-augmented system, with census.

    Returns the census: every checkpoint tensor is accounted as `identical`
    (same shape, copied whole) or `prefix` (old rows copied into the leading
    rows of a row-extended tensor). Any tensor fitting neither pattern raises.
    """

    if payload.get("schema") != 1:
        raise ValueError("unsupported fable2 checkpoint schema")
    if tuple(payload["depths"]) != system.depths:
        raise ValueError("checkpoint depths do not match the grafted system")
    organism = system.organism
    if organism.cfg.n_slow < 1:
        raise ValueError("graft target must have slow tissue (n_slow > 0)")

    old_state = payload["organism"]
    new_state = organism.state_dict()
    missing = set(old_state) - set(new_state)
    if missing:
        raise ValueError(f"checkpoint tensors absent from graft target: {sorted(missing)[:5]}")

    census = {"identical": 0, "prefix": 0}
    with torch.no_grad():
        for name, old_tensor in old_state.items():
            new_tensor = new_state[name]
            if old_tensor.shape == new_tensor.shape:
                new_tensor.copy_(old_tensor)
                census["identical"] += 1
            elif (
                old_tensor.dim() == new_tensor.dim()
                and old_tensor.shape[1:] == new_tensor.shape[1:]
                and old_tensor.shape[0] < new_tensor.shape[0]
            ):
                new_tensor[: old_tensor.shape[0]].copy_(old_tensor)
                census["prefix"] += 1
            else:
                raise ValueError(
                    f"graft cannot place {name}: {tuple(old_tensor.shape)} -> "
                    f"{tuple(new_tensor.shape)}"
                )
    organism.load_state_dict(new_state)
    if census["identical"] + census["prefix"] != len(old_state):
        raise RuntimeError("graft census does not account for every checkpoint tensor")

    census["rewired_slots"] = rewire_dormant_slots_to_slow(organism)
    census["source_update"] = int(payload["update"])
    return census


def rewire_dormant_slots_to_slow(organism) -> int:
    """Point one dormant slot per compute/output cell at a slow source.

    Weak raw logit −3.0 (the growth convention): present enough to carry
    gradient, weak enough that behavior stays near the ungrafted computation.
    Returns the number of slots rewired; the slot list is recoverable as
    `sources[cell, slot] in slow_idx`.
    """

    graph = organism.graph
    slow_cells = organism.slow_idx.tolist()
    rewired = 0
    with torch.no_grad():
        targets = torch.cat([organism.compute_idx, organism.output_idx]).tolist()
        for position, cell in enumerate(targets):
            dormant = torch.nonzero(~graph.active[cell], as_tuple=True)[0]
            if not len(dormant):
                continue
            slot = int(dormant[0])
            graph.sources[cell, slot] = slow_cells[position % len(slow_cells)]
            graph.active[cell, slot] = True
            graph.edge_logit[cell, slot] = GRAFT_EDGE_RAW_LOGIT
            rewired += 1
    if rewired == 0:
        raise RuntimeError("no dormant slots available to graft slow tissue into")
    if not graph.output_cells_are_sinks(organism.output_idx):
        raise RuntimeError("graft violated the output-sink invariant")
    if not graph.targets_read_only_from(organism.output_idx, organism.internal_idx):
        raise RuntimeError("graft let an output cell read outside internal tissue")
    return rewired


def deactivate_grafted_slots(organism) -> int:
    """Turn grafted slow-reading slots back off (contract-testing helper)."""

    graph = organism.graph
    slow = set(organism.slow_idx.tolist())
    deactivated = 0
    with torch.no_grad():
        for cell in torch.cat([organism.compute_idx, organism.output_idx]).tolist():
            for slot in torch.nonzero(graph.active[cell], as_tuple=True)[0].tolist():
                if int(graph.sources[cell, slot]) in slow:
                    graph.active[cell, slot] = False
                    deactivated += 1
    return deactivated
