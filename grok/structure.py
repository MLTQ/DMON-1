"""Bounded structural phase: causal probe credit + optional global fitness gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TrainConfig
from .model import StreamingCreature


@dataclass(slots=True)
class StructuralReport:
    rewires: int
    total_rewires: int
    mean_probe_credit: float
    mean_probe_fitness: float


@torch.no_grad()
def rotate_probes(model: StreamingCreature) -> None:
    """Advance each target's candidate to the next non-edge source."""

    n = model.config.n_cells
    sources = model.graph.sources
    probes = model.probe_sources.clone()
    for t in range(n):
        used = set(sources[t].tolist()) | {t}
        cur = int(probes[t].item())
        nxt = (cur + 1) % n
        for _ in range(n):
            if nxt not in used:
                probes[t] = nxt
                break
            nxt = (nxt + 1) % n
    model.probe_sources.copy_(probes)
    model.probe_confirmations.zero_()


@torch.no_grad()
def apply_structural_phase(
    model: StreamingCreature,
    *,
    probe_credit: torch.Tensor,
    probe_fitness: torch.Tensor,
    cfg: TrainConfig,
) -> StructuralReport:
    """Replace at most a few mature edges when probes beat incumbents.

    Requires consecutive confirmations, positive local credit, optional positive
    global fitness (S17), and preserved sensory→output reachability.
    """

    if cfg.structural_probe_gain <= 0:
        return StructuralReport(0, int(model.total_rewires.item()), 0.0, 0.0)

    n, k = cfg.n_cells, cfg.n_dendrites
    sources = model.graph.sources
    rewires = 0
    # Rank targets by probe credit.
    order = torch.argsort(probe_credit, descending=True)
    budget = max(1, n // 16)

    for t in order.tolist():
        if rewires >= budget:
            break
        local = float(probe_credit[t].item())
        global_fit = float(probe_fitness[t].item())
        if local < cfg.structural_margin:
            model.probe_confirmations[t] = 0
            continue
        if cfg.structural_require_global_fitness and global_fit <= 0:
            model.probe_confirmations[t] = 0
            continue
        model.probe_confirmations[t] += 1
        if int(model.probe_confirmations[t].item()) < cfg.structural_confirmations:
            continue

        # Replace the weakest slow weight among this target's dendrites.
        slot = int(torch.argmin(model.graph.weights[t].abs()).item())
        candidate = int(model.probe_sources[t].item())
        # No duplicate sources.
        if candidate in sources[t].tolist():
            model.probe_confirmations[t] = 0
            continue
        # Tentative swap + reachability check.
        old = int(sources[t, slot].item())
        sources[t, slot] = candidate
        if not _outputs_reachable(model):
            sources[t, slot] = old
            model.probe_confirmations[t] = 0
            continue
        # Reset that slot's slow parameters; keep others.
        model.graph.weights.data[t, slot].zero_()
        model.graph.bias.data[t, slot].zero_()
        model.probe_confirmations[t] = 0
        rewires += 1
        model.total_rewires += 1

    if rewires:
        rotate_probes(model)
    else:
        # Still rotate occasionally so candidates explore.
        if int(model.total_rewires.item()) % 1 == 0:
            # Mild exploration: rotate all probes each phase without rewire.
            rotate_probes(model)

    return StructuralReport(
        rewires=rewires,
        total_rewires=int(model.total_rewires.item()),
        mean_probe_credit=float(probe_credit.mean().item()),
        mean_probe_fitness=float(probe_fitness.mean().item()),
    )


@torch.no_grad()
def _outputs_reachable(model: StreamingCreature) -> bool:
    """BFS from sensory cells along directed dendrites; all outputs must be reached."""

    n = model.config.n_cells
    adj: list[list[int]] = [[] for _ in range(n)]
    sources = model.graph.sources
    for t in range(n):
        for s in sources[t].tolist():
            adj[int(s)].append(t)
    seen = set(model.input_idx.tolist())
    stack = list(seen)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return all(int(o) in seen for o in model.output_idx.tolist())


def accumulate_global_fitness(
    probe_effect_hidden: torch.Tensor | None,
    hidden_grad: torch.Tensor | None,
) -> torch.Tensor | None:
    """First-order organism fitness: −⟨probe_effect, ∂L/∂h⟩ per target cell.

    Positive value means the probe direction reduces loss.
    """

    if probe_effect_hidden is None or hidden_grad is None:
        return None
    # probe_effect and grad: [B, N, H] → fitness [N]
    return -(probe_effect_hidden.detach() * hidden_grad.detach()).mean(dim=(0, 2))
