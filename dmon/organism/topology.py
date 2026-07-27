"""Directed-connectome measurements shared by benchmarks and future growth rules."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class TopologyMetrics:
    cells: int
    directed_edges: int
    self_edges: int
    reachable_cells: int
    reachable_fraction: float
    reachable_outputs: int
    output_reachable_fraction: float
    mean_output_distance: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def analyze_topology(
    sources: torch.Tensor,
    sensory_indices: torch.Tensor,
    output_indices: torch.Tensor,
    active_edges: torch.Tensor | None = None,
) -> TopologyMetrics:
    """Measure forward reachability on a target-by-dendrite source table."""

    if sources.ndim != 2:
        raise ValueError("sources must have shape (cells, dendrites)")
    cells, dendrites = sources.shape
    if cells < 1 or dendrites < 1:
        raise ValueError("sources must be non-empty")
    source_rows = sources.detach().cpu().tolist()
    if active_edges is None:
        active_edges = torch.ones_like(sources, dtype=torch.bool)
    if active_edges.shape != sources.shape:
        raise ValueError("active_edges must match sources")
    active_rows = active_edges.detach().cpu().tolist()
    sensory = [int(index) for index in sensory_indices.detach().cpu()]
    outputs = [int(index) for index in output_indices.detach().cpu()]
    if not sensory or not outputs:
        raise ValueError("sensory and output populations must be non-empty")

    adjacency: list[list[int]] = [[] for _ in range(cells)]
    self_edges = 0
    directed_edges = 0
    for target, row in enumerate(source_rows):
        for source, active in zip(
            row,
            active_rows[target],
            strict=True,
        ):
            if not active:
                continue
            source = int(source)
            if not 0 <= source < cells:
                raise ValueError("source index is outside the cell field")
            adjacency[source].append(target)
            self_edges += int(source == target)
            directed_edges += 1

    distance: list[int | None] = [None] * cells
    queue: deque[int] = deque()
    for source in sensory:
        if not 0 <= source < cells:
            raise ValueError("sensory index is outside the cell field")
        if distance[source] is None:
            distance[source] = 0
            queue.append(source)
    while queue:
        source = queue.popleft()
        assert distance[source] is not None
        for target in adjacency[source]:
            if distance[target] is None:
                distance[target] = distance[source] + 1
                queue.append(target)

    output_distances: list[int] = []
    for output in outputs:
        if not 0 <= output < cells:
            raise ValueError("output index is outside the cell field")
        if distance[output] is not None:
            output_distances.append(distance[output])
    reachable_cells = sum(value is not None for value in distance)
    return TopologyMetrics(
        cells=cells,
        directed_edges=directed_edges,
        self_edges=self_edges,
        reachable_cells=reachable_cells,
        reachable_fraction=reachable_cells / cells,
        reachable_outputs=len(output_distances),
        output_reachable_fraction=len(output_distances) / len(outputs),
        mean_output_distance=(
            sum(output_distances) / len(output_distances)
            if output_distances
            else None
        ),
    )
