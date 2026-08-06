"""Persistent organism state that survives optimizer and asynchronous boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OrganismState:
    hidden: torch.Tensor
    memory_cursor: int = 0
    weight_version: int = 0

    def detach(self) -> "OrganismState":
        return OrganismState(
            hidden=self.hidden.detach(),
            memory_cursor=self.memory_cursor,
            weight_version=self.weight_version,
        )

    def clone_detached(self) -> "OrganismState":
        return OrganismState(
            hidden=self.hidden.detach().clone(),
            memory_cursor=self.memory_cursor,
            weight_version=self.weight_version,
        )
