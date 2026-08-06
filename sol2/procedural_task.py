"""Synthetic procedural-composition task with separable interfaces and semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


def _permutation(size: int, generator: torch.Generator) -> tuple[int, ...]:
    return tuple(int(value) for value in torch.randperm(size, generator=generator))


@dataclass(frozen=True)
class ProcedureRegime:
    """Surface vocabulary and latent operation semantics for one environment regime."""

    name: str
    state_surface: tuple[int, ...]
    operation_surface: tuple[int, ...]
    answer_surface: tuple[int, ...]
    operation_semantics: tuple[int, ...]
    execution_order: str = "forward"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProcedureBatch:
    """One vectorized set of latent programs and their externally encoded answers."""

    tokens: torch.Tensor
    answer_tokens: torch.Tensor
    latent_answers: torch.Tensor
    operations: torch.Tensor
    initial_values: torch.Tensor


class ProceduralTask:
    """Compose small transformations while interfaces or semantics change."""

    BOS = 0
    QUERY = 1

    def __init__(self, n_values: int = 8, n_operations: int = 4) -> None:
        if n_values < 4:
            raise ValueError("n_values must be at least four")
        if n_operations != 4:
            raise ValueError("the initial benchmark defines exactly four operations")
        self.n_values = n_values
        self.n_operations = n_operations
        self.state_offset = 2
        self.operation_offset = self.state_offset + n_values
        self.answer_offset = self.operation_offset + n_operations
        self.vocab_size = self.answer_offset + n_values

        values = torch.arange(n_values, dtype=torch.long)
        self.operation_table = torch.stack(
            [
                (values + 1) % n_values,
                (values - 1) % n_values,
                (-values) % n_values,
                (3 * values + 1) % n_values,
            ]
        )

    def base_regime(self, name: str = "A", seed: int = 0) -> ProcedureRegime:
        """Create a deterministic interface with the canonical procedure semantics."""

        generator = torch.Generator().manual_seed(seed)
        return ProcedureRegime(
            name=name,
            state_surface=_permutation(self.n_values, generator),
            operation_surface=_permutation(self.n_operations, generator),
            answer_surface=_permutation(self.n_values, generator),
            operation_semantics=tuple(range(self.n_operations)),
            execution_order="forward",
        )

    def remapped_interface(
        self, base: ProcedureRegime, name: str, seed: int
    ) -> ProcedureRegime:
        """Keep the algorithm fixed while replacing every sensor/effector code."""

        generator = torch.Generator().manual_seed(seed)
        candidate = ProcedureRegime(
            name=name,
            state_surface=_permutation(self.n_values, generator),
            operation_surface=_permutation(self.n_operations, generator),
            answer_surface=_permutation(self.n_values, generator),
            operation_semantics=base.operation_semantics,
            execution_order=base.execution_order,
        )
        if (
            candidate.state_surface == base.state_surface
            and candidate.operation_surface == base.operation_surface
            and candidate.answer_surface == base.answer_surface
        ):
            raise RuntimeError("interface remap accidentally reproduced the base regime")
        return candidate

    def changed_procedure(
        self, base: ProcedureRegime, name: str, seed: int
    ) -> ProcedureRegime:
        """Keep symbols and primitives fixed while reversing composition order."""

        del seed  # Reserved for later seed-addressable procedure families.
        return ProcedureRegime(
            name=name,
            state_surface=base.state_surface,
            operation_surface=base.operation_surface,
            answer_surface=base.answer_surface,
            operation_semantics=base.operation_semantics,
            execution_order="reverse",
        )

    def sample_batch(
        self,
        regime: ProcedureRegime,
        batch_size: int,
        steps: int,
        generator: torch.Generator,
        device: str | torch.device,
    ) -> ProcedureBatch:
        """Sample programs, execute them latently, and encode only their interface."""

        if batch_size < 1 or steps < 1:
            raise ValueError("batch_size and steps must be positive")
        initial = torch.randint(
            self.n_values, (batch_size,), generator=generator, dtype=torch.long
        )
        operations = torch.randint(
            self.n_operations,
            (batch_size, steps),
            generator=generator,
            dtype=torch.long,
        )
        value = initial.clone()
        semantics = torch.tensor(regime.operation_semantics, dtype=torch.long)
        positions = range(steps)
        if regime.execution_order == "reverse":
            positions = reversed(range(steps))
        elif regime.execution_order != "forward":
            raise ValueError(f"unknown execution order {regime.execution_order!r}")
        for position in positions:
            canonical = semantics[operations[:, position]]
            value = self.operation_table[canonical, value]

        state_surface = torch.tensor(regime.state_surface, dtype=torch.long)
        operation_surface = torch.tensor(regime.operation_surface, dtype=torch.long)
        answer_surface = torch.tensor(regime.answer_surface, dtype=torch.long)
        state_tokens = self.state_offset + state_surface[initial]
        operation_tokens = self.operation_offset + operation_surface[operations]
        answer_tokens = self.answer_offset + answer_surface[value]
        tokens = torch.cat(
            [
                torch.full((batch_size, 1), self.BOS, dtype=torch.long),
                state_tokens[:, None],
                operation_tokens,
                torch.full((batch_size, 1), self.QUERY, dtype=torch.long),
            ],
            dim=1,
        )
        return ProcedureBatch(
            tokens=tokens.to(device),
            answer_tokens=answer_tokens.to(device),
            latent_answers=value.to(device),
            operations=operations.to(device),
            initial_values=initial.to(device),
        )
