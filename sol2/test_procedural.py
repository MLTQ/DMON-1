"""CPU contracts for the SOL2 procedural task and branching benchmark."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .config import Sol2Config
from .procedural_benchmark import run_benchmark
from .procedural_task import ProceduralTask


def test_regime_factors_are_separable() -> None:
    task = ProceduralTask()
    base = task.base_regime(seed=11)
    interface = task.remapped_interface(base, "interface", seed=12)
    procedure = task.changed_procedure(base, "procedure", seed=13)
    assert interface.operation_semantics == base.operation_semantics
    assert procedure.state_surface == base.state_surface
    assert procedure.operation_surface == base.operation_surface
    assert procedure.answer_surface == base.answer_surface
    assert procedure.operation_semantics == base.operation_semantics
    assert base.execution_order == "forward"
    assert procedure.execution_order == "reverse"


def test_latent_execution_and_surface_encoding() -> None:
    task = ProceduralTask()
    regime = task.base_regime(seed=17)
    generator = torch.Generator().manual_seed(19)
    batch = task.sample_batch(regime, 7, 5, generator, "cpu")
    assert batch.tokens.shape == (7, 8)
    assert bool((batch.tokens >= 0).all())
    assert bool((batch.tokens < task.vocab_size).all())
    assert bool((batch.answer_tokens >= task.answer_offset).all())

    value = batch.initial_values.clone()
    semantics = torch.tensor(regime.operation_semantics)
    positions = range(batch.operations.shape[1])
    if regime.execution_order == "reverse":
        positions = reversed(range(batch.operations.shape[1]))
    for position in positions:
        canonical = semantics[batch.operations[:, position]]
        value = task.operation_table[canonical, value]
    assert torch.equal(value, batch.latent_answers)


def test_reverse_order_changes_the_procedure_without_changing_inputs() -> None:
    task = ProceduralTask()
    forward = task.base_regime(seed=29)
    reverse = task.changed_procedure(forward, "reverse", seed=31)
    forward_generator = torch.Generator().manual_seed(37)
    reverse_generator = torch.Generator().manual_seed(37)
    forward_batch = task.sample_batch(forward, 64, 5, forward_generator, "cpu")
    reverse_batch = task.sample_batch(reverse, 64, 5, reverse_generator, "cpu")
    assert torch.equal(forward_batch.tokens, reverse_batch.tokens)
    assert bool((forward_batch.latent_answers != reverse_batch.latent_answers).any())


def test_tiny_branching_smoke() -> None:
    task = ProceduralTask()
    cfg = Sol2Config(
        n_input=2,
        n_memory=2,
        n_compute=4,
        n_relay=2,
        n_output=2,
        hidden=8,
        n_dendrites=4,
        initial_active_dendrites=2,
        steps_per_token=1,
        organ_queries=1,
        vocab_size=task.vocab_size,
        batch_size=2,
        updates=4,
        warmup_updates=1,
        log_every=2,
        eval_every=2,
        eval_tokens=8,
        operator_bound=4.0,
        seed=23,
        device="cpu",
    )
    with TemporaryDirectory() as directory:
        result = run_benchmark(
            "creature",
            cfg,
            Path(directory),
            acquisition_updates=2,
            adaptation_updates=2,
            min_steps=1,
            max_steps=2,
            eval_batches=1,
        )
        assert set(result["records"]) == {
            "acquisition",
            "control",
            "interface",
            "procedure",
            "interface_organs_only",
            "procedure_organs_only",
        }
        assert all(len(rows) == 2 for rows in result["records"].values())
        assert (Path(directory) / "metrics.json").exists()


def main() -> None:
    tests = (
        test_regime_factors_are_separable,
        test_latent_execution_and_surface_encoding,
        test_reverse_order_changes_the_procedure_without_changing_inputs,
        test_tiny_branching_smoke,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
