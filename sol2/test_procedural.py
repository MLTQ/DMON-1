"""CPU contracts for the SOL2 procedural task and branching benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .anchored_consolidation import ProximalAnchorPolicy, make_anchor_profile
from .checkpoint import same_tensor_values
from .config import Sol2Config
from .consolidated_attachment import run_consolidated_attachment_branch
from .consolidated_attachment_analysis import analyze as analyze_consolidation
from .consolidation import (
    ConsolidationPolicy,
    calibrate_causal_utility,
    make_utility_profile,
)
from .development import DevelopmentController
from .developmental_attachment import run_developmental_attachment_branch
from .developmental_analysis import analyze as analyze_development
from .growth import grow_relay_tissue
from .model import Sol2
from .genome_rate_screen import select as select_genome_rate
from .organ_attachment import run_organ_attachment_branch
from .optim import add_optimizer_parameters, build_optimizer
from .procedural_acquisition import run_acquisition_calibration
from .procedural_benchmark import run_benchmark
from .procedural_task import ProceduralTask
from .private_transition_analysis import analyze as analyze_private_transition
from .train import build_model


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
        cell_adapter_rank=2,
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
            "interface_scratch",
            "procedure",
            "interface_organs_only",
            "procedure_organs_only",
        }
        assert all(len(rows) == 2 for rows in result["records"].values())
        lesions = result["summary"]["branches"]["control"]["trained_length"]
        assert "freeze_compute" in lesions
        assert "freeze_relay" in lesions
        assert (Path(directory) / "metrics.json").exists()


def test_acquisition_checkpoint_and_resume() -> None:
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
        cell_adapter_rank=2,
        vocab_size=task.vocab_size,
        batch_size=2,
        updates=4,
        warmup_updates=1,
        log_every=1,
        operator_bound=4.0,
        seed=41,
        device="cpu",
    )
    with TemporaryDirectory() as directory:
        out_dir = Path(directory)
        first = run_acquisition_calibration(
            cfg,
            out_dir,
            max_updates=2,
            evaluation_interval=1,
            stage_updates=1,
            eval_batches=1,
            mastery_accuracy=2.0,
            mastery_checks=2,
        )
        assert first["update"] == 2
        resumed = run_acquisition_calibration(
            cfg,
            out_dir,
            max_updates=4,
            evaluation_interval=1,
            stage_updates=1,
            eval_batches=1,
            mastery_accuracy=2.0,
            mastery_checks=2,
            resume=True,
        )
        assert resumed["update"] == 4
        assert len(resumed["evaluations"]) == 4
        assert len(resumed["evaluations"][-1]["telemetry"]["cells"]) == cfg.n_cells
        assert (out_dir / "acquisition.pt").exists()


def test_distinct_organ_attachment_is_behaviorally_silent_for_a() -> None:
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
        seed=47,
        device="cpu",
    )
    model = build_model("creature", cfg, "cpu")
    assert isinstance(model, Sol2)
    optimizer = build_optimizer(model, cfg)
    state = model.initial_state(cfg.batch_size, "cpu")
    tokens = torch.tensor([task.BOS, task.QUERY])
    rng_before = torch.get_rng_state().clone()
    logits_before, state_before, _ = model.step(tokens, state.clone_detached())
    organ = model.attach_organ("B", seed=100_007)
    assert torch.equal(torch.get_rng_state(), rng_before)
    add_optimizer_parameters(
        optimizer,
        organ.named_parameters(prefix="attached_organs.B"),
        cfg,
    )
    logits_after, state_after, _ = model.step(tokens, state.clone_detached())
    assert torch.equal(logits_before, logits_after)
    torch.testing.assert_close(
        state_before.hidden, state_after.hidden, rtol=0.0, atol=2e-8
    )
    assert not ({id(p) for p in model.organ_parameters("A")} & {id(p) for p in model.organ_parameters("B")})

    b_logits, b_state, _ = model.step(
        tokens, state.clone_detached(), organ_name="B"
    )
    detached = model.detach_organ("B")
    assert detached is organ
    model.reattach_organ("B", detached)
    restored_logits, restored_state, _ = model.step(
        tokens, state.clone_detached(), organ_name="B"
    )
    assert torch.equal(b_logits, restored_logits)
    torch.testing.assert_close(
        b_state.hidden, restored_state.hidden, rtol=0.0, atol=2e-8
    )


def test_causal_utility_and_realized_update_protection() -> None:
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
        warmup_updates=1,
        operator_bound=4.0,
        seed=49,
        cell_adapter_rank=2,
        device="cpu",
    )
    model = build_model("creature", cfg, "cpu")
    assert isinstance(model, Sol2)
    state = model.initial_state(cfg.batch_size, "cpu")
    regime = task.base_regime(seed=cfg.seed + 101)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    cells, edges, _ = calibrate_causal_utility(
        model,
        state,
        task,
        regime,
        cfg,
        batches=2,
        steps=2,
        generator_seed=cfg.seed + 30_000,
        directional_edges=True,
    )
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert float(cells.min()) >= 0.0 and float(cells.max()) <= 1.0
    assert bool((edges[~model.graph.active] == 0).all())

    consolidated = make_utility_profile(
        model,
        cells,
        edges,
        branch="consolidated",
        threshold=0.65,
        temperature=0.10,
        minimum_plasticity=0.02,
        genome_plasticity=0.05,
        shuffle_seed=cfg.seed + 31_000,
    )
    shuffled = make_utility_profile(
        model,
        cells,
        edges,
        branch="shuffled",
        threshold=0.65,
        temperature=0.10,
        minimum_plasticity=0.02,
        genome_plasticity=0.05,
        shuffle_seed=cfg.seed + 31_000,
    )
    uniform = make_utility_profile(
        model,
        cells,
        edges,
        branch="uniform",
        threshold=0.65,
        temperature=0.10,
        minimum_plasticity=0.02,
        genome_plasticity=0.05,
        shuffle_seed=cfg.seed + 31_000,
    )
    assert torch.allclose(
        uniform.cell_plasticity,
        torch.full_like(uniform.cell_plasticity, consolidated.cell_plasticity.mean()),
    )
    active = model.graph.active
    assert torch.allclose(
        uniform.edge_plasticity[active],
        torch.full_like(
            uniform.edge_plasticity[active],
            consolidated.edge_plasticity[active].mean(),
        ),
    )
    for tissue in ("input", "memory", "compute", "relay", "output"):
        tissue_cells = model.tissue_indices(tissue)
        assert torch.equal(
            consolidated.applied_cell[tissue_cells].sort().values,
            shuffled.applied_cell[tissue_cells].sort().values,
        )
        mask = model.graph.active[tissue_cells]
        assert torch.equal(
            consolidated.applied_edge[tissue_cells][mask].sort().values,
            shuffled.applied_edge[tissue_cells][mask].sort().values,
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    policy = ConsolidationPolicy(model, consolidated)
    old_rule = model.tissues.rules["compute"].target.bias.detach().clone()
    old_cell = model.cell_gain.detach().clone()
    old_adapter = model.cell_adapter_up.detach().clone()
    captured = policy.capture_before_step()
    optimizer.step()
    policy.apply_after_step(captured)
    torch.testing.assert_close(
        model.tissues.rules["compute"].target.bias - old_rule,
        torch.full_like(old_rule, -0.005),
    )
    expected_cell = -0.1 * consolidated.cell_plasticity[
        model.expression_cells
    ].unsqueeze(-1).expand_as(model.cell_gain)
    torch.testing.assert_close(model.cell_gain - old_cell, expected_cell)
    expected_adapter = -0.1 * consolidated.cell_plasticity[
        model.expression_cells
    ].view(-1, 1, 1).expand_as(model.cell_adapter_up)
    torch.testing.assert_close(
        model.cell_adapter_up - old_adapter, expected_adapter
    )


def test_proximal_anchor_is_exact_and_growth_rows_remain_free() -> None:
    task = ProceduralTask()
    cfg = Sol2Config(
        n_input=2,
        n_memory=2,
        n_compute=4,
        n_relay=4,
        n_output=2,
        hidden=8,
        n_dendrites=5,
        initial_active_dendrites=2,
        steps_per_token=1,
        organ_queries=1,
        cell_adapter_rank=2,
        vocab_size=task.vocab_size,
        batch_size=2,
        seed=51,
        device="cpu",
    )
    model = Sol2(cfg)
    optimizer = build_optimizer(model, cfg)
    cells = torch.ones(model.n_cells)
    edges = torch.ones_like(model.graph.edge_logit) * model.graph.active
    profile = make_anchor_profile(
        model,
        cells,
        edges,
        branch="measured_anchor",
        threshold=0.65,
        temperature=0.10,
        anchor_rate=0.10,
    )
    policy = ProximalAnchorPolicy(model, profile)
    anchor = model.cell_gain.detach().clone()
    model.cell_gain.grad = torch.ones_like(model.cell_gain)
    pressure = policy.gradient_pressure()
    assert pressure["pressure"] > 0.95
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    policy.apply_proximal()
    strength = profile.cell_protection[model.expression_cells].unsqueeze(-1)
    torch.testing.assert_close(
        model.cell_gain,
        anchor - 0.1 / (1.0 + 0.10 * strength),
    )

    old_rows = len(model.cell_gain)
    _, _, _ = grow_relay_tissue(model, optimizer, 2)
    with torch.no_grad():
        model.cell_gain[:old_rows].add_(1.0)
        model.cell_gain[old_rows:].add_(1.0)
    mature_before_pull = model.cell_gain[:old_rows].detach().clone()
    policy.apply_proximal()
    torch.testing.assert_close(
        model.cell_gain[:old_rows],
        anchor + (mature_before_pull - anchor) / (1.0 + 0.10 * strength),
    )
    assert float(model.cell_gain[old_rows:].detach().mean()) == 1.0


def test_development_controller_requires_persistent_pressure_and_refractory() -> None:
    controller = DevelopmentController(
        high_pressure=0.75,
        plateau_pressure=0.60,
        plateau_gain=0.03,
        patience_checks=2,
        min_update=600,
        refractory_updates=600,
        max_events=2,
    )
    first = controller.observe(update=300, b_accuracy=0.20, pressure=0.90)
    second = controller.observe(update=600, b_accuracy=0.25, pressure=0.90)
    assert not first.trigger and second.trigger
    blocked = controller.observe(update=900, b_accuracy=0.26, pressure=0.90)
    assert not blocked.trigger
    state = controller.state_dict()
    restored = DevelopmentController()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    triggered = restored.observe(update=1200, b_accuracy=0.27, pressure=0.90)
    capped = restored.observe(update=1500, b_accuracy=0.28, pressure=0.90)
    assert triggered.trigger and not capped.trigger


def test_checkpoint_tensor_invariants_compare_values_not_storage() -> None:
    reference = torch.tensor([0.0, 0.25, 1.0])
    independent = reference.clone()
    assert independent.data_ptr() != reference.data_ptr()
    assert same_tensor_values(reference, independent)
    independent[-1] = 0.5
    assert not same_tensor_values(reference, independent)


def test_tiny_true_organ_branch_integrity() -> None:
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
        cell_adapter_rank=2,
        vocab_size=task.vocab_size,
        batch_size=2,
        updates=2,
        warmup_updates=1,
        log_every=1,
        operator_bound=4.0,
        seed=53,
        device="cpu",
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        acquisition_dir = root / "acquisition"
        run_acquisition_calibration(
            cfg,
            acquisition_dir,
            max_updates=2,
            evaluation_interval=1,
            stage_updates=1,
            eval_batches=1,
            mastery_accuracy=2.0,
            mastery_checks=2,
        )
        full = run_organ_attachment_branch(
            acquisition_dir / "acquisition.pt",
            root / "full",
            "full",
            device="cpu",
            adaptation_updates=1,
            eval_every=1,
            eval_batches=1,
            final_eval_batches=1,
            max_steps=2,
            a_detached_updates=1,
            recovery_updates=1,
        )
        organ_only = run_organ_attachment_branch(
            acquisition_dir / "acquisition.pt",
            root / "organ-only-continuous",
            "organ_only",
            device="cpu",
            adaptation_updates=2,
            eval_every=1,
            eval_batches=1,
            final_eval_batches=1,
            max_steps=2,
            a_detached_updates=0,
            recovery_updates=0,
        )
        run_organ_attachment_branch(
            acquisition_dir / "acquisition.pt",
            root / "organ-only-resumed",
            "organ_only",
            device="cpu",
            adaptation_updates=1,
            eval_every=1,
            eval_batches=1,
            final_eval_batches=1,
            max_steps=2,
            a_detached_updates=0,
            recovery_updates=0,
        )
        resumed = run_organ_attachment_branch(
            acquisition_dir / "acquisition.pt",
            root / "organ-only-resumed",
            "organ_only",
            device="cpu",
            adaptation_updates=2,
            eval_every=1,
            eval_batches=1,
            final_eval_batches=1,
            max_steps=2,
            a_detached_updates=0,
            recovery_updates=0,
            resume=True,
        )
        assert full["integrity"]["a_organ_unchanged"]
        assert full["summary"]["removal_and_reattachment"] is not None
        assert organ_only["integrity"]["a_organ_unchanged"]
        assert organ_only["integrity"]["substrate_unchanged"]
        assert resumed["records"] == organ_only["records"]
        continuous_checkpoint = torch.load(
            root / "organ-only-continuous" / "adaptation.pt", weights_only=True
        )
        resumed_checkpoint = torch.load(
            root / "organ-only-resumed" / "adaptation.pt", weights_only=True
        )
        assert all(
            torch.equal(value, resumed_checkpoint["model"][name])
            for name, value in continuous_checkpoint["model"].items()
        )
        assert torch.equal(
            continuous_checkpoint["state"]["hidden"],
            resumed_checkpoint["state"]["hidden"],
        )
        assert (root / "full" / "adaptation.pt").exists()


def test_tiny_consolidated_attachment_branch() -> None:
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
        cell_adapter_rank=2,
        vocab_size=task.vocab_size,
        batch_size=2,
        updates=2,
        warmup_updates=1,
        log_every=1,
        operator_bound=4.0,
        seed=59,
        device="cpu",
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        acquisition_dir = root / "acquisition"
        run_acquisition_calibration(
            cfg,
            acquisition_dir,
            max_updates=2,
            evaluation_interval=1,
            stage_updates=1,
            eval_batches=1,
            mastery_accuracy=2.0,
            mastery_checks=2,
        )
        result = run_consolidated_attachment_branch(
            acquisition_dir / "acquisition.pt",
            root / "consolidated",
            "consolidated",
            device="cpu",
            adaptation_updates=1,
            eval_every=1,
            eval_batches=1,
            final_eval_batches=1,
            utility_batches=1,
            max_steps=2,
            directional_edges=True,
            reserve_growth=True,
        )
        assert result["update"] == 1
        assert result["integrity"]["a_organ_unchanged"]
        assert result["profile"]["genome_plasticity"] == 0.05
        assert "utility_lesions" in result["summary"]
        assert "reserve_lesions" in result["summary"]
        assert result["growth"]["grafts"]
        assert result["protocol"]["directional_edges"]
        assert (root / "consolidated" / "adaptation.pt").exists()


def test_tiny_developmental_branch_grows_and_resumes() -> None:
    task = ProceduralTask()
    cfg = Sol2Config(
        n_input=2,
        n_memory=2,
        n_compute=4,
        n_relay=2,
        n_output=2,
        hidden=8,
        n_dendrites=5,
        initial_active_dendrites=2,
        steps_per_token=1,
        organ_queries=1,
        cell_adapter_rank=2,
        vocab_size=task.vocab_size,
        batch_size=2,
        updates=2,
        warmup_updates=1,
        log_every=1,
        operator_bound=4.0,
        seed=61,
        device="cpu",
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        acquisition_dir = root / "acquisition"
        run_acquisition_calibration(
            cfg,
            acquisition_dir,
            max_updates=2,
            evaluation_interval=1,
            stage_updates=1,
            eval_batches=1,
            mastery_accuracy=2.0,
            mastery_checks=2,
        )
        arguments = dict(
            device="cpu",
            adaptation_updates=2,
            eval_every=1,
            eval_batches=1,
            final_eval_batches=1,
            utility_batches=1,
            max_steps=2,
            anchor_rate=0.01,
            growth_cells=2,
            high_pressure=0.0,
            plateau_pressure=0.0,
            plateau_gain=1.0,
            patience_checks=1,
            growth_min_update=1,
            growth_refractory=1,
            max_growth_events=1,
            max_growth_a_drop=1.0,
        )
        result = run_developmental_attachment_branch(
            acquisition_dir / "acquisition.pt",
            root / "developmental",
            "developmental",
            **arguments,
        )
        resumed = run_developmental_attachment_branch(
            acquisition_dir / "acquisition.pt",
            root / "developmental",
            "developmental",
            resume=True,
            **arguments,
        )
        assert result["update"] == 2
        assert len(result["growth_events"]) == 1
        assert len(result["growth_events"][0]["cells"]) == 2
        assert result["config"]["n_relay"] == cfg.n_relay + 2
        assert result["integrity"]["a_organ_unchanged"]
        assert result["summary"]["growth_lesions"] is not None
        assert resumed["records"] == result["records"]
        assert resumed["growth_events"] == result["growth_events"]


def test_consolidated_analysis_applies_frozen_gates() -> None:
    def payload(a_accuracy: float, b_accuracy: float) -> dict:
        lesion_rows = {
            "normal": {"answer_accuracy": b_accuracy},
            "freeze_high_utility": {"answer_accuracy": b_accuracy - 0.30},
            "freeze_low_utility": {"answer_accuracy": b_accuracy - 0.10},
        }
        return {
            "protocol": {"max_steps": 4},
            "rejected_updates": 0,
            "integrity": {"a_organ_unchanged": True},
            "summary": {
                "final": {
                    "a_fixed": {"answer_accuracy": a_accuracy},
                    "b_by_length": {"4": {"answer_accuracy": b_accuracy}},
                    "drift": {
                        "expression_by_measured_utility": {
                            "low_mean": 2.0,
                            "high_mean": 1.0,
                        },
                        "edge_by_measured_utility": {
                            "low_mean": 3.0,
                            "high_mean": 1.0,
                        },
                    },
                },
                "utility_lesions": {"A": lesion_rows, "B": lesion_rows},
            },
        }

    with TemporaryDirectory() as directory:
        root = Path(directory)
        for branch, values in {
            "plastic": (0.20, 0.90),
            "consolidated": (0.85, 0.90),
            "shuffled": (0.40, 0.90),
        }.items():
            branch_dir = root / branch
            branch_dir.mkdir()
            (branch_dir / "metrics.json").write_text(
                json.dumps(payload(*values))
            )
        result = analyze_consolidation(root)
        assert result["primary_success"]
        assert result["gates"]["b_reuses_high_utility_more_than_low"]


def test_private_transition_analysis_keeps_capability_and_allocation_separate() -> None:
    def payload(a: float, b: float, adapter_ratio: float = 2.0) -> dict:
        normal = {"answer_accuracy": b}
        return {
            "protocol": {"max_steps": 4},
            "rejected_updates": 0,
            "integrity": {"a_organ_unchanged": True},
            "growth": {"within_limit": True, "a_accuracy_drop": 0.01},
            "summary": {
                "final": {
                    "a_fixed": {"answer_accuracy": a},
                    "b_by_length": {"4": {"answer_accuracy": b}},
                    "drift": {
                        "adapter_by_measured_utility": {
                            "low_mean": adapter_ratio,
                            "high_mean": 1.0,
                        }
                    },
                },
                "reserve_lesions": {
                    "A": {"normal": {"answer_accuracy": a}},
                    "B": {
                        "normal": normal,
                        "zero_low_utility_adapters": {"answer_accuracy": b - 0.1},
                        "zero_all_internal_adapters": {"answer_accuracy": b - 0.2},
                        "disable_graft_edges": {"answer_accuracy": b - 0.1},
                    },
                },
            },
        }

    with TemporaryDirectory() as directory:
        root = Path(directory)
        for branch, values in {
            "plastic": (0.20, 0.95),
            "uniform": (0.70, 0.90),
            "consolidated": (0.85, 0.90),
            "shuffled": (0.74, 0.90),
        }.items():
            branch_dir = root / branch
            branch_dir.mkdir()
            (branch_dir / "metrics.json").write_text(json.dumps(payload(*values)))
        result = analyze_private_transition(root)
        assert result["capability_success"]
        assert result["measured_allocation_success"]
        assert result["gates"]["reserve_adapters_are_causal_for_b"]


def test_genome_rate_screen_uses_frozen_tie_break() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        candidates = {}
        for rate, a, b in (
            (0.05, 0.80, 0.60),
            (0.15, 0.61, 0.90),
            (0.30, 0.59, 0.95),
        ):
            path = root / f"{rate}.json"
            path.write_text(
                json.dumps(
                    {
                        "protocol": {"max_steps": 4},
                        "summary": {
                            "final": {
                                "a_fixed": {"answer_accuracy": a},
                                "b_by_length": {"4": {"answer_accuracy": b}},
                            }
                        },
                    }
                )
            )
            candidates[rate] = path
        result = select_genome_rate(candidates)
        assert result["selected_rate"] == 0.30


def test_developmental_analysis_separates_anchor_and_growth_gates() -> None:
    def payload(a: float, b: float, *, growth: bool = False) -> dict:
        growth_rows = None
        events = []
        if growth:
            events = [{"cells": list(range(16))}]
            growth_rows = {
                "A": {
                    "normal": {"answer_accuracy": a},
                    "freeze_grown_cells": {"answer_accuracy": a - 0.01},
                    "zero_grown_adapters": {"answer_accuracy": a},
                },
                "B": {
                    "normal": {"answer_accuracy": b},
                    "freeze_grown_cells": {"answer_accuracy": b - 0.20},
                    "zero_grown_adapters": {"answer_accuracy": b - 0.10},
                },
            }
        return {
            "protocol": {"max_steps": 4},
            "rejected_updates": 0,
            "integrity": {"a_organ_unchanged": True},
            "growth_events": events,
            "evaluations": [{"pressure": {"pressure": 0.8}}],
            "summary": {
                "final": {
                    "a_fixed": {"answer_accuracy": a},
                    "b_by_length": {"4": {"answer_accuracy": b}},
                },
                "growth_lesions": growth_rows,
            },
        }

    with TemporaryDirectory() as directory:
        root = Path(directory)
        for branch, values in {
            "plastic": (0.20, 0.90, False),
            "uniform_anchor": (0.65, 0.80, False),
            "measured_anchor": (0.80, 0.80, False),
            "developmental": (0.90, 0.90, True),
        }.items():
            branch_dir = root / branch
            branch_dir.mkdir()
            (branch_dir / "metrics.json").write_text(
                json.dumps(payload(*values[:2], growth=values[2]))
            )
        result = analyze_development(root)
        assert result["capability_success"]
        assert result["anchor_attribution_success"]
        assert result["development_success"]


def main() -> None:
    tests = (
        test_regime_factors_are_separable,
        test_latent_execution_and_surface_encoding,
        test_reverse_order_changes_the_procedure_without_changing_inputs,
        test_tiny_branching_smoke,
        test_acquisition_checkpoint_and_resume,
        test_distinct_organ_attachment_is_behaviorally_silent_for_a,
        test_causal_utility_and_realized_update_protection,
        test_proximal_anchor_is_exact_and_growth_rows_remain_free,
        test_development_controller_requires_persistent_pressure_and_refractory,
        test_checkpoint_tensor_invariants_compare_values_not_storage,
        test_tiny_true_organ_branch_integrity,
        test_tiny_consolidated_attachment_branch,
        test_tiny_developmental_branch_grows_and_resumes,
        test_consolidated_analysis_applies_frozen_gates,
        test_private_transition_analysis_keeps_capability_and_allocation_separate,
        test_genome_rate_screen_uses_frozen_tie_break,
        test_developmental_analysis_separates_anchor_and_growth_gates,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
