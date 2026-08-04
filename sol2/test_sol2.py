"""CPU contract and learning gates that must pass before SOL2 uses a GPU."""

from __future__ import annotations

import dataclasses
import json
import math
import tempfile
from pathlib import Path

import torch

from .baselines import (
    MatchedTransformer,
    gru_param_count,
    match_gru_hidden,
    match_transformer_hidden,
)
from .checkpoint import load_checkpoint, restore_rng_state, save_checkpoint, unpack_state
from .campaign import (
    campaign_status,
    claim_ready_job,
    finish_job,
    run_worker,
    write_manifest,
)
from .config import Sol2Config
from .developmental_campaign import build_jobs, geometry_for_cells
from .evaluate import evaluate_with_ablations
from .growth import activate_reserve_dendrites, grow_relay_tissue
from .hardware_preflight import run_hardware_preflight
from .interventions import (
    degree_preserving_rewire,
    disable_graft_edges,
    shuffled_private_expression,
    zero_private_adapters,
)
from .model import Sol2, count_parameters
from .optim import build_optimizer, cell_gradient_utilization
from .runtime import AsyncOrganismRuntime
from .stream import LaneStream


SMALL = Sol2Config(
    n_input=2,
    n_memory=4,
    n_compute=6,
    n_output=2,
    n_relay=4,
    hidden=12,
    n_dendrites=5,
    initial_active_dendrites=3,
    steps_per_token=2,
    vocab_size=7,
    batch_size=3,
    chunk_length=6,
    updates=100,
    warmup_updates=10,
    eval_every=50,
    eval_tokens=32,
    seed=7,
)


def test_shapes_bounds_and_topology() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL)
    state = model.initial_state(3, "cpu")
    logits, state, health = model.step(
        torch.tensor([1, 2, 3]), state, collect_health=True
    )
    assert logits.shape == (3, SMALL.vocab_size)
    assert state.hidden.shape == (3, SMALL.n_cells, SMALL.hidden)
    assert health.hidden_absmax <= 1.0 + 1e-6
    assert health.message_absmax <= SMALL.value_gain + 1e-5
    assert 1.0 <= health.organ_effective_cells <= SMALL.n_output + 1e-5
    assert model.graph.reachable(model.input_idx, model.output_idx)
    assert model.graph.reachable(model.input_idx, model.internal_idx)
    assert model.graph.output_cells_are_sinks(model.output_idx)
    assert model.graph.targets_read_only_from(model.output_idx, model.internal_idx)
    assert bool((model.graph.active.sum(dim=1) == SMALL.initial_active_dendrites).all())


def test_internal_tissue_is_a_required_output_path() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL).eval()
    active_sources = model.graph.sources[model.output_idx][
        model.graph.active[model.output_idx]
    ]
    forbidden = torch.cat([model.input_idx, model.memory_idx])
    assert not bool(torch.isin(active_sources, forbidden).any())

    state = model.initial_state(2, "cpu")
    state.hidden[:, model.output_idx] = torch.randn_like(
        state.hidden[:, model.output_idx]
    )
    frozen_logits, _, _ = model.step(
        torch.tensor([1, 5]), state, frozen_idx=model.internal_idx
    )
    assert torch.allclose(frozen_logits[0], frozen_logits[1], atol=1e-6)

    cleared = state.clone_detached()
    cleared.hidden[:, model.output_idx] = 0
    logits_from_loaded, _, _ = model.step(torch.tensor([2, 2]), state)
    logits_from_clear, _, _ = model.step(torch.tensor([2, 2]), cleared)
    assert torch.allclose(logits_from_loaded, logits_from_clear, atol=1e-6)


def test_memory_is_stream_written_only() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL)
    state = model.initial_state(2, "cpu")
    tokens = torch.tensor([2, 5])
    _, state, _ = model.step(tokens, state)
    written = torch.tanh(model.embedding(tokens)).detach()
    assert torch.allclose(state.hidden[:, model.memory_idx[0]], written, atol=1e-6)
    assert bool((state.hidden[:, model.memory_idx[1:]] == 0).all())

    targets = torch.randint(0, SMALL.vocab_size, (2, 4))
    inputs = torch.randint(0, SMALL.vocab_size, (2, 4))
    loss, _, _ = model.forward_chunk(inputs, targets, model.initial_state(2, "cpu"))
    loss.backward()
    assert model.cell_gain is not None
    assert len(model.cell_gain) == SMALL.n_mutable
    assert bool((model.expression_pos[model.memory_idx] == -1).all())
    assert float(model.cell_gain.grad.abs().sum()) > 0.0


def test_bounded_treatment_is_parameter_neutral_and_effective() -> None:
    torch.manual_seed(SMALL.seed)
    bounded = Sol2(SMALL)
    torch.manual_seed(SMALL.seed)
    unbounded = Sol2(dataclasses.replace(SMALL, bounded_operators=False))
    assert count_parameters(bounded) == count_parameters(unbounded)
    for name in ("query", "key", "value"):
        left = getattr(bounded.graph, name).weight
        right = getattr(unbounded.graph, name).weight
        assert torch.equal(left, right)

    with torch.no_grad():
        bounded.graph.value.weight.mul_(20.0)
    _ = bounded.graph.aggregate(
        bounded.initial_state(2, "cpu").hidden, bounded.mutable_idx
    )
    assert bounded.graph.value.spectral_norm() <= SMALL.operator_bound * 1.02


def test_zero_identity_is_behaviorally_silent() -> None:
    anonymous_cfg = dataclasses.replace(SMALL, cell_identity=False)
    torch.manual_seed(SMALL.seed)
    anonymous = Sol2(anonymous_cfg)
    torch.manual_seed(SMALL.seed)
    identified = Sol2(SMALL)
    inputs = torch.randint(0, SMALL.vocab_size, (2, 5))
    sa = anonymous.initial_state(2, "cpu")
    si = identified.initial_state(2, "cpu")
    for position in range(inputs.shape[1]):
        la, sa, _ = anonymous.step(inputs[:, position], sa)
        li, si, _ = identified.step(inputs[:, position], si)
    assert torch.allclose(la, li, atol=1e-6)
    assert count_parameters(identified) - count_parameters(anonymous) == (
        2 * SMALL.n_mutable * SMALL.hidden
    )


def test_zero_up_private_adapters_are_silent_and_recruitable() -> None:
    adapted_cfg = dataclasses.replace(SMALL, cell_adapter_rank=4)
    torch.manual_seed(SMALL.seed)
    baseline = Sol2(SMALL)
    torch.manual_seed(SMALL.seed)
    adapted = Sol2(adapted_cfg)
    baseline_state = baseline.initial_state(2, "cpu")
    adapted_state = adapted.initial_state(2, "cpu")
    tokens = torch.randint(0, SMALL.vocab_size, (2, 5))
    for position in range(tokens.shape[1]):
        baseline_logits, baseline_state, _ = baseline.step(
            tokens[:, position], baseline_state
        )
        adapted_logits, adapted_state, _ = adapted.step(
            tokens[:, position], adapted_state
        )
    assert torch.equal(baseline_logits, adapted_logits)
    assert torch.equal(baseline_state.hidden, adapted_state.hidden)

    targets = torch.randint(0, SMALL.vocab_size, (2, 5))
    optimizer = build_optimizer(adapted, adapted_cfg)
    loss, _, _ = adapted.forward_chunk(
        tokens, targets, adapted.initial_state(2, "cpu")
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert adapted.cell_adapter_up.grad is not None
    assert float(adapted.cell_adapter_up.grad.abs().sum()) > 0.0
    before = adapted.cell_adapter_up.detach().clone()
    optimizer.step()
    assert not torch.equal(adapted.cell_adapter_up, before)

    trained = adapted.cell_adapter_up.detach().clone()
    with zero_private_adapters(adapted, adapted.internal_idx):
        positions = adapted.expression_pos[adapted.internal_idx]
        assert float(adapted.cell_adapter_up[positions].detach().abs().sum()) == 0.0
    assert torch.equal(adapted.cell_adapter_up.detach(), trained)


def test_gradient_reaches_every_adaptive_layer() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL)
    tokens = torch.randint(0, SMALL.vocab_size, (3, 6))
    targets = torch.randint(0, SMALL.vocab_size, (3, 6))
    loss, _, _ = model.forward_chunk(tokens, targets, model.initial_state(3, "cpu"))
    loss.backward()
    parameters = dict(model.named_parameters())
    expected = (
        "embedding.weight",
        "graph.edge_logit",
        "graph.query.weight",
        "graph.value.weight",
        "tissues.rules.input.target.weight",
        "tissues.rules.compute.target.weight",
        "tissues.rules.relay.target.weight",
        "tissues.rules.output.target.weight",
        "cell_gain",
        "cell_bias",
        "output_organ.queries",
        "output_organ.key.weight",
        "output_organ.value.weight",
        "output_organ.decoder.weight",
    )
    for name in expected:
        gradient = parameters[name].grad
        assert gradient is not None and float(gradient.abs().sum()) > 0, name
    utilization = cell_gradient_utilization(model)
    assert utilization["internal_nonzero_fraction"] > 0.20
    assert utilization["internal_material_fraction"] > 0.20


def test_tiny_repeated_stream_can_be_learned() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL)
    optimizer = build_optimizer(model, SMALL)
    ids = torch.tensor([0, 1, 2, 3] * 128, dtype=torch.long)
    stream = LaneStream(ids, SMALL.batch_size, SMALL.seed, "cpu")
    state = model.initial_state(SMALL.batch_size, "cpu")
    first = None
    final = None
    for _ in range(240):
        inputs, targets = stream.next_chunk(SMALL.chunk_length)
        loss, state, _ = model.forward_chunk(inputs, targets, state)
        state = state.detach()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), SMALL.grad_clip)
        optimizer.step()
        value = float(loss.detach()) / math.log(2.0)
        first = value if first is None else first
        final = value
    assert final is not None and first is not None
    assert final < 0.35, (first, final)
    model.eval()
    causal = evaluate_with_ablations(model, ids, 16, 48, "cpu")
    assert causal["freeze_internal_delta_bpc"] > 1.0
    assert causal["topology_rewire_delta_bpc"] > 0.01


def test_growth_preserves_anatomy_state_and_optimizer() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL)
    optimizer = build_optimizer(model, SMALL)
    state = model.initial_state(2, "cpu")
    tokens = torch.randint(0, SMALL.vocab_size, (2, 5))
    targets = torch.randint(0, SMALL.vocab_size, (2, 5))
    loss, state, _ = model.forward_chunk(tokens, targets, state)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    sources_before = model.graph.sources.clone()
    active_before = model.graph.active.clone()
    edge_before = model.graph.edge_logit.detach().clone()
    gain_before = model.cell_gain.detach().clone()
    hidden_before = state.hidden.detach().clone()
    params_before = count_parameters(model)
    old_n = model.n_cells
    n_new = 3

    added, migrate, grafts = grow_relay_tissue(model, optimizer, n_new)
    expected_delta = n_new * SMALL.n_dendrites + 2 * n_new * SMALL.hidden
    assert count_parameters(model) - params_before == expected_delta
    assert torch.equal(model.graph.sources[:old_n][active_before], sources_before[active_before])
    assert bool((model.graph.active[:old_n] | ~active_before).all())
    assert torch.equal(model.graph.edge_logit[:old_n][active_before], edge_before[active_before])
    assert torch.equal(model.cell_gain[: len(gain_before)], gain_before)
    assert {graft["source"] for graft in grafts} == set(added)
    assert model.graph.reachable(model.input_idx, torch.tensor(added))
    assert model.graph.output_cells_are_sinks(model.output_idx)
    assert model.graph.targets_read_only_from(model.output_idx, model.internal_idx)

    migrated = migrate(state.detach())
    assert migrated.hidden.shape[1] == old_n + n_new
    assert torch.equal(migrated.hidden[:, :old_n], hidden_before)
    assert float(migrated.hidden[:, old_n:].abs().sum()) == 0.0
    edge_state = optimizer.state[model.graph.edge_logit]
    assert float(edge_state["exp_avg"][:old_n].abs().sum()) > 0
    assert float(edge_state["exp_avg"][old_n:].abs().sum()) == 0

    rebuilt = Sol2(model.cfg)
    rebuilt.load_state_dict(model.state_dict())
    assert torch.equal(rebuilt.relay_idx, model.relay_idx)


def test_adapter_growth_and_directional_reserve_are_append_only() -> None:
    cfg = dataclasses.replace(SMALL, cell_adapter_rank=3)
    torch.manual_seed(cfg.seed)
    model = Sol2(cfg)
    optimizer = build_optimizer(model, cfg)
    old_down = model.cell_adapter_down.detach().clone()
    old_up = model.cell_adapter_up.detach().clone()
    added, _, _ = grow_relay_tissue(model, optimizer, 2)
    assert torch.equal(model.cell_adapter_down[: len(old_down)], old_down)
    assert torch.equal(model.cell_adapter_up[: len(old_up)], old_up)
    assert float(model.cell_adapter_up[len(old_up) :].detach().abs().sum()) == 0.0
    assert len(added) == 2

    sources = model.graph.sources.clone()
    active = model.graph.active.clone()
    utility = torch.linspace(0.0, 1.0, model.n_cells)
    grafts = activate_reserve_dendrites(model, utility, target_fraction=0.5)
    expected = 2 * max(1, round(len(model.internal_idx) * 0.5))
    assert len(grafts) == expected
    assert torch.equal(model.graph.sources[active], sources[active])
    assert bool((model.graph.active | ~active).all())
    assert model.graph.output_cells_are_sinks(model.output_idx)
    assert model.graph.targets_read_only_from(model.output_idx, model.internal_idx)
    grown_active = model.graph.active.clone()
    with disable_graft_edges(model, grafts):
        assert int(model.graph.active.sum()) == int(grown_active.sum()) - len(grafts)
    assert torch.equal(model.graph.active, grown_active)


def test_causal_interventions_preserve_counts_and_restore() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL)
    sources = model.graph.sources.clone()
    active_sources = sources[model.graph.active]
    degree = torch.bincount(active_sources, minlength=model.n_cells)
    with degree_preserving_rewire(model):
        rewired = model.graph.sources[model.graph.active]
        assert not torch.equal(rewired, active_sources)
        assert torch.equal(torch.bincount(rewired, minlength=model.n_cells), degree)
        assert model.graph.targets_read_only_from(model.output_idx, model.internal_idx)
    assert torch.equal(model.graph.sources, sources)

    with torch.no_grad():
        marker = torch.arange(len(model.internal_idx), dtype=torch.float32)
        positions = model.expression_pos[model.internal_idx]
        model.cell_gain[positions, 0] = marker
    gain = model.cell_gain.detach().clone()
    with shuffled_private_expression(model):
        assert not torch.equal(model.cell_gain, gain)
    assert torch.equal(model.cell_gain, gain)


def test_matched_gru_budget() -> None:
    cfg = Sol2Config(vocab_size=65, seed=SMALL.seed)
    target = count_parameters(Sol2(cfg))
    hidden = match_gru_hidden(cfg.vocab_size, target)
    matched = gru_param_count(cfg.vocab_size, hidden)
    assert abs(matched - target) / target < 0.02


def test_matched_transformer_budget() -> None:
    # The four-head transformer has a coarse width grid at toy scale, so validate
    # the actual preregistered model budget rather than the deliberately tiny kernel.
    cfg = Sol2Config(vocab_size=65, seed=SMALL.seed)
    target = count_parameters(Sol2(cfg))
    hidden = match_transformer_hidden(
        cfg.vocab_size,
        target,
        max_len=cfg.transformer_seq_len,
    )
    matched = count_parameters(
        MatchedTransformer(
            cfg.vocab_size,
            hidden,
            max_len=cfg.transformer_seq_len,
        )
    )
    assert abs(matched - target) / target < 0.02


def test_checkpoint_restores_process() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL)
    optimizer = build_optimizer(model, SMALL)
    ids = torch.arange(128) % SMALL.vocab_size
    stream = LaneStream(ids, SMALL.batch_size, SMALL.seed, "cpu")
    inputs, targets = stream.next_chunk(SMALL.chunk_length)
    state = model.initial_state(SMALL.batch_size, "cpu")
    loss, state, _ = model.forward_chunk(inputs, targets, state)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.pt"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            state=state.detach(),
            stream=stream,
            cfg=SMALL,
            kind="creature",
            update=1,
            accepted_updates=1,
            rejected_updates=0,
            history=[],
            evaluations=[],
        )
        payload = load_checkpoint(path, "cpu")
        restored_state = unpack_state(payload["state"], "cpu")
        assert torch.equal(restored_state.hidden, state.hidden.detach())
        assert restored_state.memory_cursor == state.memory_cursor

        restored_model = Sol2(SMALL)
        restored_model.load_state_dict(payload["model"])
        restored_optimizer = build_optimizer(restored_model, SMALL)
        restored_optimizer.load_state_dict(payload["optimizer"])
        stream2 = LaneStream(ids, SMALL.batch_size, SMALL.seed, "cpu")
        stream2.load_state_dict(payload["stream"])
        assert torch.equal(stream2.cursors, stream.cursors)

        inputs1, targets1 = stream.next_chunk(SMALL.chunk_length)
        inputs2, targets2 = stream2.next_chunk(SMALL.chunk_length)
        assert torch.equal(inputs1, inputs2) and torch.equal(targets1, targets2)
        loss1, next1, _ = model.forward_chunk(
            inputs1, targets1, state.detach()
        )
        loss2, next2, _ = restored_model.forward_chunk(
            inputs2, targets2, restored_state
        )
        assert torch.equal(loss1, loss2)
        assert torch.equal(next1.hidden, next2.hidden)

        optimizer.zero_grad(set_to_none=True)
        restored_optimizer.zero_grad(set_to_none=True)
        loss1.backward()
        loss2.backward()
        optimizer.step()
        restored_optimizer.step()
        for original, restored in zip(
            model.parameters(), restored_model.parameters(), strict=True
        ):
            assert torch.equal(original, restored)


def test_background_learning_does_not_reset_ticks() -> None:
    torch.manual_seed(SMALL.seed)
    model = Sol2(SMALL)
    with AsyncOrganismRuntime(
        model, SMALL, "cpu", learning_window=4, max_staleness=20
    ) as runtime:
        for token in ([0, 1, 2, 3] * 6):
            runtime.observe(token)
        cursor_before = runtime.state.memory_cursor
        runtime.wait_for_learning()
        assert runtime.stats.ticks == 24
        assert runtime.stats.learned_windows >= 1
        assert runtime.state.memory_cursor == cursor_before
        assert runtime.state.hidden.abs().sum() > 0


def test_hardware_preflight_measures_base_and_growth() -> None:
    cfg = SMALL.scaled(
        steps_per_token=1,
        batch_size=2,
        cell_adapter_rank=2,
        device="cpu",
    )
    result = run_hardware_preflight(
        cfg,
        warmup_updates=1,
        measure_updates=2,
        program_steps=2,
        growth_cells=2,
    )
    assert result["schema"] == 1
    assert result["hardware"]["type"] == "cpu"
    assert [row["name"] for row in result["phases"]] == ["base", "grown"]
    assert all(row["accepted_updates"] == 2 for row in result["phases"])
    assert all(row["rejected_updates"] == 0 for row in result["phases"])
    assert result["phases"][1]["cells"] == result["phases"][0]["cells"] + 2
    assert len(result["growth"]["cells"]) == 2
    assert result["protocol"]["base_config"]["n_relay"] == cfg.n_relay
    assert result["capacity"]["estimated_concurrent_processes_at_80pct"] is None


def test_campaign_claims_dependencies_and_requires_artifacts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        jobs = [
            {
                "id": "acquire-s7",
                "command": ["{python}", "-c", "print('acquire')"],
                "depends_on": [],
                "expected_artifact": "acquire/metrics.json",
                "completion_check": {
                    "json_field": ["mastered"],
                    "equals": True,
                },
            },
            {
                "id": "adapt-s7",
                "command": ["{python}", "-c", "print('adapt')"],
                "depends_on": ["acquire-s7"],
                "expected_artifact": "adapt/metrics.json",
            },
        ]
        first_manifest = write_manifest(root, jobs, {"name": "test"})
        assert write_manifest(root, jobs, {"name": "test"}) == first_manifest
        first = claim_ready_job(root, "worker-a", now=10.0)
        assert first is not None and first["id"] == "acquire-s7"
        assert claim_ready_job(root, "worker-b", now=11.0) is None
        failed = finish_job(root, first["id"], "worker-a", exit_code=0, now=12.0)
        assert failed["status"] == "failed"
        assert failed["failure"] == "expected_artifact_missing"
        assert claim_ready_job(root, "worker-b", now=13.0) is None

        retried = claim_ready_job(
            root, "worker-a", retry_failed=True, max_attempts=3, now=14.0
        )
        assert retried is not None and retried["id"] == "acquire-s7"
        artifact = root / "acquire" / "metrics.json"
        artifact.parent.mkdir()
        artifact.write_text(json.dumps({"mastered": False}))
        ineligible = finish_job(
            root, retried["id"], "worker-a", exit_code=0, now=15.0
        )
        assert ineligible["failure"] == "completion_check_failed"
        retried = claim_ready_job(
            root, "worker-a", retry_failed=True, max_attempts=3, now=16.0
        )
        assert retried is not None and retried["id"] == "acquire-s7"
        artifact.write_text(json.dumps({"mastered": True}))
        completed = finish_job(root, retried["id"], "worker-a", exit_code=0, now=17.0)
        assert completed["status"] == "complete"
        assert completed["elapsed_seconds"] == 1.0
        downstream = claim_ready_job(root, "worker-b", now=18.0)
        assert downstream is not None and downstream["id"] == "adapt-s7"


def test_checkpoint_rng_restore_normalizes_serialized_state() -> None:
    original = torch.get_rng_state()
    try:
        torch.manual_seed(991)
        target = torch.get_rng_state().clone()
        torch.manual_seed(992)
        assert not torch.equal(torch.get_rng_state(), target)
        restore_rng_state({"torch_rng_state": target, "cuda_rng_state": None})
        assert torch.equal(torch.get_rng_state(), target)
    finally:
        torch.set_rng_state(original)


def test_developmental_campaign_is_capacity_and_mastery_gated() -> None:
    geometry = geometry_for_cells(800)
    assert geometry["n_compute"] == 576
    assert geometry["n_relay"] == 144
    assert geometry["growth_cells_per_event"] == 32
    jobs, protocol = build_jobs(
        seeds=[7, 13],
        sizes=[400, 800],
        acquisition_updates=10_000,
        adaptation_updates=3_000,
    )
    assert len(jobs) == 2 + 2 * 2 * 5
    by_id = {job["id"]: job for job in jobs}
    acquisition = by_id["acquire-n800-s13"]
    assert acquisition["depends_on"] == ["preflight-n800"]
    assert acquisition["completion_check"]["json_field"] == ["mastered"]
    developmental = by_id["adapt-n800-s13-developmental"]
    assert developmental["depends_on"] == ["acquire-n800-s13"]
    growth_index = developmental["command"].index("--growth-cells") + 1
    assert developmental["command"][growth_index] == "32"
    assert protocol["effective_batch_size"] == 24


def test_campaign_worker_executes_dependency_chain() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_json = (
            "import json; from pathlib import Path; "
            "p=Path(r'{root}/%s'); p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text(json.dumps(%s))"
        )
        jobs = [
            {
                "id": "first",
                "command": [
                    "{python}",
                    "-c",
                    write_json % ("first.json", "{'ready': True}"),
                ],
                "depends_on": [],
                "expected_artifact": "first.json",
                "completion_check": {"json_field": ["ready"], "equals": True},
            },
            {
                "id": "second",
                "command": [
                    "{python}",
                    "-c",
                    write_json % ("second.json", "{'done': 2}"),
                ],
                "depends_on": ["first"],
                "expected_artifact": "second.json",
                "completion_check": {"json_field": ["done"], "equals": 2},
            },
        ]
        write_manifest(root, jobs, {"name": "worker-chain"})
        exit_code = run_worker(
            root,
            "cpu-worker",
            cuda_visible_devices=None,
            retry_failed=False,
            max_attempts=2,
            reclaim_after_seconds=None,
            poll_seconds=0.01,
        )
        assert exit_code == 0
        assert campaign_status(root)["counts"]["complete"] == 2


def main() -> None:
    tests = [
        test_shapes_bounds_and_topology,
        test_internal_tissue_is_a_required_output_path,
        test_memory_is_stream_written_only,
        test_bounded_treatment_is_parameter_neutral_and_effective,
        test_zero_identity_is_behaviorally_silent,
        test_zero_up_private_adapters_are_silent_and_recruitable,
        test_gradient_reaches_every_adaptive_layer,
        test_tiny_repeated_stream_can_be_learned,
        test_growth_preserves_anatomy_state_and_optimizer,
        test_adapter_growth_and_directional_reserve_are_append_only,
        test_causal_interventions_preserve_counts_and_restore,
        test_matched_gru_budget,
        test_matched_transformer_budget,
        test_checkpoint_restores_process,
        test_background_learning_does_not_reset_ticks,
        test_hardware_preflight_measures_base_and_growth,
        test_campaign_claims_dependencies_and_requires_artifacts,
        test_checkpoint_rng_restore_normalizes_serialized_state,
        test_developmental_campaign_is_capacity_and_mastery_gated,
        test_campaign_worker_executes_dependency_chain,
    ]
    print("SOL2 CPU contract gate")
    for test in tests:
        print(f"- {test.__name__}", flush=True)
        test()
    print("all SOL2 contracts hold")


if __name__ == "__main__":
    main()
