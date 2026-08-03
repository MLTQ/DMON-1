"""CPU contract and learning gates that must pass before SOL2 uses a GPU."""

from __future__ import annotations

import dataclasses
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
from .checkpoint import load_checkpoint, save_checkpoint, unpack_state
from .config import Sol2Config
from .growth import grow_relay_tissue
from .model import Sol2, count_parameters
from .optim import build_optimizer
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
    assert health.message_absmax <= SMALL.value_gain + SMALL.identity_bias + 1e-5
    assert model.graph.reachable(model.input_idx, model.output_idx)
    assert model.graph.output_cells_are_sinks(model.output_idx)
    assert bool((model.graph.active.sum(dim=1) == SMALL.initial_active_dendrites).all())


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
    assert float(model.cell_gain.grad[model.memory_idx].abs().sum()) == 0.0


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
        2 * SMALL.n_cells * SMALL.hidden
    )


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
        "readout.weight",
    )
    for name in expected:
        gradient = parameters[name].grad
        assert gradient is not None and float(gradient.abs().sum()) > 0, name


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
    assert torch.equal(model.cell_gain[:old_n], gain_before)
    assert {graft["source"] for graft in grafts} == set(added)
    assert model.graph.reachable(model.input_idx, torch.tensor(added))
    assert model.graph.output_cells_are_sinks(model.output_idx)

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


def test_matched_gru_budget() -> None:
    target = count_parameters(Sol2(SMALL))
    hidden = match_gru_hidden(SMALL.vocab_size, target)
    matched = gru_param_count(SMALL.vocab_size, hidden)
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


def main() -> None:
    tests = [
        test_shapes_bounds_and_topology,
        test_memory_is_stream_written_only,
        test_bounded_treatment_is_parameter_neutral_and_effective,
        test_zero_identity_is_behaviorally_silent,
        test_gradient_reaches_every_adaptive_layer,
        test_tiny_repeated_stream_can_be_learned,
        test_growth_preserves_anatomy_state_and_optimizer,
        test_matched_gru_budget,
        test_matched_transformer_budget,
        test_checkpoint_restores_process,
        test_background_learning_does_not_reset_ticks,
    ]
    print("SOL2 CPU contract gate")
    for test in tests:
        print(f"- {test.__name__}", flush=True)
        test()
    print("all SOL2 contracts hold")


if __name__ == "__main__":
    main()
