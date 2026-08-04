"""Branch-controlled procedural transfer benchmark for SOL2 and recurrent controls."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import Sol2Config
from .interventions import (
    degree_preserving_rewire,
    shuffled_private_expression,
    zero_private_adapters,
)
from .model import Sol2, count_parameters
from .optim import build_optimizer, guarded_step, set_learning_rate
from .procedural_task import ProceduralTask, ProcedureRegime
from .state import OrganismState
from .train import build_model

LN2 = math.log(2.0)


@dataclass
class PhaseResult:
    records: list[dict]
    state: object
    accepted_updates: int
    rejected_updates: int


def _detach_state(state):
    return state.detach()


def _clone_state(state):
    if isinstance(state, OrganismState):
        return state.clone_detached()
    return state.detach().clone()


def _step_episode(
    model,
    tokens: torch.Tensor,
    state,
    *,
    organ_name: str = "A",
    frozen_idx=None,
):
    logits = None
    for position in range(tokens.shape[1]):
        if isinstance(model, Sol2):
            logits, state, _ = model.step(
                tokens[:, position],
                state,
                organ_name=organ_name,
                frozen_idx=frozen_idx,
            )
        else:
            logits, state = model.step(tokens[:, position], state)
    assert logits is not None
    return logits, state


def train_phase(
    model,
    optimizer,
    state,
    task: ProceduralTask,
    regime: ProcedureRegime,
    cfg: Sol2Config,
    *,
    updates: int,
    global_update_start: int,
    min_steps: int,
    max_steps: int,
    generator_seed: int,
    log_every: int,
    length_curriculum: bool = False,
    organ_name: str = "A",
) -> PhaseResult:
    """Train one uninterrupted regime and retain its answer-level learning curve."""

    generator = torch.Generator().manual_seed(generator_seed)
    records = []
    accepted = 0
    rejected = 0
    model.train()
    for local_update in range(1, updates + 1):
        global_update = global_update_start + local_update
        set_learning_rate(optimizer, cfg, global_update)
        current_max_steps = max_steps
        if length_curriculum and max_steps > min_steps:
            progress = (local_update - 1) / max(updates - 1, 1)
            current_max_steps = min(
                max_steps,
                min_steps + int(progress * (max_steps - min_steps + 1)),
            )
        steps = int(
            torch.randint(
                min_steps, current_max_steps + 1, (1,), generator=generator
            ).item()
        )
        batch = task.sample_batch(
            regime, cfg.batch_size, steps, generator, cfg.device
        )
        logits, next_state = _step_episode(
            model, batch.tokens, state, organ_name=organ_name
        )
        loss = F.cross_entropy(logits, batch.answer_tokens)
        next_state = _detach_state(next_state)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        health = guarded_step(model, optimizer, cfg)
        if health.accepted:
            accepted += 1
        else:
            rejected += 1
        state = next_state
        accuracy = float(logits.detach().argmax(-1).eq(batch.answer_tokens).float().mean())
        record = {
            "update": local_update,
            "global_update": global_update,
            "steps": steps,
            "curriculum_max_steps": current_max_steps,
            "answer_bits": float(loss.detach()) / LN2,
            "answer_accuracy": accuracy,
            "gradient_norm": health.total_norm,
            "accepted": health.accepted,
        }
        records.append(record)
        if local_update % log_every == 0 or local_update == updates:
            tail = records[-min(log_every, len(records)) :]
            mean_bits = sum(row["answer_bits"] for row in tail) / len(tail)
            mean_accuracy = sum(row["answer_accuracy"] for row in tail) / len(tail)
            print(
                f"[{regime.name}] u{local_update} bits={mean_bits:.3f} "
                f"acc={mean_accuracy:.3f} rejects={rejected}",
                flush=True,
            )
    return PhaseResult(records, state, accepted, rejected)


@torch.no_grad()
def evaluate_regime(
    model,
    state,
    task: ProceduralTask,
    regime: ProcedureRegime,
    cfg: Sol2Config,
    *,
    steps: int,
    batches: int,
    generator_seed: int,
    organ_name: str = "A",
    frozen_idx=None,
    reset_each_episode: bool = False,
    mutate_state=None,
) -> dict:
    """Evaluate answer accuracy on fixed generated programs without weight updates."""

    generator = torch.Generator().manual_seed(generator_seed)
    state = _clone_state(state)
    total_nll = 0.0
    correct = 0
    count = 0
    model.eval()
    for _ in range(batches):
        if reset_each_episode:
            state = model.initial_state(cfg.batch_size, cfg.device)
        batch = task.sample_batch(
            regime, cfg.batch_size, steps, generator, cfg.device
        )
        if mutate_state is not None:
            state = mutate_state(state)
        logits, state = _step_episode(
            model,
            batch.tokens,
            state,
            organ_name=organ_name,
            frozen_idx=frozen_idx,
        )
        total_nll += float(F.cross_entropy(logits, batch.answer_tokens, reduction="sum"))
        correct += int(logits.argmax(-1).eq(batch.answer_tokens).sum())
        count += cfg.batch_size
        state = _detach_state(state)
    nll = total_nll / max(count, 1)
    return {
        "answer_bits": nll / LN2,
        "answer_accuracy": correct / max(count, 1),
        "answers": count,
        "steps": steps,
    }


def evaluate_with_ablations(
    model,
    state,
    task: ProceduralTask,
    regime: ProcedureRegime,
    cfg: Sol2Config,
    *,
    steps: int,
    batches: int,
    generator_seed: int,
    organ_name: str = "A",
) -> dict:
    """Run task-specific state, tissue, identity, topology, and memory interventions."""

    common = dict(
        model=model,
        state=state,
        task=task,
        regime=regime,
        cfg=cfg,
        steps=steps,
        batches=batches,
        generator_seed=generator_seed,
        organ_name=organ_name,
    )
    normal = evaluate_regime(**common)
    result = {
        "normal": normal,
        "reset_each_episode": evaluate_regime(**common, reset_each_episode=True),
    }
    if not isinstance(model, Sol2):
        return result

    result["freeze_internal"] = evaluate_regime(
        **common, frozen_idx=model.internal_idx
    )
    result["freeze_compute"] = evaluate_regime(
        **common, frozen_idx=model.tissue_indices("compute")
    )
    result["freeze_relay"] = evaluate_regime(
        **common, frozen_idx=model.tissue_indices("relay")
    )
    generator = torch.Generator().manual_seed(1729)
    permutations = {}
    for name in ("compute", "relay"):
        cells = model.tissue_indices(name)
        order = torch.randperm(len(cells), generator=generator)
        permutations[name] = cells.detach().cpu()[order].to(cfg.device)

    def shuffle_state(current: OrganismState) -> OrganismState:
        hidden = current.hidden
        for name, order in permutations.items():
            cells = model.tissue_indices(name)
            hidden = hidden.index_copy(1, cells, hidden[:, order])
        return OrganismState(hidden, current.memory_cursor, current.weight_version)

    def zero_memory(current: OrganismState) -> OrganismState:
        zeros = current.hidden.new_zeros(
            current.hidden.shape[0], len(model.memory_idx), current.hidden.shape[2]
        )
        return OrganismState(
            current.hidden.index_copy(1, model.memory_idx, zeros),
            current.memory_cursor,
            current.weight_version,
        )

    result["shuffle_within_tissue"] = evaluate_regime(
        **common, mutate_state=shuffle_state
    )
    result["memory_zero"] = evaluate_regime(**common, mutate_state=zero_memory)
    for name, context in (
        ("identity_shuffled", shuffled_private_expression(model)),
        ("adapters_zero", zero_private_adapters(model, model.internal_idx)),
        ("topology_rewired", degree_preserving_rewire(model)),
    ):
        with context:
            result[name] = evaluate_regime(**common)

    for name, metrics in result.items():
        if name == "normal":
            continue
        metrics["delta_answer_bits"] = metrics["answer_bits"] - normal["answer_bits"]
        metrics["delta_answer_accuracy"] = (
            metrics["answer_accuracy"] - normal["answer_accuracy"]
        )
    return result


def _window_summary(records: list[dict], fraction: float, *, from_end: bool) -> dict:
    width = max(1, round(len(records) * fraction))
    rows = records[-width:] if from_end else records[:width]
    return {
        "answer_bits": sum(row["answer_bits"] for row in rows) / len(rows),
        "answer_accuracy": sum(row["answer_accuracy"] for row in rows) / len(rows),
        "updates": len(rows),
    }


def _length_curve(
    model,
    state,
    task: ProceduralTask,
    regime: ProcedureRegime,
    cfg: Sol2Config,
    *,
    min_steps: int,
    max_steps: int,
    batches: int,
    generator_seed: int,
    organ_name: str = "A",
) -> dict[str, dict]:
    return {
        str(steps): evaluate_regime(
            model,
            state,
            task,
            regime,
            cfg,
            steps=steps,
            batches=batches,
            generator_seed=generator_seed + steps,
            organ_name=organ_name,
        )
        for steps in range(min_steps, max_steps + 1)
    }


def _fresh_branch(
    kind: str,
    cfg: Sol2Config,
    model_state,
    optimizer_state,
    *,
    organs_only: bool = False,
):
    model = build_model(kind, cfg, cfg.device)
    model.load_state_dict(model_state)
    optimizer = build_optimizer(model, cfg)
    optimizer.load_state_dict(optimizer_state)
    if organs_only:
        if not isinstance(model, Sol2):
            raise ValueError("organs-only adaptation is defined only for SOL2")
        prefixes = ("embedding.", "sensory_gain", "sensory_bias", "output_organ.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
    return model, optimizer


def run_benchmark(
    kind: str,
    cfg: Sol2Config,
    out_dir: Path,
    *,
    acquisition_updates: int,
    adaptation_updates: int,
    min_steps: int,
    max_steps: int,
    eval_batches: int,
) -> dict:
    """Acquire one procedure, then fork exact interface/procedure transfer controls."""

    out_dir.mkdir(parents=True, exist_ok=True)
    task = ProceduralTask()
    cfg = cfg.scaled(vocab_size=task.vocab_size)
    regime_a = task.base_regime("A-acquisition", cfg.seed + 101)
    regime_b = task.remapped_interface(regime_a, "B-new-interface", cfg.seed + 202)
    regime_c = task.changed_procedure(regime_a, "C-new-procedure", cfg.seed + 303)

    model = build_model(kind, cfg, cfg.device)
    optimizer = build_optimizer(model, cfg)
    state = model.initial_state(cfg.batch_size, cfg.device)
    initial_model = copy.deepcopy(model.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_state = _clone_state(state)
    acquisition = train_phase(
        model,
        optimizer,
        state,
        task,
        regime_a,
        cfg,
        updates=acquisition_updates,
        global_update_start=0,
        min_steps=min_steps,
        max_steps=max_steps,
        generator_seed=cfg.seed + 1_000,
        log_every=cfg.log_every,
        length_curriculum=True,
    )
    snapshot_model = copy.deepcopy(model.state_dict())
    snapshot_optimizer = copy.deepcopy(optimizer.state_dict())
    snapshot_state = _clone_state(acquisition.state)

    branch_specs = {
        "control": (regime_a, False, 0, "acquired"),
        "interface": (regime_b, False, 1, "acquired"),
        "procedure": (regime_c, False, 2, "acquired"),
        "interface_scratch": (regime_b, False, 1, "initial"),
    }
    if kind == "creature":
        branch_specs.update(
            {
                "interface_organs_only": (regime_b, True, 1, "acquired"),
                "procedure_organs_only": (regime_c, True, 2, "acquired"),
            }
        )
    branches = {}
    branch_models = {}
    for name, (regime, organs_only, sample_offset, source) in branch_specs.items():
        source_model = snapshot_model if source == "acquired" else initial_model
        source_optimizer = (
            snapshot_optimizer if source == "acquired" else initial_optimizer
        )
        source_state = snapshot_state if source == "acquired" else initial_state
        branch_model, branch_optimizer = _fresh_branch(
            kind,
            cfg,
            source_model,
            source_optimizer,
            organs_only=organs_only,
        )
        phase = train_phase(
            branch_model,
            branch_optimizer,
            _clone_state(source_state),
            task,
            regime,
            cfg,
            updates=adaptation_updates,
            global_update_start=acquisition_updates,
            min_steps=min_steps,
            max_steps=max_steps,
            generator_seed=cfg.seed + 2_000 + sample_offset,
            log_every=cfg.log_every,
        )
        branches[name] = phase
        branch_models[name] = branch_model

    summaries = {
        "acquisition": {
            "early": _window_summary(acquisition.records, 0.10, from_end=False),
            "late": _window_summary(acquisition.records, 0.25, from_end=True),
            "by_length": _length_curve(
                model,
                acquisition.state,
                task,
                regime_a,
                cfg,
                min_steps=min_steps,
                max_steps=max_steps,
                batches=eval_batches,
                generator_seed=cfg.seed + 2_500,
            ),
        },
        "branches": {},
    }
    for name, phase in branches.items():
        regime, organs_only, _, source = branch_specs[name]
        branch_model = branch_models[name]
        summaries["branches"][name] = {
            "organs_only": organs_only,
            "source_checkpoint": source,
            "early": _window_summary(phase.records, 0.10, from_end=False),
            "late": _window_summary(phase.records, 0.25, from_end=True),
            "by_length": _length_curve(
                branch_model,
                phase.state,
                task,
                regime,
                cfg,
                min_steps=min_steps,
                max_steps=max_steps,
                batches=eval_batches,
                generator_seed=cfg.seed + 2_700,
            ),
            "trained_length": evaluate_with_ablations(
                branch_model,
                phase.state,
                task,
                regime,
                cfg,
                steps=max_steps,
                batches=eval_batches,
                generator_seed=cfg.seed + 3_000,
            ),
            "longer_length": evaluate_regime(
                branch_model,
                phase.state,
                task,
                regime,
                cfg,
                steps=2 * max_steps,
                batches=eval_batches,
                generator_seed=cfg.seed + 4_000,
            ),
        }
    if kind == "creature":
        summaries["organ_reuse"] = {
            "interface_organs_only_late_accuracy": summaries["branches"]
            ["interface_organs_only"]["late"]["answer_accuracy"],
            "interface_full_late_accuracy": summaries["branches"]["interface"]
            ["late"]["answer_accuracy"],
            "procedure_organs_only_late_accuracy": summaries["branches"]
            ["procedure_organs_only"]["late"]["answer_accuracy"],
            "procedure_full_late_accuracy": summaries["branches"]["procedure"]
            ["late"]["answer_accuracy"],
        }
    summaries["interface_transfer"] = {
        "full_minus_scratch_early_accuracy": summaries["branches"]["interface"]
        ["early"]["answer_accuracy"]
        - summaries["branches"]["interface_scratch"]["early"]["answer_accuracy"],
        "full_minus_scratch_late_accuracy": summaries["branches"]["interface"]
        ["late"]["answer_accuracy"]
        - summaries["branches"]["interface_scratch"]["late"]["answer_accuracy"],
    }
    summaries["return_to_a_after_interface"] = evaluate_regime(
        branch_models["interface"],
        branches["interface"].state,
        task,
        regime_a,
        cfg,
        steps=max_steps,
        batches=eval_batches,
        generator_seed=cfg.seed + 5_000,
    )

    payload = {
        "kind": kind,
        "parameters": count_parameters(model),
        "config": cfg.to_dict(),
        "benchmark": {
            "acquisition_updates": acquisition_updates,
            "adaptation_updates": adaptation_updates,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "eval_batches": eval_batches,
        },
        "regimes": {
            "A": regime_a.to_dict(),
            "B": regime_b.to_dict(),
            "C": regime_c.to_dict(),
        },
        "summary": summaries,
        "records": {
            "acquisition": acquisition.records,
            **{name: phase.records for name, phase in branches.items()},
        },
        "rejected_updates": {
            "acquisition": acquisition.rejected_updates,
            **{name: phase.rejected_updates for name, phase in branches.items()},
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="SOL2 procedural transfer benchmark")
    parser.add_argument("--model", choices=("creature", "gru"), default="creature")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--acquisition-updates", type=int, default=1_000)
    parser.add_argument("--adaptation-updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--operator-bound", type=float, default=4.0)
    args = parser.parse_args()
    cfg = Sol2Config(
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        bounded_operators=True,
        operator_bound=args.operator_bound,
        cell_identity=True,
        updates=args.acquisition_updates + args.adaptation_updates,
    )
    result = run_benchmark(
        args.model,
        cfg,
        args.out_dir,
        acquisition_updates=args.acquisition_updates,
        adaptation_updates=args.adaptation_updates,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        eval_batches=args.eval_batches,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
