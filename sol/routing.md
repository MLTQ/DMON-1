# `routing.py`

## Purpose

Runs bounded reverse-credit routing interventions inside one continuously adapting SOL
organism. It treats branch alignment as a proposal and block-randomized live crossover
traffic as the causal commit/reject signal.

## Components

### `RoutingTrafficConfig`
- **Does**: Defines cadence, warmup, block-randomized trial length, exact-test alpha,
  decision margin, bounded preference step, minimum proposal evidence, and an optional
  protected reporting boundary interval.
- **Rationale**: Experimental routing must be explicit, default-disabled, and
  checkpoint-reproducible.

### `RoutingTrafficTrial`
- **Does**: Stores one target-owned zero-sum preference perturbation, randomized arm
  schedule, every reward, provisional/exact null rank, counters, and decision ledger.
- **Does**: Proposes from live branch routing eligibility, exposes the delta only in
  candidate windows, and commits only a positive preference change that clears the
  configured exact randomization threshold.
- **Interacts with**: `ContinuousTrainer` in `train.py`, routing preference state in
  `model.py`, and structural probation in `structure.py`.
- **Rationale**: Rejection retains all ordinary lived body adaptation while leaving the
  incumbent routing policy unchanged.

### `routing_traffic_update`
- **Does**: Produces one compact trainer telemetry snapshot after a transition.

### `routing_traffic_due`
- **Does**: Admits a routing trial only on its own cadence, outside structural decision
  phases, and only when it can resolve before a protected validation/checkpoint
  boundary.
- **Rationale**: Structural confirmation and mutation phases must remain live; a
  reporting boundary should describe only fully resolved routing state.

### `routing_traffic_summary`
- **Does**: Exposes policy, active phase, aggregates, counts, and full decision history
  to benchmark artifacts.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train.py` | Candidate delta is fixed and every crossover block remains balanced | Phase or observation semantics |
| `model.py` | Delta has shape `(cells, dendrites)` and affects reverse-credit routing only | Tensor ownership or normalization |
| Checkpoint resume | State dict restores the exact next arm and proposal tensor | Field names or schedule |
| Experiment reports | Ledger identifies proposal, rewards, decision, and update bounds | Event schema |

## Notes

- Only targets with at least two active dendrites can receive a zero-sum proposal.
- A routing start shares the structural organ's non-decision evidence phase. Its short
  trial resolves before the next structural decision, so confirmations, topology
  changes, and ordinary learning retain their original cadence.
- Preference commits are centered and smoothly bounded by the model's existing routing
  limit. They do not add trainable parameters.
- S22 checkpoints with an already-active fixed `ABBA` trial finish under their
  historical decision rule; every newly started trial uses randomized `ABBA`/`BAAB`
  blocks.
