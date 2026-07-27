# `routing.py`

## Purpose

Runs bounded reverse-credit routing interventions inside one continuously adapting SOL
organism. It treats branch alignment as a proposal and live ABBA traffic as the causal
commit/reject signal.

## Components

### `RoutingTrafficConfig`
- **Does**: Defines cadence, warmup, balanced trial length, decision margin, bounded
  preference step, and minimum proposal evidence.
- **Rationale**: Experimental routing must be explicit, default-disabled, and
  checkpoint-reproducible.

### `RoutingTrafficTrial`
- **Does**: Stores one target-owned zero-sum preference perturbation, its current ABBA
  arm, reward aggregates, counters, and append-only decision ledger.
- **Does**: Proposes from live branch routing eligibility, exposes the delta only in
  candidate windows, and commits only a causally favorable bounded preference change.
- **Interacts with**: `ContinuousTrainer` in `train.py`, routing preference state in
  `model.py`, and structural probation in `structure.py`.
- **Rationale**: Rejection retains all ordinary lived body adaptation while leaving the
  incumbent routing policy unchanged.

### `routing_traffic_update`
- **Does**: Produces one compact trainer telemetry snapshot after a transition.

### `routing_traffic_due`
- **Does**: Gives routing every other due phase when structural traffic shares the same
  organism, while allowing every due phase when structure is disabled.
- **Rationale**: Deterministic sequencing must not starve either causal experiment.

### `routing_traffic_summary`
- **Does**: Exposes policy, active phase, aggregates, counts, and full decision history
  to benchmark artifacts.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train.py` | Candidate delta is fixed during one trial and ABBA arms remain balanced | Phase or observation semantics |
| `model.py` | Delta has shape `(cells, dendrites)` and affects reverse-credit routing only | Tensor ownership or normalization |
| Checkpoint resume | State dict restores the exact next arm and proposal tensor | Field names or schedule |
| Experiment reports | Ledger identifies proposal, rewards, decision, and update bounds | Event schema |

## Notes

- Only targets with at least two active dendrites can receive a zero-sum proposal.
- Structural evidence and ordinary learning continue during a routing trial; topology
  decisions are sequenced so the named fan cannot change mid-trial.
- Preference commits are centered and smoothly bounded by the model's existing routing
  limit. They do not add trainable parameters.
