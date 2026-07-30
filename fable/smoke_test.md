# smoke_test.py

## Purpose

Contract gate. Every claim the architecture depends on, checked in ~2 minutes
on CPU, before any GPU time. Runs sol's falsification #1 (a tiny repeated
corpus must be overfittable) plus fable-specific contracts.

## Components

Six test groups: shapes/reachability/sinks, mirror write-only semantics,
gradient flow to every parameter family, tiny-corpus overfit, the full growth
contract (param delta, wiring survival, graft targets, reachability, state
migration, optimizer moment surgery), and GRU matching/reproducibility.

## Decisions

- Failures exit nonzero at the first broken contract — this is a gate, not a
  report.
- Tiny geometry (24 cells) keeps the whole gate under a couple of minutes so
  it is actually run.

## Contracts

- Green smoke test is a precondition for launching run_f0.sh / run_f1.sh.
