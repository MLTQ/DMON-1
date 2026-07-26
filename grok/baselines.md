# baselines.py

## Purpose
Honest null models for S0.

## Components

### `MatchedGRU`
- **Does**: Embedding + GRU + head with `step` / `initial_state` interface

### `match_hidden_for_budget`
- **Does**: Search hidden size so GRU params ≈ creature params

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train.py --baseline` | same online `step` protocol | Interface drift |
