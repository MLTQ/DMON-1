# smoke_test.py

## Purpose
Fast, no-GPU gate: mirror write-only contract, state persistence, gradient reach (including attention), multistream batch, and loss drop on a synthetic regular stream.

## Components

### Contract tests
- Mirror pollution, persistence, dendrite/rule/attention grads, attention-off ablation, multistream

### `test_online_loss_drops`
- **Does**: Highly periodic text; assert NLL falls and beats chance
- **Rationale**: Catch "wiring works but nothing learns" before long Shakespeare runs

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CI / humans | `python -m grok.smoke_test` exits 0 | Weakening asserts without reason |

## Notes
- Uses synthetic text so it does not require network download.
- Query grads are zero from a pure-zero state (q = W h with h=0); tests warm a few steps first.
