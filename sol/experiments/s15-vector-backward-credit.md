# S15: Decoder-shaped vector backward credit

## Question

Does a channel-shaped correction from the output organ provide more reliable
cross-window credit than broadcasting one reward scalar or transporting that scalar
backward through the connectome?

## Mechanism

After each observed next-character target, compute the detached corrective decoder
signal

`W_readout^T(one_hot(target) - softmax(logits))`.

Split that vector across output cells, smoothly bound it, and retain it in the live
field state. On later ticks:

- each target transports its credit to named source cells through the signed coefficient
  of the same axon used in the forward pass;
- transport applies the transpose of the learned `message_value` transform, preserving
  hidden-state channel direction;
- credit decays but persists across characters and truncated-BPTT boundaries;
- the shared transition rule receives credit only where it aligns channel-by-channel
  with a cell's event eligibility.

The mechanism introduces no trainable parameters. Existing direct scalar reward,
scalar reverse reward, exact within-window BPTT, and one-shot fast-synapse reward remain
available as explicit controls. Older checkpoints initialize the new state to zero and
the new gain defaults to zero.

## Protected contracts

Tests require:

- the seed to equal the decoder-shaped corrective projection and occupy only output
  cells;
- exact signed sourceward transport through the transpose of `message_value`;
- channel-specific interaction with eligibility;
- no target-free tick to create credit from zero;
- equal parameter count with the disabled control;
- exact checkpoint resume, safe old-checkpoint upgrade, aligned cell-state shuffle, and
  independent training/evaluation/local-bridge telemetry.

## Reduced-scale protocol

Use the healthy RTX 4090 while the RTX 2070S is under investigation. Run Tiny
Shakespeare with 16 cells, 16 channels, four dendrites, two message steps, four lanes,
16-character windows, 800 updates, energy held at one, and fast efficacy disabled.

Compare seeds 7, 13, and 21:

1. direct scalar reward `0.25`;
2. direct scalar reward `0.25` plus scalar reverse credit `0.25`;
3. direct scalar reward `0.25` plus decoder-shaped credit, calibrated from gains
   `0.1`, `0.25`, and `0.5` on seed 7, extending upward if the boundary still wins;
4. decoder-shaped credit alone at the selected gain.

Use identical topology seed, stream order, optimizer, validation anchors, and update
schedule within each seed. Report best/final BPC, all-aligned and post-boundary paired
deltas, stability, state ablations, and live credit magnitude.

## Gate

Proceed to credit-guided axon/dendrite turnover if the mechanism:

1. remains bounded, resume-safe, and causally active outside the output cells;
2. matches or improves the direct scalar control across seeds without relying on one
   favorable endpoint;
3. is at least as consistent as scalar reverse credit in aligned validation;
4. retains strong reset and shuffled-state penalties.

If no gain passes, use the diagnostics to revise the seed/transport normalization before
allowing structural growth to consume this signal.

## Implementation result

The mechanism adds a checkpointed `(batch, cells, channels)` field but no learned
parameters. The complete local suite passes 64 tests. It verifies the exact decoder
seed, signed transpose transport, channel-specific eligibility interaction,
target-free silence, old-checkpoint upgrade, exact resume, state-shuffle alignment, and
independent training/evaluation/local-bridge telemetry.

The mean vector-credit magnitude remained bounded near `0.01`–`0.02` in reduced-scale
training. Reset-each-token evaluation correctly reports zero credit because every token
starts from a fresh field state.

## Gain calibration

Seed 7 used direct scalar reward `0.25` plus decoder-shaped credit. Because the initial
best lay at the upper boundary, calibration extended through gains `1.0` and `2.0`.

| Vector gain | Best BPC | Final BPC |
|---:|---:|---:|
| 0.10 | 3.804 | 3.856 |
| 0.25 | 3.802 | 3.847 |
| 0.50 | 3.800 | 3.805 |
| 1.00 | 3.788 | 3.853 |
| 2.00 | 3.826 | 3.888 |

Gain `0.5` supplied the best endpoint and good stability. Higher gains transiently
improved the best checkpoint but regressed later, so `0.5` was selected before
replication.

## Three-seed result

The direct control and S8 scalar controls were run under the same 16-by-16 protocol.
A fresh seed-7 direct run reproduced its archived final BPC, confirming that the
disabled vector path preserves the historical control.

### Direct scalar plus reverse credit

| Seed | Direct final | Direct + scalar final | Direct + vector final |
|---:|---:|---:|---:|
| 7 | 3.90976 | 3.76390 | 3.80468 |
| 13 | 3.81898 | 3.77596 | 3.80925 |
| 21 | 3.72659 | 3.78690 | 3.73348 |

Relative to direct reward:

| Measure | Direct + scalar | Direct + vector |
|---|---:|---:|
| Mean best delta | −0.00673 | −0.00001 |
| Mean final delta | −0.04286 | −0.03597 |
| Mean delta across 12 aligned evaluations | −0.00520 | **−0.00958** |
| Aligned wins | 7 / 12 | 7 / 12 |
| Worst seed final regression | +0.06032 | **+0.00689** |

Negative deltas are improvements. Scalar reverse credit has the better mean endpoint,
driven by seed 7, while decoder-shaped credit has the better mean aligned trajectory
and reduces the adverse seed by almost an order of magnitude.

### Reverse credit without direct broadcast

| Seed | Direct final | Scalar-only final | Vector-only final |
|---:|---:|---:|---:|
| 7 | 3.90976 | 3.89644 | 3.86315 |
| 13 | 3.81898 | 3.78568 | 3.78496 |
| 21 | 3.72659 | 3.74740 | 3.72294 |

Vector-only improved the final endpoint for all three seeds, with a mean improvement of
`0.02810` BPC. Scalar-only improved two seeds and regressed one, for a mean improvement
of `0.00860`. Vector-only learned more slowly on seed 13, so its mean across all 12
evaluations was `0.01386` BPC worse than direct even though every final endpoint won.

All vector-only final checkpoints remained causally dependent on the recurrent body.
Reset penalties ranged from `0.952` to `1.480` BPC and shuffled-cell penalties from
`0.539` to `0.991` BPC.

Adding both scalar reverse gain `0.25` and vector gain `0.5` to direct reward reached
`3.802` final BPC on seed 7. This did not beat the seed-7 scalar combination and was not
replicated.

## Decision

The decoder-shaped mechanism passes the reduced-scale gate. It carries delayed credit
without global scalar broadcast, improves every vector-only final endpoint, improves
the combined aligned trajectory more than scalar reverse credit, remains
parameter-neutral and bounded, and preserves strong state dependence.

Keep gain `0.5` as the first morphology-development setting. Do not replace exact BPTT
or delete scalar controls. Proceed to adaptive communication using traffic,
event-eligibility, and decoder-shaped reverse-credit alignment as distinct anatomical
evidence, with live exploratory traffic required before a candidate connection is
installed.
