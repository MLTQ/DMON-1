# S0 initial matched-budget result

Date: 2026-07-26

## Question

Can the fixed-topology SOL character field learn a continuous Tiny Shakespeare stream,
use its persistent distributed state causally, and match conventional models at the same
parameter and update budget?

## Protocol

- Data: Tiny Shakespeare, contiguous 90% train / 10% held-out split.
- Budget: 5,000 updates × 16 lanes × 32 characters = 2,560,000 training characters.
- Evaluation: 256-character warmup followed by 2,048 scored held-out characters.
- Optimizer: AdamW, learning rate `3e-3`.
- Hardware: SOL on RTX 4090; controls on RTX 2070 Super.
- Code:
  - SOL and GRU: `0f525f2`
  - Transformer: `05a1293`
- SOL: 64 cells, 64 channels, 8 named dendrites per cell, 8 sensory cells, 8 output
  cells, and 3 message steps per character.

## Guarded comparison

| Run | Model | Params | Ratio | Updates | Best BPC | Final BPC | Reset Δ | Shuffle Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SOL | sparse cellular field | 122,306 | 1.000 | 5,000 | 2.484 | 2.562 | +6.283 | +2.265 |
| GRU | matched recurrent control | 122,561 | 1.002 | 5,000 | 2.255 | 2.366 | — | — |
| Transformer | matched causal control | 116,673 | 0.954 | 5,000 | 2.600 | 2.618 | — | — |

The report generator rejected incomplete, non-finite, update-mismatched, or
parameter-mismatched inputs before producing this table.

## Verdict

**S0 has not passed.** SOL learns strongly and narrowly beats the matched transformer at
the final update, but the matched GRU remains better by 0.196 BPC final and 0.229 BPC at
each run's best checkpoint.

The persistent substrate is doing real work:

- Resetting state every character worsens final BPC from 2.562 to 8.845.
- Deterministically shuffling cell state worsens it to 4.827.
- Cell and edge gradients remain nonzero throughout training.
- The best checkpoint emits recognizable character-level language structure.

Example from the best checkpoint:

> Go's make men lord, lift's you servermong / Rombrake thymen stret roy his will...

This is evidence of capability, not evidence that the architecture beats the strongest
control.

## Follow-up controls still running

1. Frozen edge weights and biases: tests whether learned edge parameters contribute
   beyond fixed directed transport and the shared cell rule.
2. No metabolism: holds energy at one to isolate prediction capability from the current
   economy.
3. Lower learning rate (`1e-3`): tests whether the oscillatory held-out curve is an
   optimization artifact.

The next architecture change should be selected only after these controls complete.
Checkpoint tensors and raw JSONL histories remain in ignored `sol/runs/`.
