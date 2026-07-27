# `train.py`

## Purpose

Runs continuous truncated-BPTT next-character training and samples through the same
one-character-at-a-time organism path used during training.

## Components

### `TrainMetrics`
- **Does**: Reports language loss alongside measured energy, novelty, cell credit, and
  edge credit, fast synaptic state, causal-probe traffic, candidate advantage, and
  bounded rewiring counts, including mean first-order probe fitness and persistent
  scalar and channel-shaped backward-credit magnitudes plus reward-plastic routing
  eligibility, preference, and saturation.
- **Does**: Reports routing-traffic trial phase, actual randomized crossover arm,
  target/slot, per-arm observations and rewards, advantage, exact p-value, and
  cumulative commit/reject counts.
- **Does**: Separately reports replacements, spawns, prunes, and their cumulative
  anatomical mutation counts.
- **Does**: Reports mean viability, quiescent fraction, external energy input, actual
  metabolic spending, and transport drift for every continuous training window.
- **Does**: Reports active probation, provisional starts, commits, rollbacks, and
  accumulated prequential improvement, raw advantage, and developmental baseline.
- **Does**: Reports exploratory-traffic arm exposure, per-arm observations and reward,
  within-organism advantage, exact structural randomization p-value, and rejected
  candidates.

### `ContinuousTrainer`
- **Does**: Owns the persistent field state, corpus cursor, optimizer, and credit loop.
- **Interacts with**: `SparseAxonField` in `model.py`, structural phases in
  `structure.py`, routing trials in `routing.py`, and `ContinuousCharStream` in
  `stream.py`.
- **Rationale**: An optimizer boundary detaches history but never resets experience.
- **Structural order**: Accumulates causal evidence and performs any fixed-shape
  rewiring only after backward, gradient clipping, and the optimizer step.
- **Structural fitness**: After backward, contracts every measured probe intervention
  with its target hidden state's exact global loss gradient before detaching the next
  continuous window.
- **Structural evidence**: Accumulates installed-edge traffic plus signed
  reverse-credit/event-memory alignment for installed and exploratory paths before the
  between-window anatomy decision.
- **Probation order**: Accumulates the just-completed window's signed prequential reward,
  resolves a due probation after the optimizer step, and skips a new graft on that same
  update. It passes the exact resolution update into the permanent trial ledger.
  Ordinary body learning is never paused.
- **Exploratory probation**: Before each streamed window, gates only the selected
  candidate probe according to a checkpointed crossover schedule. Both arms update the
  same persistent body and optimizer; all unselected probes remain live. Fixed ABBA is
  the historical policy; randomized ABBA/BAAB blocks require exact one-sided evidence.
  Resolution compares candidate-on and candidate-off reward before any anatomical
  mutation.
- **Developmental baseline**: Tracks signed reward outside and during probation with a
  checkpointed EMA. A graft is judged by improvement over the body's pre-graft trend,
  not by reward that ordinary learning would have produced anyway.
- **Backward credit**: Optionally replaces or supplements direct scalar cell reward with
  an output-originating wave transported against signed axons. The wave remains in
  `FieldState`; exact BPTT and one-shot scalar fast-synapse reward continue unchanged.
- **Output-error credit**: Optionally persists a detached decoder-correction vector
  across optimizer windows, transports it sourceward through the actual message
  transform, and combines it channel-by-channel with event eligibility. It adds no
  parameters and is reported independently from scalar reverse reward.
- **Reward-plastic routing**: Carries target-owned branch eligibility and preference
  across optimizer windows so later prequential reward can assign fitness to an earlier
  decoder-credit routing event without pausing ordinary BPTT.
- **Exploratory routing**: Applies a fixed zero-sum preference delta only during
  checkpointed candidate windows, randomizes balanced `ABBA`/`BAAB` blocks, observes
  every reward after ordinary optimization, and commits only a positive
  within-organism difference that clears exact randomization inference.
- **Exploration order**: A routing trial may start only on a structural non-decision
  phase and that phase still accumulates confirmations. The routing trial resolves
  before the next structural decision phase; protected evaluation/checkpoint boundaries
  start neither trial. Both organs therefore retain their cadence without overlapping
  causal decisions.

### `ContinuousTrainer.state_dict` / `from_state_dict`
- **Does**: Round-trips model weights, optimizer moments, field state, stream cursor,
  vocabulary, structural policy and buffers, hyperparameters, update count, and
  random-number state.
- **Does**: Includes the complete active graft backup and probation evidence so resume
  produces the same future commit or rollback, plus the structural assignment, next
  arm, raw rewards, exact null rank, and all completed trial records.
- **Does**: Includes routing-traffic configuration, proposal tensor, exact next
  crossover arm, raw rewards, null rank, counters, and decision ledger.
- **Interacts with**: `save_checkpoint` and `load_checkpoint` in `checkpoint.py`.
- **Rationale**: Resuming only weights would silently create a new organism and a new
  stream position.

### Frozen-parameter policy
- **Does**: Excludes explicitly named parameters from optimization and persists that
  policy in checkpoints.
- **Interacts with**: `benchmark.py --freeze-edges`.
- **Rationale**: A fixed-connectome control must survive resume without silently making
  its edges trainable.

### `generate`
- **Does**: Feeds a prompt and sampled characters through ordinary `tick` calls.
- **Rationale**: Generation cannot use a separate decoder path that bypasses the field.

### CLI
- **Does**: Trains on a supplied text file or a bundled tiny corpus and prints a sample.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| CLI users | `python -m sol.train` is self-contained | Flag names |
| Tests | `step()` returns finite metrics and preserves `state` | Metric fields |
| `checkpoint.py` | Trainer snapshots contain the complete continuous process | State keys |

## Notes

- Cross-entropy is the current continuous task reward. A bounded per-example score is
  also delivered on the next tick through persistent cell and edge eligibility.
- This is a capability probe, not yet a competitive NanoGPT implementation.
