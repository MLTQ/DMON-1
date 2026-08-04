# `procedural_benchmark.py`

## Purpose

Measures whether a continuously trained model retains a reusable transformation
procedure when its sensors and effectors change. Text compression is absent; only
answers produced by latent program composition receive loss.

## Components

### `train_phase`

- **Does**: Trains answer-token predictions in one regime while preserving recurrent
  state across episodes and recording every update's answer bits and exact accuracy.
- **Interacts with**: `guarded_step` in `optim.py` and `ProceduralTask` in
  `procedural_task.py`.
- **Rationale**: Acquisition expands program length from one transformation to the
  configured maximum. The transfer forks start at full length, so curriculum forms the
  composition circuit without making remapping artificially easy.

### `evaluate_regime`

- **Does**: Scores generated programs without weight updates from a cloned live state.
- **Rationale**: Evaluation must preserve learned mode state without mutating the
  organism's continuing lifetime.

### `_length_curve`

- **Does**: Reports exact answer metrics separately for every trained program length.
- **Rationale**: Aggregate random-length accuracy can hide interface adaptation that
  works on primitives but has not yet recovered multi-step composition.

### `evaluate_with_ablations`

- **Does**: Applies reset, internal freeze, within-tissue state shuffle, memory zero,
  private-expression shuffle, and topology rewire interventions on procedural answers.
- **Rationale**: High accuracy alone does not establish that differentiated substrate
  structure carries the procedure.

### `run_benchmark`

- **Does**: Acquires regime A, snapshots the exact model/optimizer/live state, then
  forks matched continuation, new-interface, and new-procedure branches. SOL2 also
  receives organ-only interface and procedure forks with its internal substrate frozen.
- **Rationale**: Forking removes training-age and initialization confounds. Successful
  interface adaptation through only sensory and output organs directly tests whether a
  fixed procedural core can be reconnected; the matched reversed-procedure fork tests
  whether the frozen core can be bypassed by those organs.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment manifests | Exact A snapshot shared by every branch | Independent branch initialization |
| Offline analysis | `metrics.json` retains raw per-update records | Removing or renaming record fields |
| SOL2 interventions | All ablations see identical generated batches | Changing generator seeds between ablations |
| Full/organ-only forks | Matched interface or procedure pairs see identical training programs | Per-branch sample seeds |

## Decisions

- BPC over unscored scaffolding tokens is never computed.
- The length curriculum follows F6/F9's localized bootstrap lesson: sparse or complex
  rewards cannot train a circuit that has never received a dense solvable gradient.
- The transformer is omitted initially because the benchmark concerns persistent online
  adaptation; a stateless chunk transformer would answer a different question.
- GRU remains a descriptive recurrent control, not an arbiter of organism success.
- Organs-only SOL2 forks preserve optimizer moments, but frozen core parameters receive
  no gradient or weight decay; trainable parameters are embedding, sensory affine
  identity, and the attentive output organ.
- The first harness is intentionally short and not checkpoint-resumable. GPU promotion
  requires a manifest and service-level restart boundary after the CPU gate establishes
  that the task produces learnable, nontrivial curves.
