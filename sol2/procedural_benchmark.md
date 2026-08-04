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
- **Organ contract**: An explicit organ name selects the physical sensor/effector port;
  legacy callers default to A.

### `evaluate_regime`

- **Does**: Scores generated programs without weight updates from a cloned live state.
- **Rationale**: Evaluation must preserve learned mode state without mutating the
  organism's continuing lifetime.
- **Organ contract**: Evaluation and every lesion use the named port without changing
  any attached organ registry.

### `_length_curve`

- **Does**: Reports exact answer metrics separately for every trained program length.
- **Rationale**: Aggregate random-length accuracy can hide interface adaptation that
  works on primitives but has not yet recovered multi-step composition.

### `evaluate_with_ablations`

- **Does**: Applies reset, whole-internal, compute-only and relay-only freezes,
  within-tissue state shuffle, memory zero, private-expression shuffle, and topology
  rewire interventions on procedural answers.
- **Rationale**: High accuracy alone does not establish that differentiated substrate
  structure carries the procedure. Tissue-specific freezes distinguish computation
  from routing; these are acute evaluation lesions, never robustness training targets.
- **Adapter contract**: Adapter-zero removes cell-local transition residuals from
  internal tissue while leaving its shared genome and recurrent state operational.

### `run_benchmark`

- **Does**: Snapshots the exact initial and acquired model/optimizer/live states, then
  forks matched continuation, new-interface, new-procedure, and new-interface-from-
  scratch branches. SOL2 also receives organ-only interface and procedure forks with
  its internal substrate operational but parameter-frozen.
- **Rationale**: Forking removes training-age and initialization confounds. Successful
  full-organism interface adaptation is the biological capability. Its matched scratch
  branch measures whether acquisition transfers an advantage. Organ-only adaptation is
  a diagnostic of modular reuse, not a requirement that the neurology remain immutable;
  the reversed-procedure fork checks whether organs can route around the core.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment manifests | Exact A snapshot shared by every branch | Independent branch initialization |
| Offline analysis | `metrics.json` retains raw per-update records | Removing or renaming record fields |
| SOL2 interventions | All ablations see identical generated batches | Changing generator seeds between ablations |
| Full/organ-only forks | Matched interface or procedure pairs see identical training programs | Per-branch sample seeds |
| Acquired/scratch interface forks | Same B programs, update count, and learning-rate phase | Different sample seeds or curricula |
| Multi-organ runners | Explicit A/B port selection reaches every train/evaluation step | Dropping `organ_name` during a nested evaluation |

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
- Lesion penalties establish causal localization; larger damage is not optimized or
  treated as inherently better. Recovery after organ changes is a separate plasticity
  experiment with learning enabled.
- The first harness is intentionally short and not checkpoint-resumable. GPU promotion
  requires a manifest and service-level restart boundary after the CPU gate establishes
  that the task produces learnable, nontrivial curves.
