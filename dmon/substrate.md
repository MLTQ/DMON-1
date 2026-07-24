# substrate.py

## Purpose

The M0 physics: NCA cells coupled to a diffusing resource field, with a metabolic
ledger deciding who lives. Contains no target image and no morphological objective.
The bet is that diffusion-limited uptake makes surface area valuable, and that
intra-body energy transport makes distance from a source expensive, and that those
two pressures alone are enough to produce shape.

## Components

### `SubstrateConfig`
- **Does**: Every physical constant of the ecology. No parameter here describes the
  creature; they all describe the world it has to survive.
- **Rationale**: Separated from the module so a contingency sweep varies *world*
  parameters while holding architecture fixed.

### `Substrate`
- **Does**: One coupled step of field diffusion, perception, rule application, and
  metabolic accounting.
- **Interacts with**: `train_m0.py` → `Pool`, `train`, `evaluate`.

### `Substrate.step`
- **Does**: Field moves first (indifferently), then cells perceive self + field, then
  the rule fires stochastically within the body mask, then the ledger settles.
- **Rationale**: Field-first ordering means the environment is never a function of
  the creature's intentions within a step — only of its past consumption.

### `Substrate.rollout`
- **Does**: Runs N steps, returns mean soft-alive mass over the final `tail` steps.
- **Rationale**: Averaging over a tail rather than reading mass at T stops the rule
  from learning a terminal spike that isn't survival.

### `make_sources`
- **Does**: Source geometries (`center`, `west`, `poles`, `corners`, `ring`).
- **Rationale**: These are the independent variable of the M0 experiment.

### `descriptors_per_sample`
- **Does**: mass, compactness (perimeter/√area), radius of gyration, box-counting
  dimension — one value **per batch element**, as `(B,)` tensors.
- **Rationale**: the probe needs within-run spread and the contingency test needs a
  noise floor; both are destroyed by averaging over the batch before they are seen.

### `descriptors`
- **Does**: batch means of the above, as plain floats. Unchanged interface.
- **Rationale**: Cheap and interpretable. The M0 verdict is read off these, not off
  renders. If they don't separate across geometries, M0 failed.

## Decisions

- **The rule may read energy and may not write it.** Channel 0 is masked out of the
  update. Without this the creature prints its own currency and the economy is
  decorative.
- **The rule gets exactly two levers, both metabolic**: uptake effort (ch1) and
  transport conductance (ch2). It has *no* direct control over shape. Morphology is
  a consequence of differential investment — cells that don't fund themselves starve
  out. This is the core bet of M0 and the thing most likely to be wrong.
- **Gate biases initialised to (+2, −2)** — uptake open, transport near-closed — so a
  fresh seed feeds itself before bleeding energy into its halo. The first version had
  ungated transport and every seed diluted itself to death in ~40 steps.
- **Rejected free-alpha aliveness** (Growing NCA's approach). Alpha is metabolically
  earned here; that substitution is the entire point of the file.
- **Tried hidden channels with no path to the ledger** — gradient was exactly zero.
  The rule could compute anything and it never touched survival. That failure is what
  produced the two-lever design.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train_m0.py` | `seed()` returns `(x, r)`; `rollout()` returns `(x, r, mass)` | Return arity, channel layout |
| `train_m0.py` | `descriptors()` keys: mass, compactness, gyration, box_dim | Key names |
| `probe.py` | `descriptors_per_sample()` returns `(B,)` tensors, same keys | Return shape |
| `render.py` | channel 2 is transport, `gate_bias[1]` is its bias | Channel assignment |
| future M1 | channel 0 = energy, 1 = effort, 2 = transport | Channel assignment |

## Notes

- **Energy is not exactly conserved.** The transport laplacian is masked by `body`,
  which breaks conservation at the boundary — energy leaks slightly at the growing
  edge. Tolerable for M0; must be fixed before any claim about metabolic efficiency.
  A conservative flux-exchange formulation is the correct replacement.
- **Light-cone constraint**: perception radius is 1, so `steps ≥ grid` for information
  to cross the field. `SubstrateConfig.light_cone_ok` checks this. Violating it is the
  single most likely cause of a mysteriously featureless result.
- `e_max = 2.0` and `seed_energy = 2.0` means the seed starts full — the yolk is just
  a full tank, not a special case.
