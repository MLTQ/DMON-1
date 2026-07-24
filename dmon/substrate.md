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

### `Substrate._transport`
- **Does**: conservative pairwise flux exchange between adjacent body cells. Pair
  conductance is `min(c_i, c_j)`; `sum(de)` is identically zero by construction.
- **Rationale**: replaces a conductance-modulated laplacian that let the rule mint
  energy. `min` rather than an average because a channel is as narrow as its narrowest
  point, and because it lets either cell unilaterally refuse to share — the primitive
  M1's kin discrimination will need.
- **Note**: the `/4` is explicit-scheme stability across four neighbours. It also
  changes what `e_transport` means numerically, so **any tuning of `e_transport` from
  before the fix is void.**

### `make_sources`
- **Does**: Source geometries (`center`, `west`, `poles`, `corners`, `ring`), scaled by
  default to equal *total* emission.
- **Rationale**: These are the independent variable of the M0 experiment, so everything
  else has to be held constant — including how much food exists. Unnormalised the
  geometries differ by 248x in supply (`west` has one source cell, `ring` has 248), so a
  contingency test across them would be measuring abundance rather than arrangement and
  would report a confident PASS for the wrong reason. `source_rate` therefore means
  food entering the world per step, not food per source cell.

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

- **Energy conservation.** The original note here said energy "leaks slightly at the
  growing edge" and called it "tolerable for M0". That assessment was wrong in both
  direction and magnitude: the operator *created* energy, by a factor of five, and it
  was the dominant dynamic rather than a rounding error. A rule trained under it grew a
  stable body on a grid containing no food. See Decisions above and `HANDOFF.md`.
  Now fixed by `_transport`, and covered by `test_conservation.py`.
  Two deliberate non-conservations remain, both losses and both design choices rather
  than bugs: a cell caps at `e_max`, and energy in cells that fall out of the body mask
  is destroyed rather than released to neighbours. Real decomposition returns nutrients;
  this does not. Revisit at M1, where death becomes common enough to matter.
- **Light-cone constraint**: perception radius is 1, so `steps ≥ grid` for information
  to cross the field. `SubstrateConfig.light_cone_ok` checks this. Violating it is the
  single most likely cause of a mysteriously featureless result.
- `e_max = 0.2` and `seed_energy = 0.2` means the seed starts full — the yolk is just
  a full tank, not a special case.
- **The cost scale was recalibrated ~20x down.** The first draft's ecology could support
  a body of about *four cells* (`feasibility.py` computes this in a second; nobody ran
  it, and a 40-minute sweep was spent inside that regime). Cost came down rather than
  the field's magnitude going up, because `r` is fed straight into the rule's perception
  alongside O(1) inputs and a field ranging to 100 would wreck the conditioning.
  **Every metabolic constant tuned before this is void.**
- **Uptake is capped by remaining capacity** (`e_max - e`). Without it a cell near a
  source strips 1.4/step from the field, keeps 0.2, and annihilates the rest — which
  destroys energy and hides how scarce the world is. It also supplies the incentive the
  whole design wants: a full cell has to move energy onward before it can eat again, so
  harvesting *requires* transport rather than merely permitting it.
- **`spread` and the curriculum**: `SubstrateConfig.spread_at()` walks sources outward
  from the seed during training. `spread_end` defaults to 0.35 because that is the
  furthest position `feasibility.py` certifies as survivable, *not* because 1.0 was
  aesthetically rejected. Raise it only after confirming a trained rule can bridge the
  gap.
