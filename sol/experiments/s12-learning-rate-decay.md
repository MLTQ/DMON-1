# S12: Late optimization decay

## Motivation

Several 5,000-update SOL organisms reached their best held-out checkpoint around update
3,750 and then regressed while AdamW remained at `3e-3`. S9 seed 7 regressed by 0.10218
BPC from best to final, and its energy-one control regressed by 0.08168. Physiology and
structural mechanisms should not be credited or blamed for a preventable optimizer
failure.

## Hypothesis

Holding the established learning rate during early development and then decaying it
smoothly can preserve late capability without changing the organism, parameter count,
stream state, reward, energy, or topology.

## Mechanism

- Keep `3e-3` through an explicit absolute update.
- Cosine-decay to a configured fraction of the base rate at an explicit end update.
- Apply the same pure schedule to SOL and matched behavioral controls.
- Record the effective rate in every training row and the policy in summaries and SOL
  checkpoint metadata.
- A zero decay end is the exact historical constant-rate path.

## Preflight

Use the existing 16-cell, 800-update, three-seed Tiny Shakespeare protocol with no
metabolism and no fast efficacy. Candidate decay is update 400 through 800 to `0.1` of
the base rate. Compare every aligned evaluation, best BPC, final BPC, and post-best
regression against fresh constant-rate controls.

The candidate and control were exactly identical through update 400 on every seed.
After the boundary, decay won all 12 paired evaluations:

| Seed | Control best/final | Decay best/final | Best delta | Final delta |
|---:|---:|---:|---:|---:|
| 7 | 3.61888 / 3.75525 | 3.58847 / 3.58847 | -0.03041 | -0.16678 |
| 13 | 3.55620 / 3.64429 | 3.49612 / 3.49612 | -0.06008 | -0.14817 |
| 21 | 3.52084 / 3.59386 | 3.48016 / 3.49602 | -0.04068 | -0.09783 |

Aggregate candidate-minus-control deltas:

- all 24 evaluations, including the 12 exactly tied pre-boundary points: `-0.02752`;
- 12 post-boundary evaluations: `-0.05505`, 12/12 wins;
- mean best checkpoint: `-0.04372`;
- mean final checkpoint: `-0.13759`.

Every decayed run finished at its best checkpoint. The preflight therefore passes: the
schedule removes a reproducible late optimizer failure rather than selecting one lucky
seed.

## Full-scale gate

Only run a 64-cell candidate if the preflight improves mean final and paired-evaluation
BPC without a seed-level collapse. The full candidate must beat the current live
checkpoint (`2.43925` BPC), remain stable at its final update, and preserve the usual
persistent-state, shuffle, topology, and physiology gates before promotion.

The selected full-scale candidate uses the current language-capability control: seed 7,
64 cells/channels, 5,000 updates, energy held at one, no fast efficacy, `3e-3` through
update 2,500, then cosine decay to `3e-4` at update 5,000. Its exact constant-rate
comparison is the completed S9 energy-one seed-7 run.
