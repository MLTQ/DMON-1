# S8: Output-originating backward credit

## Question

Can SOL preserve event-addressed reward while sending learning signal in both directions
through its directed connectome, rather than broadcasting one scalar reward to every
cell at once?

## Mechanism

- The observed next-character surprise still becomes one bounded signed reward.
- Fast synaptic efficacy consumes that scalar reward once, as before.
- An optional parameter-neutral `backward_credit` field is injected at designated output
  cells.
- Each target sends credit to its named sources using the signed coefficient of the same
  axon that carried forward activity.
- The bounded wave decays but persists across characters and optimizer windows.
- Credit affects a cell only where it meets that cell's retained event eligibility.
- Direct scalar cell reward and backward credit have separate gains, allowing direct,
  reverse-only, and combined controls without changing parameter count.

Older checkpoints add a zero-valued credit field. Evaluation shuffles credit with cell
identity, and the local bridge reports its live magnitude.

## Protected contracts

Tests require:

- exact target-to-named-source reverse direction;
- signed coefficient transport with bounded state;
- output reward launching a persistent wave;
- zero reward creating no credit;
- credit changing only event-tagged cell state;
- exact checkpoint resume and safe old-checkpoint upgrade;
- aligned state shuffle and held-out telemetry;
- unchanged one-shot fast-synapse reward semantics.

The full repository suite passes 57 tests, with the same four pre-existing
`PytestReturnNotNoneWarning` warnings in `dmon/test_conservation.py`.

## Reduced-scale protocol

Tiny Shakespeare, 16 cells by 16 channels, four dendrites, two message steps, 800
updates, four lanes, 16-character windows, no metabolism, no fast efficacy, and seeds
7, 13, and 21. The matched control uses direct cell reward gain `0.25`. Candidate waves
use decay `0.8`.

### Reverse-only calibration

Seed 7 selected reverse gain `0.25`; gains `0.5` and `1.0` were worse. Replicating gain
`0.25`:

| Seed | Control best/final | Reverse-only best/final | Final improvement |
|---:|---:|---:|---:|
| 7 | 3.80448 / 3.90971 | 3.79346 / 3.89644 | +0.01327 |
| 13 | 3.69447 / 3.81898 | 3.70216 / 3.78568 | +0.03331 |
| 21 | 3.71146 / 3.72659 | 3.70798 / 3.74740 | −0.02081 |

Mean final improvement was `0.00859` BPC and mean best improvement was `0.00227`.
Across all paired evaluations, reverse-only was `0.00490` BPC worse. Reverse transport
can replace broadcast reward without collapse, but does not improve capability
consistently on its own.

### Direct plus reverse

Adding a reverse gain of `0.25` to the existing direct gain:

| Seed | Control best/final | Combined best/final | Final improvement |
|---:|---:|---:|---:|
| 7 | 3.80448 / 3.90971 | 3.76390 / 3.76390 | +0.14582 |
| 13 | 3.69447 / 3.81898 | 3.71979 / 3.77596 | +0.04302 |
| 21 | 3.71146 / 3.72659 | 3.70653 / 3.78690 | −0.06032 |

Mean improvements were:

- best checkpoint: `0.00673` BPC;
- final checkpoint: `0.04284` BPC;
- all paired evaluations: `0.00520` BPC.

Lowering reverse gain to `0.1` reduced the mean final improvement to `0.01417` BPC and
worsened the all-evaluation mean by `0.00793`, so it did not solve the seed sensitivity.

## Decision

The mechanism passes: reward measurably travels backward through actual signed axons,
retains event memory across optimizer boundaries, and remains stable and
parameter-neutral. Capability evidence is preliminary, not conclusive. Gain `0.25`
improves all three aggregate criteria, but seed 21's endpoint regression prevents a
robust positive claim.

Keep the implementation and queue one original-budget matched comparison after CUDA is
restored. Do not promote a checkpoint or combine the mechanism with structural growth
until that comparison shows a stable, meaningful win.
