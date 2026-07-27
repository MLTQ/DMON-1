# S17: Global-loss-gated adaptive connectome

## Question

Does requiring a candidate axon to improve the exact current sequence loss make
credit-guided growth and pruning more capable than S16's decoder-credit-qualified
policy?

S16 established that adaptive sparse anatomy is useful: it beat an equally sparse
fixed anatomy in 14 of 15 terminal paired evaluations across three seeds. It did not
match fixed full fan-in. S17 therefore changes credit selectivity, not model scale,
input/output cell count, exploratory traffic, or the fan-in budget.

## Intervention

The treatment adds `--structural-global-fitness` to the S16 adaptive configuration.
Every proposed source still needs positive scalar or decoder-vector evidence, repeated
confirmation, endpoint energy, reachability, and a successful live ABBA traffic trial.
The additional gate requires the candidate probe's exact first-order effect on the
ordinary sequence loss to be beneficial before it can earn a confirmation.

This signal does not freeze the body or evaluate a detached organ. Ordinary parameters,
hidden state, eligibility, reverse credit, all other probes, and optimizer state remain
live throughout candidate-on and candidate-off windows.

## Matched protocol

- Tiny Shakespeare character stream;
- 16 cells, 16 channels, four parameter slots per target;
- two live dendrites per target at birth;
- four sensory and four output cells;
- two message steps per character;
- output-error credit gain `0.5`;
- structural vector-credit gain `10`;
- weak locality gain `0.001`;
- one structural phase every 25 updates after update 50;
- two confirmation phases and 20-update exploratory probation;
- variable fan-in with one-edge minimum;
- identical learning rate, batch, chunk, seed, and 2,000-update horizon;
- seeds 7, 13, and 21 run concurrently on the RTX 4090.

The S16 adaptive organisms are the matched controls. Their terminal evaluations from
updates 1,000 through 2,000 used the same 4,096-character validation regime as S17.

## Decision rule

The horizon is meaningful when the paired treatment/control ordering is persistent
relative to terminal residual noise; mathematical zero slope is not required. Report:

- every held-out BPC curve through update 2,000;
- treatment minus S16 adaptive gaps at the five aligned terminal evaluations;
- per-seed and pooled terminal wins, endpoint gaps, and effect/noise ratios;
- spawn, prune, rejection, and active-edge counts;
- comparison with fixed full fan-in as descriptive headroom, with device differences
  disclosed.

Promote the global-fitness gate only if the capability benefit replicates. Fewer
structural events alone is not success.

## Results

All three runs completed the predeclared 2,000-update horizon on the RTX 4090. Relative
to the same-seed S16 adaptive organisms, the five aligned terminal evaluations gave:

| Seed | Mean BPC improvement | Terminal wins | Effect / residual noise | Endpoint improvement |
|---|---:|---:|---:|---:|
| 7 | 0.0738 | 5 / 5 | 4.10x | 0.0550 |
| 13 | 0.0005 | 2 / 5 | 0.02x | 0.0193 |
| 21 | 0.0248 | 4 / 5 | 0.84x | 0.0172 |

The treatment wins 11 of 15 seed-level terminal measurements and all three endpoints.
The three-seed mean favors the gate at every terminal checkpoint, by `0.0331` BPC on
average and `0.0305` BPC at update 2,000. Its mean-curve effect is `2.71x` the combined
terminal residual noise, so the pooled ordering clears the comparison-level horizon
gate despite continued learning in both arms.

Seed trajectories are meaningfully different. Seed 7 independently supports the
intervention; seed 13 is indistinguishable over the terminal window despite its better
endpoint; seed 21 is directionally favorable but noisy. Three seeds therefore prevent
the strong seed-7 result from being mistaken for a universal effect.

## Anatomy

Global-loss gating preserved actual growth and pruning:

| Seed | Active edges at 2,000 | Spawns | Prunes | Traffic commits | Traffic rejects |
|---|---:|---:|---:|---:|---:|
| 7 | 36 / 64 | 18 | 14 | 18 | 19 |
| 13 | 31 / 64 | 12 | 13 | 12 | 21 |
| 21 | 30 / 64 | 5 | 7 | 5 | 12 |

S16 ended with 36, 31, and 38 active edges. The gate leaves seeds 7 and 13 at the same
net fan-in while sharply limiting seed-21 growth, yet all three gated endpoints improve.
That is evidence of better selectivity rather than an indiscriminate capacity increase.

## Decision

S17 is a modest positive screening result. Retain the global-fitness gate as the working
adaptive policy for the next credit experiment, but do not promote this small model to
the live UI or claim a per-seed universal capability gain. Mean endpoint BPC improves
from `3.4424` to `3.4118`; fixed full fan-in remains lower at `3.2938`, leaving `0.1181`
BPC of descriptive headroom.

The next isolated intervention should improve the reverse credit signal that selects
and trains communication paths, while keeping the S17 gate, live exploratory traffic,
and 2,000-update representation unchanged.
