# S23: Randomized reverse-credit inference

## Question

Can live reverse-credit traffic identify a branch-specific causal benefit after removing
the corpus-window phase that dominated S22's fixed ABBA decisions?

S22 kept the body and morphology organ live, learned bounded routing preferences, and
was capability-neutral. Its causal verdict was invalid: 14 of 15 trial decisions had
the same sign in all three seeds and pairwise advantage correlations were `0.82–0.88`.
The shared Tiny Shakespeare windows, not the seed-specific route proposal, determined
most commit/reject outcomes.

## Intervention

Retain S22's parameter-neutral branch proposal and continuously adapting organism, but
replace one fixed crossover phase with checkpointed block randomization:

1. split each 20-update trial into five four-update blocks;
2. independently assign each block either `ABBA` or its complement `BAAB`, where `A`
   applies the routing candidate and `B` retains incumbent routing;
3. derive the five assignment bits from a stable mix of topology seed, trial identity,
   target, slot, and start update, without consuming the training RNG;
4. store the complete assignment and every observed prequential reward;
5. at resolution, enumerate all `2^5 = 32` valid block assignments on that reward
   sequence;
6. compute the exact one-sided fraction whose candidate-minus-incumbent advantage is at
   least as large as the observed assignment;
7. commit only when advantage is positive, clears the configured margin, and has exact
   randomization `p <= 0.10`.

Every block contains two candidate and two incumbent windows. Both `ABBA` and `BAAB`
have zero covariance with a linear within-block time trend, so randomization removes
fixed corpus phase without reintroducing simple learning-curve drift.

## Continuous-organism constraints

- Slow parameters, hidden state, optimizer, energy, stream cursor, exploratory probes,
  and morphology remain live in every arm.
- Routing may start only on a structural non-decision phase; that phase still updates
  structural confirmations.
- The routing trial resolves before the next structural decision.
- No trial may start or cross a held-out evaluation/checkpoint boundary.
- Rejection retains all lived body adaptation and changes no committed preference.
- Randomization consumes no global PyTorch or Python RNG state.

## Checkpoint and telemetry

Active trial state must retain:

- assignment code and full candidate/incumbent schedule;
- next schedule index;
- individual reward sequence plus per-arm aggregates;
- observed advantage, exact null rank, and one-sided p-value;
- target, slot, direction, proposal evidence, start update, and proposed delta.

The resolved ledger must retain the same evidence. An active checkpoint from S22 that
lacks randomized observations finishes under its historical fixed-phase decision rule;
all newly begun trials use S23.

## Experimental gate

Before another GPU capability comparison:

1. schedules must be exactly balanced and blockwise trend-neutral;
2. schedules must differ across topology seeds without advancing global RNG;
3. an injected candidate effect must reach the exact threshold and commit;
4. a null or reversed effect must reject;
5. exact mid-trial resume must preserve the next arm, reward sequence, p-value, and
   verdict;
6. a natural-stream smoke must keep morphology live and reporting boundaries clean;
7. the reduced three-seed graph must show trial advantages are no longer phase-locked;
8. only then may a non-adverse capability result continue beyond 1,000 updates.

## Results

### Implementation and local preflight

The implementation stores the complete crossover assignment, every reward, observed
advantage, exact extreme-assignment count, null size, and p-value. A provisional exact
rank is available after each complete four-window block. The observed arm is checked
against the checkpointed schedule before a reward can be attributed, and an active S22
checkpoint finishes under its historical fixed-phase rule.

The full repository gate passes: `101` tests with four pre-existing DMON return-value
warnings. Focused coverage proves schedule balance, zero blockwise linear trend,
different assignments for topology seeds `7`, `13`, and `21`, no global RNG
consumption, exact null/effect ranks, positive/harmful verdicts, live body learning,
structural/reporting phase separation, and exact mid-trial resume.

An initial 400-update CPU smoke supplied the complete morphology configuration but
omitted `--structural-plasticity`. Its fixed anatomy makes it invalid for the living-body
gate; it is retained separately and excluded.

The corrected uninterrupted 16-cell Tiny Shakespeare smoke kept both organs live:

| Mechanism | Started | Committed | Rejected | Active at update 400 |
|---|---:|---:|---:|---:|
| Randomized routing traffic | 5 | 1 | 4 | 0 |
| Structural exploratory traffic | 7 | 3 | 4 | 0 |

Routing trials ran at updates `100–120`, `150–170`, `250–270`, `300–320`, and
`350–370`, so neither the update-200 nor update-400 evaluation boundary contained an
active intervention. All five trials retained 20 raw rewards and an exact rank out of
32 assignments. Their p-values were `0.03125`, `0.96875`, `0.875`, `0.46875`, and
`0.28125`; only the first cleared the preregistered positive-effect gate.

Morphology performed eight mutations—three spawns and five prunes—while routing was
tested. Active edges changed from `32` to `30`; all 16 cells and all four output cells
remained reachable. Held-out BPC improved from `4.956` at update 200 to `4.283` at
update 400. Two validations establish mechanism survival, not convergence or a
capability ordering.

The local gate therefore permits the reduced matched GPU run. Cross-seed phase
decorrelation and anatomy cadence matching remain unproven until that run completes.

### Matched RTX 4090 result

Three seed-matched organisms ran concurrently on the RTX 4090. The first 1,000-update
gate passed: all corresponding schedules differed across seeds, only one of 15 trial
signs was unanimous, morphology remained live, and the capability curve was non-adverse.
The same checkpoints then continued without resetting to 2,000 updates. All services
completed successfully and released the GPU.

Across the complete horizon, each seed resolved 31 routing trials:

| Seed | Commit / reject | Final mutations | Spawn / prune | Final output reachability |
|---:|---:|---:|---:|---:|
| 7 | 2 / 29 | 32 | 18 / 14 | 4 / 4 |
| 13 | 2 / 29 | 25 | 12 / 13 | 4 / 4 |
| 21 | 5 / 26 | 12 | 5 / 7 | 4 / 4 |

Nine of 93 trials cleared `p <= 0.10`, close to the `8.72` expected from a discrete
five-block null alone. This is not population-level evidence that arbitrary routing
proposals help. The nine exact-positive local decisions remain useful as bounded online
interventions; their downstream capability must still earn a matched improvement.

#### Causal-validity verdict

Randomization removed the fixed corpus phase:

| Seed pair | S22 advantage correlation | S23 correlation | S23 sign match | Identical schedules |
|---|---:|---:|---:|---:|
| 7 / 13 | 0.882 | -0.380 | 14 / 31 | 1 / 31 |
| 7 / 21 | 0.874 | 0.276 | 16 / 31 | 2 / 31 |
| 13 / 21 | 0.821 | 0.012 | 17 / 31 | 0 / 31 |

Only eight of 31 aligned trial updates had one sign across all three seeds, almost
exactly the `31 / 4 = 7.75` expected from independent binary signs. S22 produced 14 of
15. S23 therefore passes the causal-validity gate.

Structural decision opportunities remained cadence-matched to S17. Cumulative mutation
counts matched at 23 of 24 seed/evaluation points; seed 13 made one mutation one
evaluation later and matched again thereafter. That transient divergence is an allowed
downstream consequence of changed reverse credit, not a frozen-organ confound. Every
final topology retained all cells and outputs.

#### Capability and horizon

Over the last five held-out evaluations:

| Seed | Mean S23 - S17 BPC | Endpoint | S23 wins |
|---:|---:|---:|---:|
| 7 | -0.000058 | -0.000236 | 3 / 5 |
| 13 | -0.000856 | -0.000771 | 5 / 5 |
| 21 | +0.000027 | +0.000010 | 1 / 5 |
| Three-seed mean curve | **-0.000296** | **-0.000332** | **4 / 5** |

The apparent mean advantage is only `0.034` times combined terminal residual noise.
S23 and S17 still improve at `-0.01600` and `-0.01594` BPC per 100 updates
respectively, with a relative slope of only `-0.000062` BPC per 100. All runs are stable
and finish at their best checkpoint, but the capability ordering is inconclusive.

Do not extend S23 further and do not promote it as a capability improvement. It solved
the measurement defect it was designed to solve; its routing preferences are
capability-neutral at this scale.

### Next falsifiable step

The same audit exposes the next confound in the adaptive communication organ:
structural exploratory probation still uses fixed ABBA traffic. At shared trial starts,
its advantage correlations are `0.767` for seeds 7/13 and `0.791` for seeds 13/21;
the latter shares the same sign in 12 of 13 trials. S24 should apply checkpointed
block randomization and exact inference to axon/dendrite spawn and prune decisions
while keeping S23 routing, ordinary learning, and both organ cadences live.
