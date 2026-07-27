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
