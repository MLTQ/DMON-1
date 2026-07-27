# S22: Counterfactual reverse-credit traffic

## Question

Can SOL learn which dendrite should carry backward decoder credit by testing that routing
choice inside the same continuously adapting organism?

S21 learned a material, bounded routing policy from delayed global reward correlation,
but its capability was indistinguishable from the unrouted S17 control. The global
reward tells the organism whether a later prediction was surprising; it does not isolate
whether one earlier routing choice caused the change.

## Intervention

S22 turns branch alignment into a proposal rather than a fitness verdict:

1. installed-edge event/decoder alignment nominates one active branch and a bounded
   zero-sum preference perturbation inside its target-owned dendrite fan;
2. incumbent anatomy remains installed and forward traffic is unchanged;
3. the candidate perturbation is applied only to reverse decoder-credit traffic during
   a deterministic ABBA sequence of streamed optimizer windows;
4. candidate-on and incumbent windows both update the same hidden state, weights,
   optimizer, anatomy, structural probes, energy, event memory, and corpus position;
5. signed prequential reward is accumulated separately for the two routing conditions;
6. a positive candidate-minus-incumbent advantage commits the bounded preference
   change; a non-positive trial rejects it without reverting anything the body learned;
7. the next trial begins only after the previous decision is checkpointed.

The intervention is parameter-neutral. It does not create another organism, replay a
frozen organ, resize the network, or use a detached validation pass as the learning
signal.

## Scheduling

Only one routing trial is active at a time. Routing trials do not begin while a
structural exploratory-traffic trial is active or on a structural consolidation phase.
A routing start still executes that update's structural evidence/confirmation phase,
then resolves before the next structural decision. This separates two whole-body causal
questions without pausing ordinary learning or consuming the morphology organ's cadence.

Use the existing even-window ABBA convention:

```text
candidate · incumbent · incumbent · candidate
```

Each arm receives the same number of windows. The proposal affects reverse-credit
allocation during its candidate windows; reward observed on the live stream is retained
with the exact target, branch, direction, phase, update, and observation counts.

The matched preflight uses a 25-update shared cadence, two-phase structural
confirmation, a 75-update routing warmup, and 20-window trials. Structural decisions
remain due at 75, 125, 175, ..., 975. Routing uses the intervening non-decision phases:
100, 150, 200, 300, ..., 950, skipping each 250-update validation/checkpoint boundary.
Every routing trial resolves 20 updates after it starts, before the next structural
decision, and every plotted point therefore describes fully decided routing state.

## State and invariants

Checkpointed routing-trial state must include:

- active/inactive status and deterministic phase;
- target, branch slot, proposed zero-sum preference delta, and incumbent preference;
- candidate/incumbent reward sums and observation counts;
- start and decision updates;
- commit/reject counters and an append-only decision ledger.

Required invariants:

- default-disabled execution is bitwise historical;
- fan-in-normalized routing preserves the original total reverse-credit scale;
- dormant slots and unrelated target fans never receive the perturbation;
- rejection changes no preference, anatomy, parameters, optimizer moments, or stream
  state beyond the body's ordinary lived adaptation;
- commit changes only the target fan's bounded routing preference;
- forward messages, exploratory probes, and ordinary BPTT remain live in both arms;
- exact checkpoint resume produces the same next arm, reward aggregates, and decision;
- parameter count remains unchanged.

## Experimental gate

Before a full capability comparison:

1. synthetic positive and negative traffic trials must commit and reject respectively;
2. a test must prove shared body parameters change in every ABBA arm;
3. a reduced natural-stream run must produce balanced observations, both decisions when
   the evidence permits, finite routing state, and no topology or reachability failure;
4. graph every reduced validation together with trial timing and routing decisions;
5. continue to the matched RTX 4090 horizon only if trials resolve, routing changes are
   bounded, and the paired curve is not already adverse.

The primary capability control remains S17. S20 and S21 are mechanistic context, not
new baselines.

## Results

### Implementation and natural-stream smoke

The default-disabled implementation adds a separate checkpointed routing-trial policy
without learned parameters. The model preserves accepted preference, continuously
updates branch proposal eligibility, and accepts a temporary `(cells, dendrites)`
zero-sum delta only during candidate windows. Trainer telemetry and benchmark summaries
retain every arm and decision.

The first 200-update, 12-cell Tiny Shakespeare smoke deliberately gave routing and
structural traffic the same ten-update cadence. A first scheduling attempt allowed
structural probation to claim every phase and produced zero routing trials. That is an
invalid mechanism test, not a negative result. A provisional alternating schedule
produced:

| Mechanism | Started | Committed | Rejected | Active at boundary |
|---|---:|---:|---:|---:|
| Routing traffic | 10 | 7 | 3 | 0 |
| Structural traffic | 9 | 4 | 4 | 1 |

Every completed routing trial had exactly two candidate and two incumbent observations.
Ordinary body weights changed in both arms; structural probes and evidence continued;
the named topology remained fixed during each routing trial; structural traffic resumed
on its next reserved phase. Final anatomy had 25 active edges from 24 at birth.

Accepted routing was active and bounded:

| Mean active preference | Maximum preference | Mean routing deviation | Maximum deviation | Saturated slots |
|---:|---:|---:|---:|---:|
| 0.01289 | 0.11213 | 1.28% | 11.17% | 0% |

Held-out BPC moved from `5.450` at update 100 to `5.444` at update 200. With only two
validations this is a mechanism smoke, not a stopping-horizon or capability claim.

Initial implementation verification: 97 repository tests passed. They cover local zero-sum/constant-scale
traffic, positive commit, negative rejection, shared-body adaptation in both arms,
deterministic non-starvation, policy exclusivity, dormant masks, old-checkpoint
compatibility, and exact active-trial resume.

### Matched preflight

#### Rejected parity launch

The first three-seed RTX 4090 launch exposed a scheduler defect that the equal-warmup
smoke did not: routing occupied exactly every even structural phase, while structural
consolidation was also allowed only on those even phases. By update 500, every seed had
completed nine routing trials but **zero structural trials, rewires, spawns, or prunes**.
The live and birth-topology evaluation were therefore identical in every seed.

| Seed | Update 250 BPC | S22 − S17 | Update 500 BPC | S22 − S17 |
|---:|---:|---:|---:|---:|
| 7 | 4.447995 | +0.404574 | 3.927811 | +0.285330 |
| 13 | 4.308367 | +0.084082 | 3.807570 | +0.059470 |
| 21 | 4.268418 | -0.090179 | 3.931390 | -0.218864 |

The mixed loss is not routing evidence because anatomy exposure is unmatched. All
three services were stopped after their completed update-500 evaluations/checkpoints,
and the artifacts were retained as a scheduler regression case.

#### Corrected scheduler smoke

The scheduler now asks the structural policy whether a phase is a decision phase rather
than alternating from an unrelated warmup origin. Routing uses only non-decision
phases, still calls the structural phase to retain confirmations, and protects benchmark
evaluation boundaries. A fresh 200-update smoke with the same phase geometry produced:

| Mechanism | Started | Committed | Rejected | Active at boundary |
|---|---:|---:|---:|---:|
| Routing traffic | 7 | 3 | 4 | 0 |
| Structural traffic | 7 | 5 | 2 | 0 |

All fourteen trials were balanced two-candidate/two-incumbent observations. Anatomy
performed seven mutations (five spawns and two prunes), ending at 27 active edges from
24 at birth with complete sensory/output reachability. Routing skipped both
100/200-update reporting boundaries. Held-out BPC moved from `5.449` to `5.377`; with
two evaluations this remains a mechanism smoke, not a capability or convergence claim.

Corrected verification: 98 repository tests pass with four pre-existing DMON
return-value warnings.

#### Corrected 1,000-update result

All three corrected RTX 4090 services completed successfully and released the GPU.
Every 250-update evaluation boundary was trial-free. At each boundary, S22 exactly
matched S17's cumulative structural commit/reject, spawn, prune, and total-mutation
counts:

| Seed | Mutations at 250 / 500 / 750 / 1,000 | Final spawns | Final prunes |
|---:|---|---:|---:|
| 7 | 5 / 10 / 12 / 18 | 11 | 7 |
| 13 | 4 / 6 / 6 / 9 | 5 | 4 |
| 21 | 2 / 4 / 8 / 8 | 3 | 5 |

Against S17 over the last three preflight evaluations:

| Seed | Mean S22 − S17 BPC | Endpoint | S22 wins | Effect / noise |
|---:|---:|---:|---:|---:|
| 7 | +0.000020 | +0.000035 | 0 / 3 | 0.0011 |
| 13 | -0.000033 | +0.000026 | 2 / 3 | 0.0030 |
| 21 | +0.000068 | +0.000184 | 1 / 3 | 0.0032 |
| Three-seed mean curve | **+0.000018** | **+0.000081** | **2 / 3** | **0.0064** |

Pooled across seed/checkpoint pairs, S22 wins 3 of 9. Both mean curves are still
improving at about `-0.0544` BPC per 100 updates, but their relative slope is only
`+0.0000165` BPC per 100. Capability is indistinguishable at this horizon.

The mechanism is active and bounded:

| Seed | Trials | Commit / reject | Mean routing deviation | Maximum deviation | Saturation |
|---:|---:|---:|---:|---:|---:|
| 7 | 15 | 10 / 5 | 0.88% | 11.42% | 0% |
| 13 | 15 | 10 / 5 | 1.51% | 10.79% | 0% |
| 21 | 15 | 9 / 6 | 1.46% | 9.41% | 0% |

However, the supposed causal decisions are dominated by shared corpus phase. Seeds 7
and 13 make the same commit/reject decision in all 15 trials; seed 21 differs only once.
Fourteen of fifteen updates are unanimous across all three independently routed
organisms. Pairwise advantage correlations are `0.82–0.88`, despite different proposals
and weak/inconsistent proposal-evidence correlations.

This phase locking fails the causal-validity gate even though capability is non-adverse.
Do not resume S22 to 2,000 updates and do not promote it. A longer run would accumulate
more preferences selected mostly by which fixed ABBA windows happen to contain easier
corpus segments.

The next trial must randomize and checkpoint ABBA/BAAB block orientation independently
per proposal, retain the individual reward sequence, and require an exact
randomization-inference threshold before committing. That makes corpus-window phase a
measured null rather than a hidden fitness signal.
