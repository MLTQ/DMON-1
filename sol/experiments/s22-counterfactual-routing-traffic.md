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
structural exploratory-traffic trial is active, and structural trials do not overwrite
the routing trial's named branch. This separates two whole-body causal questions
without pausing ordinary learning or disabling either exploration system.

Use the existing even-window ABBA convention:

```text
candidate · incumbent · incumbent · candidate
```

Each arm receives the same number of windows. The proposal affects reverse-credit
allocation during its candidate windows; reward observed on the live stream is retained
with the exact target, branch, direction, phase, update, and observation counts.

The matched preflight uses a 25-update shared cadence and a 75-update routing warmup.
Routing therefore starts at updates 75, 125, ..., 975 and each 20-window ABBA trial
resolves by 95, 145, ..., 995. The 250-update validation/checkpoint boundaries remain
trial-free, so every plotted point describes a fully decided routing state.

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
invalid mechanism test, not a negative result. Deterministic phase sharing now reserves
alternating opportunities, and a fresh run produced:

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

Verification: 97 repository tests pass. They cover local zero-sum/constant-scale
traffic, positive commit, negative rejection, shared-body adaptation in both arms,
deterministic non-starvation, policy exclusivity, dormant masks, old-checkpoint
compatibility, and exact active-trial resume.

### Matched preflight

Pending.
