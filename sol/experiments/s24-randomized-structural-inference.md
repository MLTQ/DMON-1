# S24: Randomized structural traffic inference

## Question

Can SOL decide which axons and dendrites to spawn or prune from candidate-specific
causal evidence after removing the fixed corpus phase from structural probation?

S23 proved that block-randomized `ABBA`/`BAAB` traffic eliminates phase locking for
reverse-credit routing. Its capability remained neutral. The same audit found that the
morphology organ still uses fixed ABBA traffic: shared structural-trial advantage
correlations reached `0.767` for seeds 7/13 and `0.791` for seeds 13/21, with 12 of 13
shared 13/21 trials taking the same sign.

## Intervention

Retain the incumbent connectome and S23 randomized routing, but add an explicit
default-disabled randomized mode to structural exploratory probation:

1. a locally and globally qualified probe nominates one candidate source/target/slot;
2. the incumbent anatomy remains installed for the complete trial;
3. five four-window blocks independently use `ABBA` or `BAAB`, where `A` exposes only
   the nominated candidate probe and `B` gates only that probe off;
4. hidden state, parameters, optimizer, energy, all other probes, routing traffic, and
   structural evidence remain live in both arms;
5. the assignment is keyed by topology seed, trial identity, target, slot, candidate,
   and start update without consuming training RNG;
6. every reward and the exact null rank are checkpointed;
7. a graft occurs only when candidate-minus-incumbent reward is positive, clears the
   configured margin, and has exact one-sided `p <= 0.10`;
8. rejection changes no anatomy and spends no growth energy.

Fixed ABBA remains available as the historical control. Older checkpoints retain that
policy exactly because randomized structural traffic has a separate configuration bit.

## Organ scheduling

Structural trials continue to start only on structural decision phases after the
required local/global confirmations. Randomized routing uses the intervening
non-decision phases and resolves before the next structural decision. A protected
evaluation/checkpoint boundary admits neither unfinished intervention.

The two interventions may influence the same living organism over time, but they never
overlap their causal traffic windows or own the same decision phase.

## Experimental gate

Before a GPU comparison:

1. randomized structural schedules are balanced and trend-neutral;
2. proposal identity changes the schedule without advancing global RNG;
3. injected benefit commits with exact rank while null and harm reject;
4. incumbent anatomy and energy remain exact until commit;
5. exact resume preserves the next arm, raw rewards, rank, and verdict;
6. an active historical fixed-ABBA checkpoint finishes under its old rule;
7. routing and structural phase cadences remain live and disjoint;
8. a natural-stream smoke retains full cell/output reachability;
9. the reduced matched graph must show structural advantage decorrelation before a
   longer horizon.

## Comparison

The primary control is S23: identical randomized routing and living-body configuration,
with historical fixed-ABBA structural probation. S17 remains context for capability and
anatomy, not the direct S24 causal control.

## Results

### Local living-network gate

A 400-update seed-7 CPU smoke retained the full S23 organism and changed only the
structural assignment/inference policy. All seven structural trials completed with
balanced candidate/incumbent exposure and exact `32`-assignment ranks:

| Trial starts | Assignment codes | Commit / reject |
|---|---|---|
| 75, 125, 175, 225, 275, 325, 375 | 16, 2, 31, 27, 6, 10, 25 | 1 / 6 |

The only commit had advantage `+0.20943` and exact `p = 1/32`; smaller positive
advantages of `+0.01098` and `+0.02050` rejected at `p = 0.375` and `0.3125`.
Randomized routing remained live and completed five independent trials. The connectome
made three total mutations (one spawn, two prunes), retained all 16 cells and all four
outputs as sensory-reachable, and finished with 31 active directed edges.

Held-out BPC improved from `5.0847` at update 200 to `4.6828` at update 400. With only
two evaluations this is explicitly an implementation/survival gate, not a convergence
or capability result. It qualified the matched multi-seed GPU comparison against S23.

### Matched GPU result

Seeds 7, 13, and 21 ran concurrently on the same RTX 4090 as their S23 controls. The
preregistered reduced stop was 1,000 updates; its graph showed complete schedule
decorrelation and unresolved capability movement, so each exact checkpoint resumed to
2,000 updates. Validation used 4,096 tokens every 250 updates.

Randomization removed the fixed structural phase:

| Aligned seed pair | S23 advantage correlation | S24 advantage correlation | Identical schedules, S23 → S24 |
|---|---:|---:|---:|
| 7 / 13 | +0.767 | +0.220 | 31/31 → 2/32 |
| 7 / 21 | +0.409 | −0.588 | 17/17 → 0/21 |
| 13 / 21 | +0.791 | −0.476 | 13/13 → 1/22 |

The mean signed schedule correlations in S24 were `+0.013`, `−0.124`, and `+0.200`;
the negative reward correlations therefore did not come from systematically
complementary schedules. Across trial starts shared by all three organisms, unanimous
advantage signs fell from 9/13 in S23 to 2/19 in S24. The causal decorrelation gate
passed.

The more reliable gate also changed development substantially:

| Policy | Structural trials | Growth commits | Prunes | Total mutations |
|---|---:|---:|---:|---:|
| S23 fixed ABBA + positive advantage | 87 | 35 | 34 | 69 |
| S24 randomized + exact `p <= 0.10` | 95 | 10 | 25 | 35 |

All six runs retained complete cell and output reachability. S24 ended with 26, 29, and
26 active edges for seeds 7, 13, and 21, versus 36, 31, and 30 in S23.

Capability paid for the lower mutation rate. S24 minus S23 mean held-out BPC was:

| Update | 250 | 500 | 750 | 1,000 | 1,250 | 1,500 | 1,750 | 2,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean gap | +0.155 | +0.085 | +0.082 | +0.134 | +0.087 | +0.132 | +0.080 | +0.077 |

Over the final five evaluations, S23 won all five mean points by `0.1021 BPC` on
average; the effect/noise ratio was `6.14`, so the comparison horizon supports S23 even
though neither mean curve is flat. S24 was improving faster (`−0.02077` versus
`−0.01600 BPC / 100 updates`), and seed 7 crossed S23 by `0.00046 BPC` at the endpoint.
Seed 13 remained `0.0333` worse and seed 21 `0.1994` worse. A linear continuation would
close the mean gap after roughly another 1,624 updates, but that is a diagnostic
extrapolation, not evidence to extend this experiment.

## Decision

S24 succeeds as a causal measurement intervention and fails as a capability promotion.
The next experiment should preserve randomized assignments and raw exact inference while
separating the inference threshold from developmental mutation rate. A small
`p`-threshold sweep can test whether S24 is simply too conservative, without returning
to the phase-locked fixed schedule or scaling unrelated input/output organs.
