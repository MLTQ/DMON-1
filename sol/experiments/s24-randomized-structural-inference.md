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
or capability result. The matched multi-seed GPU comparison against S23 remains
pending.
