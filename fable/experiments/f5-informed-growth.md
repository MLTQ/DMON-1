# F5 (DRAFT): reward-tied growth of transport tissue, with informed proposals

*Draft filed 2026-07-30 during design discussion. Supersedes the
"hand-placed mixed tissue" plan (DMON-1-yx1), which rested on an incoherent
premise — see Reframing.*

## Reframing: why static placement cannot answer this

The earlier plan was to hand-place a relay population, with random placement
as an unbiased control, to establish whether a transport cell type has value
before building any growth mechanism. That plan is wrong for two reasons.

**1. White matter is definitionally positional.** A transport cell has no
local computation to contribute; its entire function is the path it
completes. A relay at a random position is not a weak relay — it is noise
occupying a dendrite slot that previously carried signal. So "test the type,
then test the placement" is not a coherent decomposition for this type. (It
would be coherent for the liquid cell of F3, which has local value
independent of position.)

**2. The random-wiring condition is already measured, and it is the failure
mode.** `DendriteGraph._wire` samples dendrites uniformly; fable's connectome
*is* random. F0 measured shuffle-internal deltas of +0.066 (s7) and +0.045
(s13): randomly wired internal tissue is near-inert while the creature still
reaches GRU parity on BPC via port-cell state. A random-placement arm would
have reproduced exactly this regime and reported "no marginal value" for a
type whose value is conditional on placement.

## Reinterpretation of sol's structural record (S2–S25)

sol's search generated candidates as **one weak random exploratory probe per
target**, then improved the *measurement* apparatus across ten experiments —
reversible probation (S5), balanced within-organism ABBA traffic (S6),
counterfactual crossover (S22), block randomization with exact inference
(S23/S24), alpha calibration (S25). The proposal distribution stayed random
throughout.

S23's headline is the diagnostic: **9 of 93 trials cleared p ≤ 0.10 against
8.72 expected under the pure null.** That is not low statistical power. That
is rigorous measurement of a proposal stream containing no signal.

Conclusion carried into F5: the historical bottleneck was **proposal
quality, not inference quality**. The inference machinery (S23/S24) is
validated and should be reused as-is; the thing that has never been tried is
proposing edges from a distribution that carries information.

## Informed proposal distribution

Both halves are already computed in this codebase:

- **Demand** — decoder-shaped reverse credit (sol S15, the best-behaved
  backward signal found: better aligned-trajectory behaviour than scalar
  reverse credit and an order of magnitude smaller adverse-seed regression).
  Identifies cells whose state change would reduce output error.
- **Supply** — forward edge activity / message flow. Identifies cells
  carrying signal. Closest precedent is the one physiology mechanism that
  cleanly passed a full-scale gate: S11's maintenance floor on
  activity-measured flow.

Propose axons from high-supply sources to high-demand sinks **with no
current path between them** (path check reuses the existing reachability
BFS). This is "grow white matter where information needs to travel and
cannot" — the biological reading being that myelination follows activity,
not a fitness prediction.

Note this is *not* the S19–S21 routing family, which was rejected. Those
used alignment to reweight *existing* branches' reverse-credit amplitude.
F5 uses these signals only to *propose new edges*, with commit decisions made
by measured outcome.

## Primary metric changes: shuffle-internal delta, not BPC

F0 established that BPC is dominated by port-cell state — the creature
reached GRU parity while internal tissue was inert. BPC is therefore a poor
instrument for a mechanism whose job is to make tissue load-bearing.

- **Primary**: shuffle-internal delta. Growth that moves it from ~0.05
  toward ~1.0 has built a functioning field.
- **Secondary**: held-out BPC, reset/mirror deltas, commit counts, and the
  fraction of commits that survive to the end of the run.
- Growth that improves BPC while shuffle-delta stays flat has improved the
  ports and **must not be credited to structure**.

## Arms (3 seeds, F0 substrate/geometry where parity is established)

| Arm | Proposals | Commits |
|-----|-----------|---------|
| `static` | — | — (F0 creature, incumbent) |
| `probes_only` | informed | none (mandatory per sol findings #11: same exploratory traffic, no installs) |
| `random_growth` | random | randomized-inference commits |
| `informed_growth` | demand×supply | randomized-inference commits |

`informed_growth` vs `random_growth` **is** the claim under test: if the
space of connections carries as much information as the computation, informed
proposals beat random ones by a wide margin. `probes_only` separates commits
from the disturbance of probe traffic.

## Pass / fail

- **Pass**: `informed_growth` raises shuffle-internal delta substantially
  above `static` and `random_growth`, consistently across seeds, without a
  BPC regression > 0.05 vs `static`.
- **Informative negative**: informed ≈ random on both metrics — the
  proposal-quality thesis is wrong and the bottleneck is elsewhere.
- **Fail (mechanism)**: neither growth arm differs from `probes_only` —
  commits are doing nothing and the disturbance is the whole effect.

## Inference reuse, with S24/S25's rate-knob caution

Reuse block-randomized crossover with exact randomization inference
(S23/S24) — it converted seed-pair decision correlations of 0.82–0.88 to
approximately zero and is the validated decision procedure. But S24 showed
reliability without a rate knob is expensive: p ≤ 0.10 cut growth commits
35 → 10 and cost 0.1021 BPC. Pool evidence **per proposal class** rather than
per slot, and treat alpha as a calibrated parameter (S25's open question),
not a constant.
