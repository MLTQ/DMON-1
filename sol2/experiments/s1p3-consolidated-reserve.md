# S1-P3: utility-gated consolidation with plastic reserve

Status: seed-7 4090 run complete; graded consolidation improved the stability/plasticity
frontier, but simultaneous mastery and measured-utility attribution gates failed.

## Question

Can a larger organism preserve mastered procedure A while learning the same procedure
through a new B sensor/effector organ, if modifications to causally useful cells and
connections cost more energy while low-utility reserve remains plastic?

This tests self-organized overlapping circuitry, not isolated expert modules or an
explicit router. It also protects functional modification rather than topology alone:
S1-P2 forgot A even though no source indices were rewired.

## Starting organism

Train seed 7 from scratch to the same two-check mastery gate used by S1-P2:

- 400 cells: 8 input, 64 stream-written memory, 256 compute, 64 relay, 8 output.
- Hidden width 192, 12 of 16 dendrites active, 5 microsteps/token, 8 organ queries.
- Batch 24, constant 1e-3 AdamW after 200 warmup updates.
- Lengths one through four admitted in fixed 1,000-update stages.
- Mastery requires at least 80% held-out length-four accuracy twice consecutively;
  hard cap 10,000 updates.

If this organism does not master A, S1-P3 stops. A smaller mastered organism may be used
only in a separately named pilot, never silently substituted into the result.

## Frozen utility definition

On 64 held-out fixed-length-four A batches, without updates, accumulate:

1. RMS gradients of each cell's private target/time expression.
2. Absolute gradients of installed edge-attention logits.
3. Incoming edge demand and downstream-reader demand for each cell.

Rank each component within tissue, combine cell utility as 50% private-expression,
25% incoming-edge, and 25% outgoing-reader demand, then rank again within tissue.
An edge's utility is the maximum of its own rank and the utilities of its source and
target. Dormant slots have zero measured utility and stay fully plastic.

Utility is causal first-order sensitivity, not mere activation. It is still an
approximation: success promotes it; failure does not prove consolidation impossible.

## Matched branches

Every branch starts from the exact mastered checkpoint, attaches the identically seeded
B organ, consumes identical B examples, and trains B for 2,000 updates without A
rehearsal.

1. `plastic`: ordinary whole-organism plasticity.
2. `consolidated`: measured utility controls modification cost.
3. `shuffled`: the same utility values are permuted within tissue before protection.

For the two protected arms, protection is
`sigmoid((utility - 0.65) / 0.10)` and realized cell/edge AdamW deltas are multiplied by
`1 - 0.98 * protection`. Typed tissue rules and shared graph query/key/value operators
use 0.05 plasticity. The B organ is fully plastic. Scaling the realized delta, rather
than the raw gradient, prevents Adam normalization and decoupled weight decay from
bypassing the energy cost.

## Measurements and gates

- A and B fixed-length-four accuracy every 200 updates; 64-batch terminal estimates.
- Per-length B curves, rejected updates, lesion penalties, and neuron telemetry.
- Parameter drift by equal-count measured-utility quartile for private expression and
  edges.
- Equal-size high-utility and low-utility internal-cell lesions on both A and B.
- Primary success: terminal A and B are both at least 80%.
- Attribution: consolidation improves `min(A,B)` by at least 20 percentage points over
  `plastic`, and shuffling removes at least half that gain.
- Recruitment: low-utility private/edge structure changes more than high-utility
  structure, while a high-utility lesion still harms B enough to show partial reuse.
- Stability: zero rejected updates and finite telemetry.

Some unused internal neurons are expected and desirable reserve. Neither success nor
failure requires every neuron to become active. One seed establishes a mechanism, not
a general capability claim; a passing result must be repeated at seeds 13 and 21.

## Compute boundary

All result-bearing processes select only physical RTX 4090 UUID
`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. The RTX 2070S is reserved for `jewels` and
must never receive an S1-P3 process. At most two branches may share the 4090 after
observed memory confirms that doing so is safe.

## Seed-7 result

The 400-cell organism mastered A at update 6,000 with 94.53% held-out length-four
accuracy, 1,288,790 parameters, and zero rejected updates. Attaching B raised each
branch to 1,445,292 parameters. Plastic and measured branches ran together at roughly
22.9/24.6 GB on the 4090; shuffled ran alone. No SOL2 process used the reserved 2070S.

| Branch | A length 4 | B length 4 | Worst(A,B) | Rejected updates |
|---|---:|---:|---:|---:|
| Plastic | 13.48% | 97.46% | 13.48% | 0 |
| Measured consolidation | 86.98% | 33.72% | 33.72% | 0 |
| Shuffled consolidation | 83.14% | 34.05% | 34.05% | 0 |

Measured consolidation improved the worst retained capability by 20.25 points over
plastic and therefore barely cleared that isolated gate. Shuffled consolidation
improved it by 20.57 points and was marginally better, so shuffling removed none of the
gain. Neither protected arm approached simultaneous 80/80; primary success is false.
The complete aggregate is retained in `s1p3-consolidated-reserve-result.json`.

The measured mask did exactly localize change. Low-A-utility cell expression drifted
7.63 times more than the high-utility quartile, and low-utility edges drifted 11.25
times more. Yet this localization did not improve capability over shuffled placement.
The result supports graded slowing and heterogeneous plastic reserve, but does not
validate this first-order gradient ranking as the allocator of that reserve.

Measured-consolidation B accuracy by length was 98.83%, 89.45%, 56.97%, and 33.72%.
It learned primitives and shallow composition but not a second deep procedure. Acute
freezing of all internal cells reduced its B accuracy from 34.24% to 11.91%, confirming
that the partial B capability used the organism. Freezing 80 high-A-utility internal
cells cost B 17.97 points, versus 5.66 points for 80 low-utility cells: B reused some
consolidated A machinery while recruiting reserve. A-organ parameters remained
byte-identical in every branch.

## Architectural conclusion

This is not evidence that consolidation is the wrong direction. It exposes two
specific limitations of this implementation:

1. Doubling 208 cells to 400 added only 64,512 parameters because the typed transition
   rules remain shared. Reserve cells own target/time biases and edge logits, but no
   independent transition operator; slowing the shared genome therefore protects A
   and simultaneously limits what reserve can compute.
2. Edge protection used the maximum utility of the edge, source, and target. That makes
   a low-utility reserve cell pay to begin reading a useful consolidated source, which
   violates the intended directional rule that reading stable knowledge should be
   cheap while modifying the producer should be expensive.

The next stepping stone should retain the natural shared organism rather than add an
expert router: give each mutable cell a small private low-rank transition residual;
protect those adapters by causal utility; protect an edge according to its own A demand
and target utility, not source utility; and activate dormant dendrites only on reserve
targets so they can cheaply read consolidated circuits. Add a uniform-slowdown control
beside plastic, measured, and shuffled arms to separate the slow-genome effect from the
benefit of heterogeneous plastic islands. Sweep a small number of genome rates rather
than treating 0.05 as uniquely correct. Longer training of the current arm may trace a
frontier, but it is unlikely to create the missing cell-local computational capacity.
