# S1-P3: utility-gated consolidation with plastic reserve

Status: protocol frozen 2026-08-04 before implementation tests or GPU results.

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
