# S1-P4: anchored consolidation and pressure-triggered development

Status: protocol frozen 2026-08-04 before implementation tests or GPU results.

## Question

Can the mastered organism retain procedure A while learning through organ B when mature
structure pays an increasing cost for cumulative displacement, and can persistent
learning demand on that anchored structure trigger growth of new plastic neurons?

S1-P3b showed that private adapters become causally necessary, but scaling each AdamW
delta merely delays unlimited drift. It also showed that preinstalled reserve grafts
were ignored. S1-P4 therefore changes the training dynamics and makes growth conditional
rather than adding an expert router or assigning cells permanently to an organ.

## Starting organism

Every branch begins from the exact seed-7 S1-P3b acquisition checkpoint:

- 400 cells: 8 input, 64 memory, 256 compute, 64 relay, and 8 output.
- Hidden width 192, private rank-4 transition adapters, 12 of 16 active dendrites,
  5 microsteps, and 8 output queries.
- A mastered at update 6,000 with 92.90% held-out length-four accuracy and zero rejected
  updates.

No S1-P3b reserve graft is applied before B training. Growth must be caused by measured
plasticity pressure.

## Utility and anchors

Recompute directional A utility on 64 held-out length-four batches. Cell utility uses
private identity/adapters (50%), incoming demand (25%), and outgoing-reader demand
(25%). Edge utility is the maximum of its own gradient rank and target utility; source
utility is excluded.

For every mature substrate parameter, retain an immutable A checkpoint anchor. After
each accepted AdamW step, apply the exact proximal update

`theta <- anchor + (theta_adam - anchor) / (1 + 0.01 * protection)`.

This is a cumulative restoring potential, not a gradient multiplier or learning-rate
reduction. Cell identity and adapter rows use cell protection; installed edge logits
use edge protection; shared tissue and graph operators use protection 1. Dormant edges,
new cells, newly installed readers, and organ B have protection 0.

Measured protection is `sigmoid((utility - 0.65) / 0.10)`. The uniform arm gives every
mature cell and installed edge the corresponding global mean protection. The plastic
arm has no anchor.

## Plasticity pressure and development

Before each optimizer step, measure squared B-gradient demand on the mature substrate.
Plasticity pressure is the fraction of this demand weighted by anchor protection.
Attached-organ gradients do not enter the signal. New unanchored rows contribute to
the accessible-plastic denominator.

At each 300-update evaluation, average pressure over the interval. A growth event is
eligible after update 600 when B is below 80% and either:

1. pressure is at least 0.75 for two consecutive checks; or
2. pressure is at least 0.60 and B improved by less than 3 points for two consecutive
   checks.

Growth has a 600-update refractory period and a maximum of two events. Each event
appends 16 relay cells with private rank-4 adapters, sensory-reachable inputs, and one
weak dormant reader per new cell. Existing anatomy, parameters, optimizer moments, and
living state are preserved exactly. New state and adapter-up rows begin at zero; new
structure remains unanchored for this adaptation episode. A fixed-example pre/post
growth drop greater than 5 points invalidates the event.

## Matched branches

All branches attach the same B organ, consume identical samples, train without A
rehearsal for 3,000 updates, evaluate every 300 updates on 16 batches, and use 64-batch
terminal estimates:

1. `plastic`: ordinary whole-organism plasticity, no growth.
2. `uniform_anchor`: mean anchor protection, no growth.
3. `measured_anchor`: measured anchor placement, no growth.
4. `developmental`: measured anchors plus pressure-triggered growth.

## Measurements and gates

- A/B length-four accuracy and worst-organ accuracy over time.
- Interval pressure, anchored versus accessible gradient demand, and anchor energy.
- Growth trigger ledger, geometry, parameter count, and fixed-example perturbation.
- Drift by measured-utility quartile for identity, adapters, edges, and shared genome.
- Terminal tissue lesions and equal-size high/low-utility lesions.
- For `developmental`, freeze all newly grown cells and zero their adapters on matched
  examples.
- A-organ byte integrity, rejected updates, and exact checkpoint resume.

Capability success requires one anchored arm to finish with A and B both at least 80%.
Anchor attribution requires measured anchor `min(A,B)` to exceed uniform anchor by at
least 10 points. Development succeeds separately if it triggers only under frozen
pressure conditions, improves `min(A,B)` by at least 10 points over measured anchor,
and removing grown cells costs B at least 10 points without improving A by more than
5 points.

One seed identifies a mechanism, not a general capability claim. A failed strength,
threshold, or growth size does not establish that anchored consolidation or endogenous
development is impossible.

## Compute boundary

All result-bearing processes select only physical RTX 4090 UUID
`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. The RTX 2070S remains reserved for other
work. At most two branches may share the 4090 after memory is measured.
