# F9a: the associative memory organ (preregistered 2026-07-31)

## Question

F6 rounds 1–2: at 350k params with sparse exposure, no architecture —
creature, GRU, or transformer-within-window — bootstraps cued recall; every
arm sat at the 4.700-bit chance line. NTM (arXiv:1410.5401, Max's find) is
the decade-old existence proof that when addressable memory is supplied as a
*primitive*, small networks learn retrieval — they only have to learn to use
it. F9a transplants that claim: does a minimal memory organ let the creature
learn the pointer stream where three architectures could not?

## The organ (`fable/memory.py`)

- **Writes: ring buffer** — every token stores (key, value) projections of
  the current context (embedding + input-cell state) at the next slot. No
  learned write addressing; nothing for write-side optimization to game.
- **Reads: content-based** — learned query from output-cell state, cosine
  over stored keys, sharpened softmax, weighted value sum. Read enters the
  field as micro-step-0 drive on a dedicated 8-cell memory port (last 8
  internal cells).
- **Gate zero at init**: behaviorally identical to a memory-less creature
  (contract-checked); the organism must learn to consult the organ. The
  query-then-write order within a token means a query can never match its
  own write.
- 48 slots × width 64; +49,858 params (+13.4% over the F6 creature).
  Disclosed: the F6-round-2 nulls are the comparison and they are smaller
  models; the claim under test is *capability existence* vs a chance line,
  not parameter-matched efficiency. (A capacity-confound control exists for
  free: rounds 1–2 already show the failure is not capacity-shaped — the
  transformer had the capacity and the window and still sat at chance.)
- Lifetime caveats deliberately out of scope: 48 ring slots cover ~48 recent
  episodes; a never-reset creature needs DNC-style allocation/forgetting
  (usage tracking as organ metabolism) before this organ ships on the main
  line. This is a 12k-update capability test.

## Arms

Creature + organ on the pointer stream at the **exact F6 round-2 dial**
(B=12, 12k updates, inject 0.06, delays 16–1024, load 8), seeds 7/13/21, on
the 2070S. Baselines: the nine F6-round-2 arms already run (all at chance).

## Pass / fail

- **Pass**: recall BPC ≥ 1.0 bit below chance (≤ 3.70) in the d16–511
  buckets on ≥ 2 seeds. **Strong pass**: ≤ 2.0 bits below chance anywhere,
  or below-chance recall at d512–1024 (beyond the ring's ~48-episode span —
  would mean the field itself is helping).
- **Organ unused**: recall at chance AND gate near zero — gradient declined
  the organ; report the gate trajectory. Distinct from:
- **Organ used but useless**: gate open, recall still at chance — the
  primitive is consulted but not exploited; the retrieval-learning problem
  is upstream of addressing.
- Health/stability reported as always (the organ adds learned scale paths;
  the week's lesson says watch them).

## F9b (sketch, gated on F9a's plumbing verdict)

The memory that matches Max's actual thesis — remember *ways of thinking*,
not facts: slots store **expression profiles** (F8's γ/β configurations);
reads re-instate tissue modes rather than injecting content. Testbed: F2's
regime cycling, against the +5.05 retention penalty — snapshot the regime-A
configuration instead of protecting weights. Preregistered fully only after
F9a establishes the read/write plumbing works at all.
