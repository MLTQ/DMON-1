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

---

# RESULTS, round 1 (2026-07-31): **organ unused — the gate never opened**

Seed 7 (seeds 13/21 lost to a transient disk-full on Aine; rerunning):
recall at chance in every bucket (4.694–4.739 vs 4.700), train-half also at
chance, natural BPC healthy (2.200) — and **gate = −0.011 after 12,000
updates**. The preregistered "organ unused" branch, cleanly.

Diagnosis: a bootstrap trap. At init the organ's projections are random →
reads are noise → opening the gate adds noise → gradient pins the gate at
zero → the projections never receive gradient. The zero-gate silent-graft
discipline (right for stability grafts all week) is wrong for an organ that
must be used to be learned. NTM itself has no gate — its memory path is
live from step one. The gate was our deviation from the reference design,
and it reproduced a known pathology of gated-module bootstrapping.

## Round 2 (preregistered): NTM-faithful liveness + bounded path

1. Gate initialized at 0.1 (live from the start; still learnable).
2. The read drive gets a tanh bound before entering the field — with a live
   gate it is otherwise an unbounded learned-scale path into the recurrence,
   this week's four-times-documented detonation pattern.
3. All three seeds rerun at the same dial. Bars unchanged. The identity
   contract in smoke is replaced by a bounded-drive contract (behavioral
   identity at init is deliberately given up — that was the trap).

---

# RESULTS, round 2 (2026-07-31): still chance — but the organ was uninformative by construction

All seeds completed, zero skips. Recall 4.699–4.763 vs 4.700 chance; gates
*shrank* from 0.1 to 0.029–0.056 (sharpen grew slightly). The organ was
live, trained end-to-end, consulted — and rationally turned down, because
the design could not work: **the ring writes every token**, so 48 slots
span the last 48 tokens, not 48 episodes. Pairs are evicted ~48 tokens
after injection while queries arrive at delays up to 1024. As built, the
organ is a slightly wider mirror ring, and the gradient priced it as one.
The round-1 prereg's "48 slots cover ~48 recent episodes" was wrong at the
write-cadence level — an unexamined assumption, caught by the result.

## Round 3 (preregistered): surprise-gated writes

Write only when the incoming token's own prediction loss exceeds a per-lane
running EMA plus a fixed margin (0.7 nats). Task-general (no sentinel
knowledge: episodes self-select because random letters are maximally
surprising against text), biologically honest (surprise-gated episodic
encoding), and it fixes coverage: at ~10–15% write rate, 48 slots span
~350–500 tokens. Mechanics: state carries previous logits (surprise
computed inside the organism, no caller changes), per-lane ring cursors.
Bars unchanged. **Final round for F9a**: the failure ladder has been
"can't survive" → "won't engage" → "engaged but starved of coverage"; if a
live, bounded, surprise-fed organ still sits at chance, F9a banks as three
clean negatives and the retrieval question moves to the dense-curriculum
option (F6 round 3).
