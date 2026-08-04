# S1-P3b: private transition reserve and directional consolidation

Status: protocol frozen 2026-08-04 before implementation tests or GPU results.

## Question

Can a shared SOL2 organism retain mastered A and learn deep B without rehearsal when
reserve cells have genuine private transition capacity and may cheaply read stable A
circuits, while modification of useful producers remains expensive?

This is the direct successor to S1-P3. It does not add an expert router, isolated
subnetworks, explicit organ-conditioned modes, or external memory.

## Architecture

Use the same 400-cell geometry: 8 input, 64 stream-written memory, 256 compute, 64
relay, and 8 output cells; hidden width 192; 12 of 16 dendrites active; 5 microsteps;
8 output queries.

Every mutable cell receives a private rank-4 residual transition:

`r_i = 0.5 * tanh(tanh([h_i, message_i, drive_i] D_i) U_i)`

`D_i` is randomly initialized and `U_i` begins at zero, making the feature exactly
behaviorally silent before training. The residual changes the cell's target logits but
the typed tissue rule remains its shared genome. This adds local computational
capacity, not a private decoder or bypass.

## Acquisition and utility

Train seed 7 from scratch with batch 24, constant 1e-3 AdamW after 200 warmup updates,
the same fixed 1,000-update length stages, and a 10,000-update cap. Mastery remains two
consecutive 64-batch length-four checks at or above 80%.

After mastery, calibrate on 64 fixed-length-four A batches. Cell utility combines
private-expression/adapter gradient demand (50%), incoming edge demand (25%), and
downstream-reader demand (25%), ranked within tissue. An installed edge's protection
is the maximum of its own A-gradient rank and its target cell's utility. Source utility
is deliberately excluded: a reserve target may begin reading a stable useful source
without paying to modify that source.

## Directional reserve growth

Before B attachment, select the bottom 50% of internal targets by the branch's applied
cell utility and activate two dormant slots on each:

1. one reads a top-quartile internal source;
2. one reads another reserve internal source.

No active source or slot is removed. New slots begin at raw edge bias -1.5 and remain
fully plastic. Plastic, uniform, and measured arms use measured A utility for the same
graft anatomy; shuffled uses its within-tissue permutation. A accuracy immediately
after grafting must remain within 5 points of the pre-graft estimate or the structural
operation is invalid.

## Genome-rate screen

Run three `uniform` pilot branches for 1,000 B updates at genome plasticity 0.05, 0.15,
and 0.30. Cell and installed-edge plasticity are constant at the mean values induced by
the measured profile; new reserve edges are fully plastic. Select the rate with highest
terminal `min(A,B)` on 32 batches, breaking ties within 2 points toward higher B. This
screen tunes the frontier and is not itself a result arm.

## Result arms

From the exact mastered checkpoint, train each arm for 3,000 B updates without A
rehearsal, using identical B samples, evaluations every 300 updates, 16-batch interval
and 64-batch terminal estimates:

1. `plastic`: all parameters plastic; measured directional graft anatomy.
2. `uniform`: selected slow genome rate and constant mean cell/edge plasticity.
3. `consolidated`: selected rate and measured graded cell/adapter/directional-edge
   protection.
4. `shuffled`: identical protection distributions and selected rate, but cell/edge
   placement and reserve graft targets are permuted within tissue.

Protection remains `sigmoid((utility - 0.65) / 0.10)` with minimum plasticity 0.02.
The realized AdamW delta is scaled so normalization and weight decay cannot bypass the
cost. B-organ parameters and newly activated reserve edges remain fully plastic.

## Measurements and gates

- A and B accuracy by composition length and worst-organ accuracy over time.
- Drift of expression, private adapters, installed edges, and tissue genome by
  equal-count measured-utility quartile.
- Graft ledger, post-graft A perturbation, and learned use of new edges.
- High/low A-utility cell lesions, adapter-reset lesions, tissue freezes, and topology
  lesions.
- Stability and byte-identical A-organ integrity.

Capability success: at least one protected result arm ends with both A and B at 80%.
Measured allocation succeeds separately if its `min(A,B)` exceeds uniform by 10 points
and shuffling removes at least half that advantage. Reserve recruitment requires more
low- than high-utility adapter drift, causal B loss when recruited adapters/new edges
are reset, and no requirement that every cell be occupied.

A passing seed 7 mechanism must be repeated at seeds 13 and 21. Failure of one rank,
rate screen, or utility proxy does not establish that consolidation or cellular
differentiation is impossible.

## Compute boundary

All result-bearing work selects only physical RTX 4090 UUID
`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. The RTX 2070S remains reserved for
`jewels`. At most two branches share the 4090 after memory is measured.
