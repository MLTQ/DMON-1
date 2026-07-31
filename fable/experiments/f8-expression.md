# F8: per-cell expression — can cells own an identity? (preregistered 2026-07-31)

## The observation that produced this experiment (Max)

"Each cell should have many parameters." It doesn't. Audit at F0 geometry
(128 cells, h=128, 348,609 params):

| Owner | Params | Shared? |
|---|---:|---|
| cell rule (one GRUCell) | 148,224 | one copy, all cells |
| attention q/k/v | 49,152 | one copy, all cells |
| embedding + readout + norms | ~145,600 | global |
| sensory affine | 4,096 | input cells only |
| **per-edge dendrite logits** | **1,536** | **the only per-cell params** |

**Twelve scalars per cell** — 0.4% of the model. A cell's entire learnable
identity is how loudly it listens to each of 12 randomly-drawn inputs. Two
cells with similar wiring are the same animal.

This is the mechanical explanation of every differentiation null measured so
far: shuffle-internal ~0.02 in every arm at every size (F0, F1), 4× tissue
worth 0.014 BPC (F1), and F2's forgetting (a mean field has no substructure
to protect). Cells are interchangeable **by construction**, so the shuffle
probe was measuring a symmetry the architecture guarantees.

Biology shares the genome but *not* the parameters: each neuron carries
thousands of individually tunable synapses and differentiates by gene
expression — persistent per-cell modulation of shared machinery. Our cells
have 12 synapse-equivalents and no expression.

## The change

Per-cell expression vectors: `msg_i → msg_i·(1+γ_i) + β_i`, γ/β learned per
cell, **zero-initialized** so the organism starts behaviorally identical to
today's (contract-checked) and every cell earns its identity by gradient.
Same shared rule (genome); per-cell expression on top.

Cost 2·N·H = 32,768 params (+9.4% at F0 geometry). Baselines rematch
automatically. Growth compatibility: a new cell is born with zero
expression — undifferentiated, stem-like, differentiating in place — but
the row-append + optimizer surgery is not written yet, so `grow.py` raises
rather than silently mis-growing (F1+F8 needs that first).

## Arms

Creature with expression vs the F0 incumbent, F0 protocol (annealed 8k,
B=12), seeds 7/13/21, plus a freshly parameter-matched GRU (the +9.4% is
disclosed and matched, not smuggled).

## Pass / fail — the primary metric is differentiation, not BPC

- **Differentiation pass**: shuffle-internal delta ≥ 0.30 BPC mean across
  seeds (vs the incumbent's 0.036–0.066). This is the first architecture in
  this project able, in principle, to break cell-permutation symmetry; the
  test is whether it does.
- **Capability**: BPC must not regress > 0.05 vs incumbent at matched
  budget. Improvement is welcome but is not what F8 is for.
- **Expression-spread read** (descriptive): the distribution of ‖γ_i‖ across
  cells at the end. Collapse toward zero = gradient declined the offer, and
  the differentiation problem is *not* parametric — a strong, surprising
  result that would redirect effort to F5 (structure) and the task side.
- **Fail**: shuffle delta stays < 0.10 despite the parameters existing.
  Then per-cell capacity is not the binding constraint and the mean-field
  behaviour is caused by the task or the objective, not the architecture.

## Addendum (2026-07-31, before results): the constraint was never intended

Max, on reading the audit: the shared-rule/interchangeable-cells design was
a **misunderstanding**, not a commitment — likely "kernel update laws" read
somewhere in the project's history as "one shared rule for every cell."
PROJECT.md's "params in the rule / capacity in the lattice" framing inherited
it. Consequence: full per-cell parameterization is sanctioned design space,
not a rule-bend. F8's expression vectors are the *minimal* step into that
space; richer per-edge transforms, per-cell rules, and heterogeneous tissue
(F5/DMON-1-w4m) no longer need special justification. PROJECT.md's framing
should be revised with Max once F8's result is in.

## Why this is worth running before any pivot

It is the cheapest possible test of the load-bearing question behind the
whole substrate thesis — *can these cells differentiate at all* — and it was
never tested, because until now the architecture made it impossible.
