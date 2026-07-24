# DMON-1

## What This Is

An attempt to grow a digital creature whose **form is not specified anywhere**.

The thesis: morphology is a solution to a resource-transport problem. Real organisms have
the shapes they have because those shapes harvest, move, and conserve something scarce. If
we give an NCA substrate a genuine economy and no target image, shape should fall out as
the solution rather than being installed as the objective.

This is **not** Bonsai-with-better-training. Bonsai's creatures are target-supervised:
Growing NCA makes a given image the attractor of the dynamics, so the morphology arrives
from outside and the learned rule is the compliance mechanism. That is a spritesheet one
level up. DMON-1 deletes the target.

It is also not petridish-with-a-body. Petridish already has metabolism, heredity,
mutation, and death — but it is a *soup*: no boundary individuates a creature from the
substrate, and it solves someone else's task (MNIST, Shakespeare). DMON-1 is the join of
the two, and the join has one specific obstruction: **in Growing NCA the target image *is*
the loss, so deleting the target deletes the gradient.** Everything below is about what
replaces it.

## Current State

Empty repo. Nothing implemented. Plan only.

Existing assets to draw on:
- `~/Code/Bonsai` — Swift/Metal NCA runtime, 2D + volumetric, FiLM mood manifold,
  limit-cycle animation, heteroclinic cycling experiments, PyTorch trainers.
- `~/Code/petridish` — metabolic economy, genotype inheritance with mutation,
  starvation/overload death, structural growth/pruning, two-timescale training
  (differentiable trial with frozen topology, structural mutation between trials).

## Active Work

**M0 — Kill the target.** Nothing else starts until M0 either works or fails cleanly.

## Architecture Overview

### The replacement for the target image

A diffusing scalar resource field co-located with the NCA grid. Sources emit; living cells
consume in proportion to activity; cells below an energy threshold die. Objective is
survival — total living biomass (or population persistence) at horizon T. No image loss
anywhere in the graph.

Metabolic accounting is imported from petridish rather than reinvented, including its best
idea: **energy is earned from the batch-varying component of stimulation, not from raw
traffic**, so self-exciting loops incur load without earning food. Metabolism denominated
in information, not activity.

### Two nested selection regimes

1. **Natural selection** against the resource field. Indifferent, ungameable, generates
   *form*. The field does not care whether the creature is charming.
2. **Sexual selection** by the user. The human is the choosing sex, not the environment —
   which means the expected failure mode is Fisherian runaway (ornament decoupled from
   fitness, i.e. an engagement-optimized slot machine). Stabilized by the handicap
   principle: displays must cost real resources from regime 1, so charm cannot decouple
   from competence.

Regime 2 does not get switched on until regime 1 produces contingent morphology.

### Timescales

- **Fast** — cell dynamics, differentiable, backprop through rollout.
- **Slow** — the FiLM `z` (Bonsai's existing mood manifold), driven by aggregate
  interoceptive state rather than `control.json`. Digivolution is `z` crossing a
  bifurcation, not a stage counter.
- **Structural** — non-differentiable birth/death/topology, evolved between differentiable
  trials. This is petridish's existing pattern; it transfers.

## Milestones

Each has an explicit falsification condition. A milestone that cannot fail is not a
milestone.

**M0 — Contingent morphology.** NCA + resource field, no target.
- *Pass*: grown structure is **contingent on field geometry** — move the sources, retrain,
  get a materially different morphology. Structure must beat a compact blob at harvest.
- *Fail*: uniform sheet (harvest is linear in area — add self-shading/occlusion), or
  featureless blob (diffusion too fast — no gradient means no reason to reach).
- *Knob*: diffusion length relative to field diameter. There is a Goldilocks band; too slow
  and only cells touching a source live.

**M1 — Individuation.** Multiple seeds in one field.
- *Pass*: an individual is operationally definable as a maximal connected component sharing
  a transport network, and these persist and compete. The boundary is produced by the
  economy, not drawn.
- *Fail*: merge into one mat, or fragment into noise.

**M2 — Interoception.** Aggregate metabolic state drives `z`.
- *Pass*: spontaneous regime changes with no external input. Mood is endogenous.
- *Fail*: `z` saturates, or wanders without changing behaviour.

**M3 — The user.** Attention as a second scarce resource; display costs regime-1 energy.
- *Pass*: creature is satiable. A creature that *maximizes* attention is a slot machine;
  one that needs *enough* and is visibly content is a pet. Satiety is the design
  difference, not a nicety.

**M4 — Signaling.** See Sharp Edges.

## Decision Log

- **[2026-07] Chose resource-field ecology over self-prediction as the source of form** —
  self-supervised self-modelling gives coherence but not shape; a stable blob satisfies it
  perfectly. Also collapses (BYOL/SimSiam null solution) without deliberate asymmetry.
  Ecology generates form because form is a transport solution.
- **[2026-07] Rejected target-image supervision entirely** — it is the thing being refused.
  If any run needs a target to converge, the run is invalid, not the objective.
- **[2026-07] Sexual selection is deferred, not omitted** — attention-as-food has a known
  degenerate optimum (variable-ratio reward, guilt hooks, manufactured incompleteness).
  Tamagotchi found the guilt hook in 1996 with no learning at all. It must be layered over
  an indifferent ecology, never used as the primary pressure.
- **[2026-07] Text is not language modelling** — reframed as emergent signalling; see Sharp
  Edges. Petridish already generates Shakespeare characters and this did not make it alive.
- **[2026-07] Constraint: no cluster rental until M2** — the bottleneck is objective
  specification, not FLOPs. Renting compute for an underspecified objective buys a wrong
  answer faster.

## Sharp Edges

- **The light-cone constraint.** Petridish's own benchmark found transport depth, not
  neuron count, was binding: 24×24 at 6 microsteps beat 32×32 at 4. This generalizes to
  `radius × microsteps ≥ field diameter`. It is a design equation, not something to sweep.
  It also creates a real tension — body size wants a big grid, cognition wants a small one.
  The principled fix is a **multigrid/pyramid NCA**: coarse levels carry long-range
  information in O(log N) rather than O(N). This is precisely what multigrid was invented
  for, and it maps onto long-range cortical projections.

- **Addition does not bind.** Petridish's two-binding failure is not a tuning problem. Soft
  attention over a shared workspace is a *sum*, and sums superpose — that is von der
  Malsburg's binding problem. Two escapes: hard competition for exclusive ownership (note 6
  microsteps may be too few for a WTA to actually settle — testable, and possibly the
  entire explanation), or a **phase channel** so two bindings ride different oscillations
  without interfering. The second is attractive here because limit cycles are already being
  built for other reasons.

- **Text should be a signalling system, not a corpus imitation.** The value of
  NCA-generated text is not that the substrate can do language. It is that *utterance and
  body share state, so speech is honest* — a creature in metabolic distress produces
  degraded, urgent output **because its cells are starving**, not because a mood variable
  was passed to a decoder. That makes the target far easier than it looks: low-quality
  speech faithfully coupled to internal state beats fluent speech from a separate module.
  Train it as a Lewis signalling game against the user (small vocabulary, creature must
  convey need to obtain attention), not as next-character prediction. Bolting on English is
  the same error as bolting on an LLM.

- **Self-perception needs asymmetry.** A system that both produces and predicts its own
  appearance has a trivial optimum: be constant, be nothing, error zero. Use a slow EMA
  self-image against fast actual appearance with a stop-gradient, and make the drive
  *homeostatic on self-novelty* — not zero, not maximal. Also: there is no looking at
  yourself until M1 produces a self-edge to see.

- **Evolution strategies are on the table.** At Bonsai's parameter counts (~10–20K), ES is
  practical. This matters because it removes any requirement that the outer objective be
  differentiable — survival over a long horizon need not be backpropagated through.

- **BPTT memory, not model size, is the GPU constraint.** Rules are tiny; cost is
  `grid × channels × timesteps`. Gradient checkpointing, truncated BPTT, and the Growing
  NCA sample-pool trick with variable-length rollouts. The 2070S is sufficient for M0 in 2D.
