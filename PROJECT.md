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

M0 substrate and trainer implemented (`dmon/substrate.py`, `dmon/train_m0.py`),
~420 lines, ~11K parameters. Runs on CPU. A 300-iteration smoke run on a 24x24 grid
took mass 1 -> 23 with box dimension near 1.54 — this establishes that the gradient
path is intact and the survival objective is learnable, and **nothing else**. No run
at scale. No contingency test performed. The central hypothesis is untested.

Now a real package (`dmon/`), with checkpointing (`dmon/checkpoint.py`), diagnostic
rendering (`dmon/render.py`), and the legibility probe (`dmon/probe.py`). Full chain
runs end to end on CPU. Compute is `m@192.168.0.202` (host `Aine`): RTX 4090 24GB +
RTX 2070S 8GB, torch 2.11/cu130.

Blocking issues, both in `contingency()`:
1. It reports between-geometry spread but never measures the within-geometry baseline
   that spread must exceed, so as written the test cannot fail. The baseline must come
   from independent *training* runs (`--seed`), not repeated evaluation — `seed()` is
   deterministic, so re-evaluating one rule samples only firing jitter.
2. Cross-evaluation results are computed and then discarded; `spread` uses only the
   `self` entries. That is the half that catches a memoriser.

Fix both before any run is interpreted. See `HANDOFF.md`.

Existing assets to draw on:
- `~/Code/Bonsai` — Swift/Metal NCA runtime, 2D + volumetric, FiLM mood manifold,
  limit-cycle animation, heteroclinic cycling experiments, PyTorch trainers.
- `~/Code/petridish` — metabolic economy, genotype inheritance with mutation,
  starvation/overload death, structural growth/pruning, two-timescale training
  (differentiable trial with frozen topology, structural mutation between trials).

## Active Work

**M0 — Kill the target.** Nothing else starts until M0 either works or fails cleanly.

Immediate order of operations (detail in `HANDOFF.md`):
1. Sweep diffusion length relative to grid diameter — this decides whether M0 can work.
2. One real training run on the 4090: 64x64, >=64 steps, batch 32, 20k+ iters.
3. **Legibility probe** on that single rule (`ARCHITECTURE.md` §M2). Needs one rule,
   not twenty, so it runs first in wall-clock despite being conceptually later. It is
   the cheapest falsification in the project and sits under M2, M3 and M4 at once.
4. Make the contingency test capable of failing (training-seed baseline + actually use
   the cross-evaluation), then the full 20-run sweep.

Deeper reasoning lives in `ARCHITECTURE.md`.

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

- **[2026-07] Rule may read energy, may not write it** — channel 0 masked out of the
  update. Otherwise the creature prints its own currency and the economy is decorative.
- **[2026-07] Rule gets two metabolic levers, no direct control of shape** — uptake
  effort and transport conductance. Morphology is a consequence of differential
  investment. This is the central bet; if M0 fails this is the likely reason, and the
  response is a third lever, not abandoning the ecology.
- **[2026-07] Tried hidden channels with no path to the ledger** — gradient was exactly
  zero. The rule could compute anything and never touch survival. Any new channel needs
  a path to the ledger or it is decoration.
- **[2026-07] Neither Bonsai nor petridish is a viable fork base** — Bonsai's trainers
  are all target-shaped; petridish's physical space is an address space for a
  computation graph, not a body. M0 written fresh so it stays cheap to throw away.

- **[2026-07] Body and display are separate organs; the no-target rule is scoped to
  the body** — the old formulation ("form is not specified anywhere") contradicts the
  actual end state, which is a creature a human wants to look at. Restated: *no target
  on the body, targets permitted on the display, display paid for out of regime 1.*
  Stricter than a blanket ban, because it names why the body is protected — its shape
  is a claim about the world, and specifying it makes the claim fraudulent. A display
  makes no such claim. See `ARCHITECTURE.md` §M3.
- **[2026-07] The display is structurally an M3 object and cannot be prototyped early**
  — it is deliberately not load-bearing for survival, so before an attention field
  exists it has no path to the ledger and the gradient is exactly zero. This is the
  bug that already happened once, not a hypothesis.
- **[2026-07] Morphology drifts over a continuous morphospace; it is never selected
  from a menu** — a creature choosing among trained forms is a selector over a library
  someone else authored, i.e. the spritesheet one level further up. Looking like a
  chibi must be an outcome, not a decision.
- **[2026-07] Display cost is M3's Goldilocks knob** — same status as `field_diffusion`
  at M0. Too cheap, ornament decouples (Fisherian runaway); too expensive, selection
  deletes the display. Sweep it first.
- **[2026-07] The weeks timescale is enforced by information rate, not a
  hyperparameter** — attention-derived reward from one human is a few hundred bits a
  week, which cannot move 11K parameters quickly at any learning rate. The pet/feed
  distinction is therefore structural. Strongest case yet for ES at the outer loop:
  there is no differentiable path from a human's regard to a cell rule.
- **[2026-07] Rejected "objective increases with usage"** — that is the operator's
  objective wearing the creature's face. What makes an animal compelling is having an
  agenda that isn't you; the wanting is load-bearing because it is intermittent. The
  failure mode to design against is not "wants attention" but "has nothing else."

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
