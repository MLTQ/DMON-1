# DMON-1 — Architecture

This document carries the *reasoning*. `PROJECT.md` is the session briefing;
`HANDOFF.md` is what to do first. Read this one when you need to know why something
is shaped the way it is, or when you are about to propose changing it.

---

## 1. The problem

Build a digital creature whose form is not specified anywhere.

The failure mode being avoided is not "hand-drawn sprites." It is one level subtler.
Growing Neural Cellular Automata trains a per-cell rule such that a **target image is
the attractor of the dynamics**. The result is genuinely dynamic — it grows from a
seed, it regenerates from damage, its animation is a limit cycle of its own physics —
and the morphology still arrives entirely from outside. The learned rule is the
compliance mechanism. That is a spritesheet at one remove, and it is what `~/Code/Bonsai`
currently is.

The complementary failure is `~/Code/petridish`: a genuine developmental economy
(metabolism, heredity with mutation, starvation and overload death, structural growth
and pruning) with no body and no individual. Its "physical space" is an address space
for a computation graph. It is a soup solving someone else's task.

DMON-1 is the join. The join has exactly one obstruction:

> **In Growing NCA the target image *is* the loss. Delete the target and you delete
> the gradient.**

Everything below is an answer to: what replaces it?

---

## 2. The core substitution

**Morphology is a solution to a resource-transport problem.**

Organisms have the shapes they have because those shapes harvest, move, and conserve
something scarce. Coral, lichen, lungs, vasculature, root systems — all are
surface-area or transport geometries. None of them was specified; all of them are
what the physics made cheap.

So: put a diffusing scalar resource field on the grid. Sources emit. Living cells
consume. Cells below an energy threshold die. The objective is sustained living mass.
No image loss anywhere in the graph.

Two mechanisms do the work, and both are already physics rather than objective:

1. **Diffusion-limited uptake rewards surface area.** A dense clump locally depletes
   the field faster than diffusion replenishes it, so the interior starves and only
   the boundary feeds. You do not need to add self-shading; finite diffusion rate
   gives it to you. This is why biofilms and corals grow the way they do.
2. **Intra-body transport makes reach expensive.** Energy moves through the body by
   gated diffusion, so a cell far from a source is fed by a chain of cells that each
   pay maintenance. Distance costs. That is the pressure that produces branching
   rather than sprawl.

### What the rule can and cannot do

The rule reads energy and **cannot write it** (channel 0 is masked out of the update).
Without this the creature prints its own currency and the economy is decorative.

The rule acts through exactly two levers, both metabolic:

| Channel | Lever | Meaning |
|---|---|---|
| 1 | uptake effort | how hard this cell works to absorb resource |
| 2 | transport conductance | how freely it passes energy to neighbours |

It has **no direct control over shape**. Morphology is a consequence of where it
chooses to invest: cells that do not fund themselves starve out, and the body extends
only where transport pushes a halo cell over the death threshold.

This is the central bet of the project. If M0 fails, this is the most likely reason,
and the response is to add a third lever — not to abandon the ecology.

---

## 3. Milestone ladder

Each milestone has an explicit falsification condition. **A milestone that cannot
fail is not a milestone.** This is the single most important discipline in the
project, because everything here is easy to fake and pretty to look at.

### M0 — Contingent morphology  *(code exists, untested at scale)*

NCA + resource field, no target.

- **Pass**: morphology is contingent on field geometry. Train separate rules under
  different source layouts; the descriptor vectors must separate by more than
  within-geometry seed noise. Additionally, cross-evaluation must show that a rule
  trained on geometry A produces a *different* body when dropped into geometry B.
- **Fail (blob)**: featureless clump. Diffusion too fast — no gradient, no reason to
  reach. Lower `field_diffusion`.
- **Fail (sheet)**: uniform mat. Harvest is effectively linear in area. Raise
  `maintenance` or lower `uptake_rate`.
- **Fail (dead)**: everything starves. Seed nearer a source; consider a curriculum
  that starts sources adjacent to the seed and walks them outward.
- **Fail (memorised)**: *the dangerous one.* Rule learns a fixed shape and ignores the
  field. Looks like success. Only cross-evaluation catches it.

The key knob is **diffusion length relative to field diameter**. There is a Goldilocks
band; too slow and only cells touching a source survive, too fast and the field is
uniform. Expect to sweep this before anything else.

**Record the morphospace, not just the verdict.** The contingency run already produces
descriptor vectors for every (rule, geometry, seed) triple. Keep the whole point cloud
and measure its dimensionality, because that number is what the entire bridge to M3
rests on. A 4-D descriptor cloud that collapses onto a 1-D curve means the ecology
admits essentially one morphology with a single axis of variation — the body is not
steerable and the display has to carry all the expressivity. A cloud genuinely
spanning 3–4 dimensions means the body itself has room to be selected over. This costs
one array and a singular-value decomposition on top of a run that has to happen
anyway, and nothing downstream can be planned without it.

### M1 — Individuation

Multiple seeds in one field. The question is where a *boundary* comes from.

The proposed answer: **individuality is shared metabolism.** An individual is a
connected component of the *conductance graph* — cells that pass each other energy —
not of the occupancy mask. This matches the biological question (what distinguishes a
colony from an organism is exactly whether resources are shared).

To get a boundary rather than a mat, conductance needs something to discriminate on.
Proposal: a heritable per-cell tag (petridish already has genotype inheritance with
Gaussian mutation — lift it), with conductance modulated by tag similarity between
neighbours. Sharing energy with non-kin is exploitable; restricting it is selected
for; the membrane is where tag similarity drops.

- **Pass**: stable multi-individual populations; components persist, compete, and
  have identifiable boundaries not drawn by us.
- **Fail**: merge into one mat (conductance everywhere), or fragment into dust
  (conductance nowhere).

Note this is a prerequisite for self-perception: there is no looking at yourself until
there is a self-edge to see.

### M2 — Interoception

Aggregate metabolic state drives the FiLM `z` that modulates the rule. Bonsai already
has the manifold machinery; it is currently written by `control.json` from outside
(autopilot, CLI, UMAP panel, an agent's mood). Drive it from inside instead.

Interoceptive signal candidates: mean energy, energy variance, mass, mass derivative,
uptake rate, fraction of body in deficit, distance to nearest resource.

**Digivolution is a bifurcation.** Fast timescale = cell dynamics. Slow = `z`. When a
slow parameter crosses a bifurcation, the fast attractor set reorganises. That is
metamorphosis with a mathematical definition instead of a stage counter, and
Izhikevich's taxonomy of bursters is precisely a classification of slow-fast
bifurcation structures — there is a principled taxonomy of digivolutions available.

**Testing this is concrete, not vibes**: sweep `z` slowly along a path, run fast
dynamics to steady state from several initial conditions at each point, record
descriptors. Look for discontinuities, and specifically for **hysteresis** — the
signature of a saddle-node. If sweeping `z` up and back down traces different curves,
you have a real bifurcation.

- **Pass**: spontaneous regime changes with no external input; measurable hysteresis.
- **Fail**: `z` saturates at an extreme, or wanders without changing behaviour.

#### The legibility probe — runs at M0, tests M2/M3/M4

Everything above M2 rests on one unexamined assumption: that **internal metabolic
state has a visible signature.** M2 wants mood readable from the outside; M3's
handicap wants a display that degrades when the body is failing; M4 wants speech that
goes urgent because cells are actually starving. All three are the same claim, and
none of them has been tested.

It can be tested now, on M0 output, with no new architecture and no training. Take a
trained rule, run it to steady state, then starve it mid-rollout — cut `source_rate`,
or translate the sources away — and record descriptors against mean energy across the
transition.

- **Pass**: shape tracks metabolic state legibly and gradually. Then a display grown on
  the same substrate inherits that coupling *for free*, and honesty is a property of
  the physics rather than something written in.
- **Fail**: morphology is invariant to energy right up until the creature abruptly
  dies. Then there is no signature to read, and any display that appears to show
  distress would have to be coupled artificially — which is precisely the mood-vector-
  to-decoder architecture rejected in §M4.

This is the cheapest falsification available anywhere in the project and it sits under
three milestones at once. It needs **one** trained rule, not the twenty that the
contingency test needs, so in wall-clock it can run first even though it is
conceptually later.

### M3 — The user

Attention as a second scarce resource. This is where the creature meets a human.

**The regime changes here and that is the whole risk.** Against the resource field the
creature faces natural selection: the environment is indifferent and ungameable, so
the only route to more energy is to genuinely solve the transport problem. Against a
human it faces **sexual selection** — the human is the choosing sex, not the
environment. The prediction from evolutionary biology is Fisherian runaway: ornament
decoupled from fitness. The engineering translation is that an attention-maximiser
converges on variable-ratio reward, manufactured incompleteness, and guilt hooks.
Tamagotchi found the guilt hook in 1996 with no learning at all; a system that can
search will find it in an afternoon.

Two stabilisers, both required:

1. **Costly signalling (Zahavi's handicap principle).** Display must consume real
   regime-1 energy. Then attractiveness is a proxy for competence in the indifferent
   ecology and cannot be faked cheaply.
2. **Satiety.** The creature needs *enough* attention, not maximal attention. A
   maximiser has no attractor — it just runs. A satiable creature has a setpoint and
   is visibly content when met. **This is the difference between a pet and a slot
   machine and it is a one-line change to the objective.**

M3 does not start until M0 produces contingent morphology. Sexual selection layered
over nothing is just an engagement optimiser.

#### The body/display split

There is an apparent contradiction between "form is not specified anywhere" and the
actual end state, which is a creature that looks like something a human wants to look
at — a chibi, a shoggoth, a moss spirit. Chibi-ness is a target. It is imposed from
outside, by a human, for aesthetic reasons, and it solves no transport problem.

The contradiction is real and the resolution is anatomical: **give it a body and a
display, and do not make them the same organ.**

> **No target on the body. Targets are permitted on the display. The display is paid
> for out of regime 1.**

This is stricter than a blanket "no targets," not looser, because it says *why* the
body is protected. The body's shape is a **claim about the world** — this is what the
transport problem made cheap. Specify it and the claim is fraudulent, which is exactly
the Growing-NCA failure this project exists to escape. The display makes no such
claim. It is a signal, and signals are permitted to be about the receiver. What keeps
it honest is that it costs.

Three consequences, all mechanical:

- **The display cannot be built before the attention field exists.** A display channel
  is deliberately not load-bearing for survival — which means no path to the regime-1
  ledger, which means *exactly zero gradient*. That is not a hypothetical; it is the
  bug that already happened once and produced the two-lever design (see §2 and
  `substrate.md`). Display channels only become non-decorative once there is a second
  field they earn from. The display is structurally an M3 object and cannot be
  prototyped early "just to see how it looks."

- **The two organs need different morphospaces, and expressivity is the reason —
  honesty is only the second reason.** If display appearance is driven by the body's
  own shape parameters, then preference-gradient ascent runs in coordinates that the
  transport problem defines, and *cute has no coordinate there*. You would be pushing
  in a direction the substrate cannot represent. The display needs its own dimensions,
  constrained only by cost. Likely implementation: the same cell machinery, a second
  field growing on the body, a different ledger — it spends where the body earns.

- **Display cost is M3's `field_diffusion`.** The handicap works only in a band. Too
  cheap and ornament decouples from fitness (Fisherian runaway). Too expensive and
  displaying kills you, so selection deletes the display and you are back to coral.
  This knob has the same status as diffusion length at M0: it decides whether the
  milestone can work at all. Sweep it first, not last.

#### Morphology drifts; it is not selected from a menu

A creature that *picks* a form — chibi, shoggoth, moss spirit — from a set of trained
NCAs is a selector over a library someone else authored. That is the spritesheet again,
one level further up. The admissible version is selection over a **continuous
morphospace**: the creature does not choose chibi, it drifts toward chibi-ness because
chibi-adjacent displays got fed. Looking like a chibi is an outcome, not a decision.

#### The three timescales

- **Seconds** — cell dynamics, posture, display responding to whether the user is
  present. Fast, cheap, reversible.
- **Minutes to hours** — the `z` manifold moving under interoceptive state. Mood. This
  is M2, and it is where "it seems different today" lives.
- **Weeks** — slow drift of rule parameters under attention-derived reward. This is
  where morphology actually changes, and it must be slow. A creature that morphs to
  please you within a session is a slot machine; one that has visibly grown into
  something over two months is a pet you have a history with.

**The slow timescale is enforced by information rate, not by a hyperparameter.**
Attention-derived reward from one human is a few hundred bits a week, optimistically.
That will not move 11K parameters quickly at any learning rate. The property that
makes this a pet rather than a feed is therefore structural rather than a setting
someone can turn up in a moment of impatience. It is also the clearest case for
evolution strategies at the outer loop (§7): there is no differentiable path from a
human's regard to a cell rule, and at this parameter count there does not need to be.

#### The agenda that isn't you

"The creature should want the user to use it" is the right goal and the wrong
objective. An objective monotonic in usage has one attractor and it is not a creature.

The stronger argument is not about degenerate optima. **What makes an animal
compelling is that it has an agenda that is not you.** A pet that always wants
attention is exhausting. A cat that mostly conducts its own inscrutable business and
occasionally decides you are worth sitting on is the thing people organise their lives
around. The wanting is load-bearing *because* it is intermittent, and because there is
a whole creature there in the gaps.

Regime 1 is already the mechanism for this. When fed, the creature is busy in its own
world — solving transport, growing, maintaining — and the user is watching a thing
live rather than a thing solicit. The failure mode is not "the creature wants
attention." It is "the creature has nothing else."

Note also whose objective is whose: a creature trained to want usage is the operator's
objective wearing the creature's face. A creature that survives and happens to need
attention has its own. Users feel that difference without being able to name it.

- **Pass (satiety)**: with the attention field saturated, the creature keeps doing
  interesting things in its own ecology.
- **Fail**: it idles waiting for input. The internal world is not rich enough yet and
  M3 was premature.
- **Pass (honesty)**: see the legibility probe below — starve it, and the display
  degrades with no code path anywhere that says "when starving, look worse."

### M4 — Signalling, not language modelling

The intuition that "text generation from an NCA would unlock this" is half right, and
the wrong half is expensive.

Petridish **already generates text** — Tiny Shakespeare, character level, 68×68 field,
working. It did not make petridish any more alive than MNIST did. Language as a
*capability* is not the unlock.

What is load-bearing: **if utterance and body share state, speech is honest.** A
creature that goes terse and urgent because its cells are actually starving is
categorically different from one where a mood vector was passed to a decoder, and the
difference is immediately legible to a human in a way fluency does not substitute for.

That inverts the difficulty. The target is not good language. It is **bad language
that cannot lie.** Three words, faithfully coupled, beats a paragraph of Shakespeare
from a bolted-on module.

Concrete setup — a Lewis signalling game, not next-token prediction:

- Small vocabulary (8–32 symbols), no corpus.
- Creature emits symbols from a boundary readout of the same substrate.
- A user policy observes symbols and acts (e.g. places a resource source).
- The creature survives better when the action matches its need. Symbols acquire
  meaning because acting on them changes survival.
- Train against a *scripted user policy* first, co-adapting. Then swap in a human.
- **Risk**: with a single scripted partner the pair develops a private degenerate
  code that does not survive contact with a human. Mitigate by training against a
  population of user policies and adding channel noise.

Bolting on English is the same error as bolting on an LLM — less obvious only because
English feels like a substrate rather than a module.

---

## 4. Cross-cutting constraints

### The light-cone constraint

Perception radius is 1, so information travels one cell per step. Therefore:

```
radius × microsteps ≥ field diameter
```

Petridish's own benchmark independently discovered this: 24×24 at 6 microsteps beat
32×32 at 4, because transport depth — not neuron count — was binding. This is a
**design equation, not something to sweep**, and violating it is the single most
likely cause of a mysteriously featureless result.

It also creates a real tension: body size wants a big grid, cognition wants a small
one. The principled resolution is a **multigrid / pyramid NCA** — coarse levels carry
long-range information in O(log N) rather than O(N). This is exactly what multigrid
was invented for, and it maps onto long-range cortical projections. Expect to need it
by M2.

### The ecology has to be able to support a body

Three design equations, none of which were written down before the first sweep, and all
three of which the default configuration violated. They are checkable in seconds with no
training, and they should be checked before any GPU time is spent — the first diffusion
sweep burned 40 minutes sweeping a knob that could not matter, in a regime where nothing
could have lived.

**1. Supply must exceed demand.** A cell at full uptake effort costs
`maintenance + effort_cost` per step. A source emits `source_rate` per step. So

```
N_max  ≈  (total source emission) / (maintenance + effort_cost)
```

is a hard ceiling on sustainable body size, before any losses to diffusion, decay or
imperfect capture. At the original defaults — one source at 0.06, costs at 0.014 — that
ceiling is **about four cells**. No morphology of any kind is available at four cells.
A body of a few hundred cells needs total emission two to three orders of magnitude
higher, and because a single cell saturates at `field_cap`, that means *more source
cells*, not a bigger number on one of them.

**2. The field has its own light cone, and it is slower than the creature's.** §4's
light-cone constraint governs *information*, which moves ballistically at one cell per
step. Resource moves *diffusively*:

```
d  ≤  2 √(D · T)
```

Food more than `2√(D·T)` from the body simply does not arrive within a rollout of length
`T`. At `D = 0.4, T = 64` that is about 10 cells. A seed at the centre of a 64×64 grid is
30 cells from a west-edge source, so under those settings the food may as well not exist
— and no value of `D` fixes it, because `D` is bounded above by numerical stability.

**3. Concentration must exceed break-even where the body is.**

```
r*  =  (maintenance + effort_cost) / uptake_rate
```

A cell sitting in a field below `r*` loses energy by trying to eat. At the original
defaults `r* = 0.04`, and the field never exceeded `0.019` anywhere the creature could
reach.

The three interact: raising `source_rate` past the point where the source cell saturates
at `field_cap` does nothing at all, which is why the first parameter scan showed
identical reach for emission rates of 0.5 and 2.0.

### Addition does not bind

Petridish's two-binding retrieval failure is not a tuning problem. Soft attention over
a shared workspace is a **sum**, and sums superpose — that is von der Malsburg's
binding problem, and it is why both the broadcast workspace and the fast-weight matrix
plateaued at the "remember one of two values" strategy.

Two escapes:

1. **Hard competition for exclusive ownership.** Note that 6 microsteps may be too few
   for a winner-take-all to actually settle — this is testable and might be the entire
   explanation for the failure. Test it before designing anything new.
2. **A phase channel**, so two bindings ride different oscillations without
   interfering. Attractive here because limit cycles are already being built for other
   reasons.

### Self-perception needs asymmetry

A system that both produces and predicts its own appearance has a trivial optimum: be
constant, be nothing, error zero. This is BYOL/SimSiam collapse and the fixes are all
asymmetry — stop-gradient, EMA teacher, predictor on one branch only.

The non-collapsed formulation: a slow EMA self-image against fast actual appearance,
with the drive **homeostatic on self-novelty** — not zero, not maximal. Same shape as
the metabolism, in the same currency.

Note that self-prediction gives *coherence*, not *form*. A stable blob satisfies it
perfectly. It is a good identity mechanism and a bad morphogen; this is why the
ecology, not self-modelling, is the source of shape.

---

## 5. Where the heteroclinic story actually stands

Stated honestly, because this is the least settled part of the design.

The appeal is real: heteroclinic channels give metastable states with rapid switching,
where the *sequence* carries the information rather than the fixed points, and where
input selects among the outgoing branches at each saddle (Rabinovich's winnerless
competition). That is a good model of a behavioural repertoire, and it is legible —
dwell-then-switch is exactly what makes animal behaviour readable as an ethogram.
It also gives spontaneous behaviour for free: such a system does not need input to
move, so a creature has idle dynamics and the user steers rather than triggers.

**The unsolved part**: heteroclinic dynamics are normally *designed in* — you choose
the connection matrix, usually via asymmetric inhibition among competing modes.
Getting them to *emerge* from a metabolic substrate is itself an open problem, and I
do not have a route to it. Treat "the creature will develop heteroclinic structure" as
a hypothesis to be tested late, not a design step to be implemented.

The nearest thing to a concrete path: M2's bifurcation sweep will reveal what
attractor structure the substrate actually produces. Look at what is there before
deciding what to impose.

---

## 6. Observability

HBPE already contains the right instrument, applied to the wrong subject. Its
three-tier token hierarchy with living vocabularies discovered by HDBSCAN/BERTopic is
a *discovered* behavioural vocabulary rather than a hand-written one. Run it over the
creature's own state trajectory instead of over user activity:

- Discovered clusters = the metastable states.
- Dwell-then-switch = a token.
- L0/L1/L2 tiers = timescale separation.

This yields an ethogram nobody wrote, which is the standard blocker in open-endedness
work — the behaviour characterisation has to evolve too, or the search collapses to
whatever you thought a creature was on day one.

---

## 7. Compute

Models are tiny (~11K parameters at M0). **The GPU constraint is BPTT rollout memory,
not model size**: cost scales as `grid × channels × timesteps`. Reach for gradient
checkpointing, truncated BPTT, and the Growing-NCA sample pool before reaching for a
smaller grid.

Evolution strategies are viable at this parameter count, which matters because it
removes any requirement that the outer objective be differentiable — long-horizon
survival need not be backpropagated through. Petridish's two-timescale pattern
(differentiable trial with frozen topology, structural mutation between trials) is the
right template and it transfers directly.

**Do not rent a cluster before M2.** The bottleneck is objective specification, not
FLOPs. Renting compute for an underspecified objective buys a wrong answer faster and
attaches a sunk cost to it. The 4090 covers everything through M1; the 2070S covers
M0 in 2D.
