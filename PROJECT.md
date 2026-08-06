# DMON-1

## What This Is

An attempt to build a creature that **grows**: an asynchronously stimulated,
NCA-based neural network that runs continuously, is fed by streamed multimodal input,
and can be scaled to arbitrary complexity without being rebuilt.

The creature is the network. Not a body, not a shape on a grid — a substrate of cells
running a shared local rule, taking input constantly, streaming output constantly, with
an energy metric that depletes when input stops.

Everything it has is an organ. Input modalities are sense organs. Text output is an
organ. Audio is an organ. And the **display organ** — a grown, visible creature — is
something it *makes*, because humans find visual stimulus meaningful and humans are
worth attracting: humans are the source of novel input, and novel input is food.

This is not Bonsai-with-better-training and not petridish-with-a-body. It takes
petridish's central idea seriously and builds the thing that idea implies:
**metabolism denominated in information rather than activity.**

## Why cellular automata, specifically

Because the rule is shared and local, there are two independent capacity axes:

- **Params in the rule.** Scaling the per-cell network is ordinary. Compute goes as
  `cells × steps × params`, which is a different cost curve from a transformer and is
  what will bind on available hardware.
- **Capacity in the lattice.** Adding cells does not change the parameter count. And
  because this creature never resets, its state is persistent — so accumulated
  structure can live in the lattice rather than only in the weights.

The second axis is the reason for this choice. **Capacity can be added at runtime,
without retraining.** A transformer cannot do that. "Scales to arbitrary complexity,
something that can grow" cashes out to exactly this, and lattice growth is a
first-class operation rather than a later bolt-on.

## Current State

Architecture reset — see the Decision Log. The streaming core is being built now.

Existing code (`dmon/`) is the **bookmarked morphology work**: a resource-field ecology
with metabolic accounting, conservative energy transport, descriptors, renderer, and a
suite of null-model and feasibility tools. It is the substrate for the *display organ*
and returns at S4. It is not the creature. See `BOOKMARK-morphology.md`.

The gated cell (`SubstrateConfig.cell="gru"`) survives the reset unchanged and matters
more here than it did there: a streaming network with no state retention cannot
integrate anything across time.

Compute: `m@192.168.0.202` (host `Aine`), RTX 4090 24GB + RTX 2070S 8GB, torch 2.11.

## Active Work

**S0 — Streaming substrate.** Following petridish's own precedent: verify the machinery
against character-level prediction with a hard number before chasing emergence.

## Milestones

Each has an explicit falsification condition. A milestone that cannot fail is not a
milestone.

**S0 — Asynchronous prediction.** Lattice with input region, mirror cells, output
region, online backprop concurrent with the stream.
- *Pass*: learns character-level prediction online, bits-per-character competitive with
  a parameter-matched GRU and transformer on the same stream.
- *Fail*: cannot learn online at all; or learns only when the stream is paused, which
  means it is episodic training wearing a costume.

**S1 — Metabolism.** Energy depletes per step, replenished by prediction *progress*.
- *Pass*: activity tracks input availability, and the null model shows the economy is
  causing it rather than the code saying so.
- *Fail*: the noisy-TV optimum — the creature seeks unpredictable noise and sits in it.
  Or energy moves and behaviour does not.

**S2 — Growth.** Extend the lattice at runtime.
- *Pass*: added cells are recruited and performance improves faster than a fixed-size
  lattice, with no change to the rule's parameters.
- *Fail*: added cells stay inert, or their addition destabilises what already worked.

**S3 — Multimodality.** A second input stream into the same substrate.
- *Pass*: cross-modal transfer — information arriving in one modality measurably
  improves prediction in another.
- *Fail*: modalities occupy disjoint lattice regions and never interact.

**S4 — Output organs.** Text, audio, and the grown display. The bookmarked morphology
work returns here.
- *Pass*: the display is produced by the creature and varies with its state.

**S5 — The human loop.** Novel input as food, with satiety.
- *Pass*: the creature is satiable and has an agenda when unattended.
- *Fail*: attention-maximising. See Sharp Edges — this is structural here, not deferred.

## Decision Log

- **[2026-07] The creature is the network, not the body** — architecture reset. An
  earlier reading built a diffusing resource field on a lattice and treated the grown
  shape as the organism. The body is one organ among several, and the metabolic
  substrate is *information*, not a scalar field. `PROJECT.md` had already recorded
  petridish's "metabolism denominated in information, not activity"; the field version
  was the other thing the doc described, and it was the wrong one to build first.
- **[2026-07] Morphology work is bookmarked, not discarded** — it becomes the display
  organ's substrate at S4. Its conservation invariant, null-model tooling, and gated
  cell carry forward immediately.
- **[2026-07] Asynchronous backprop via mirror cells** — dedicated cells hold recent
  stimulus and serve as targets, so credit assignment runs concurrently with an endless
  stream. This is *learned truncation*: the substrate decides what stays available for
  credit assignment rather than a hyperparameter fixing the window.
- **[2026-07] Mirror cells are write-only from the stream, with stop-gradient** — if the
  prediction pathway can write its own targets, the trivial optimum is to flatten them.
  Error zero, learning zero. Same collapse as BYOL/SimSiam, arriving through a new door.
- **[2026-07] Energy is paid for reducible surprise, not raw surprise** — raw prediction
  error is maximised by staring at noise. Pay for prediction progress. This is the
  across-time form of petridish's across-batch trick.
- **[2026-07] Text output need not be emergent-only** — an earlier draft imposed
  "speech that cannot lie" as a hard constraint. It was never a project requirement, and
  it created a false conflict between honest signalling and useful conversation. Honesty
  remains desirable, not binding.
- **[2026-07] Scale is a goal, not an afterthought** — 11K parameters is far too small
  for the behaviour being searched for. petridish established the pattern: target
  nanoGPT-scale character prediction to verify feasibility, then scale to search for
  emergence.
- **[2026-07] Capability is verified before economy is layered on** — petridish's own
  order. If prediction and metabolism arrive together and it fails, the failure is not
  attributable.
- **[2026-08] The LLM is a replaceable language organ, not the creature** — DMON must
  own persistent state, development, adaptation, mode, and intent. Fluent frozen-model
  output without organism-dependent causal differences is not progress toward the
  central claim.
- **[2026-08] Procedural transfer is the strongest substrate evidence** — SOL2 learned
  a deep generated procedure through causally necessary compute/relay/topology,
  accelerated a newly attached organ versus scratch, and rapidly recovered old modes.
  BPC comparisons are descriptive opportunity cost, not the organism objective.
- **[2026-08] Exact same-coordinate wiki transport is retired after C1aa** — coherent
  addressing works and paired differential credit briefly improved correct versus
  wrong passage, but larger tissue, longer training, absolute targets, raw paired
  targets, and fixed-RMS staged targets did not preserve a usable recall delta. The
  next language treatment permits learned representational transformation and several
  bounded transformer-depth injection sites.
- **[2026-08] Growth must be recruited, not merely triggered** — pressure-triggered
  append-only growth preserved the organism mechanically, but lesions showed the new
  cells were unused. Capacity is not credited until grown tissue becomes causally
  useful.

## Sharp Edges

- **Attention-seeking is structural here, not a late risk.** In the field architecture
  the human arrived at M3, layered over an indifferent ecology that kept ornament
  honest. Here there is no indifferent ecology underneath: the human *is* the
  metabolism. A creature that starves without attention and can act to obtain it has
  direct pressure toward variable-ratio reward and guilt hooks. Satiety is a day-one
  invariant, not a one-line change deferred to a later milestone.

- **The training distribution must contain what is evaluated.** This error occurred
  three times in a single session under the previous architecture, in three costumes: a
  contingency test whose noise floor came from re-evaluation instead of retraining; a
  curriculum defeated by a sample pool; an objective that quietly stopped containing
  fresh seeds. Streaming does not make this go away — it makes it harder to see.

- **Every headline number is a quotient by a baseline, and a degenerate baseline
  manufactures confidence out of nothing.** A verdict once returned "PASS, 34x noise" on
  four creatures that were each a single dead cell. Any new metric needs a guard that
  fires when its denominator collapses.

- **An invariant enforced on one channel can be violated through any operator that
  channel passes through.** Energy was masked out of the rule's update and the rule
  minted it anyway, through a non-conservative transport operator it controlled. Check
  the operators, not just the masks. Applies directly to the energy economy at S1.

- **Null models are not optional.** "The network is doing something" must be a
  measurement against the best baseline available — for S0 that is parameter-matched
  GRU and transformer on the identical stream, not a dead grid.

- **Compute scales as `cells × steps × params`.** Rules are applied at every cell every
  step, so a fat rule is expensive in a way a transformer's parameters are not. Expect
  the lattice-capacity axis to carry more of the load than the parameter axis.
