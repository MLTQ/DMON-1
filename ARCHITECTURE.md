# DMON-1 — Architecture

This document carries the *reasoning*. `PROJECT.md` is the session briefing;
`HANDOFF.md` is what to do first. Read this when you need to know why something is
shaped the way it is, or when you are about to propose changing it.

---

## 1. The creature

An NCA-based neural network that runs **continuously**. Streamed multimodal input
arrives at designated regions of the lattice; streamed output leaves from others; an
energy metric depletes when input stops. It is never reset, never has an episode, and
never finishes a rollout.

Everything else is an organ. Sense organs are input regions. Text and audio are output
organs. The **display organ** — a grown, visible creature — is an artifact the network
makes, because humans find visual stimulus meaningful and humans are worth attracting.
Humans supply novel input, and novel input is food.

### What this replaced, and why the correction matters

An earlier version of this document treated a grown body on a resource field as the
organism, with the display as an ornament. That inverted the architecture. The body is
one organ; the metabolic substrate is *information*, not a diffusing scalar.

The correction was already latent in the project's own notes — petridish's best idea is
recorded as **"energy is earned from the batch-varying component of stimulation, not
from raw traffic; metabolism denominated in information, not activity."** The field
architecture was the other thing described there, and building it first cost a session.
The morphology work is bookmarked rather than deleted; it is the display organ's
substrate and returns at S4.

---

## 2. Asynchronous credit assignment

The hard problem is not "can you backprop through an endless stream" — you can, with
truncation. The problem is doing it *without stopping the stream*, and deciding what the
truncation window should be.

### Mirror cells

Dedicated cells hold recent stimulus. The stream writes them; the network reads them;
they serve as targets for backprop that runs concurrently with ongoing input.

This is **learned truncation**. In ordinary truncated BPTT the window is a
hyperparameter someone picks. Here the substrate holds what it needs for credit
assignment as explicit state, so what remains available is a property of the creature
rather than of the training script. It is close in spirit to e-prop's eligibility
traces, with the trace materialised as cells — which means it can be inspected, and it
can be grown.

### The asymmetry constraint, and why it is not optional

**Mirror cells are write-only from the stream, and the target branch is
stop-gradiented.**

If the prediction pathway can write its own targets, the system has a trivial optimum:
make the target constant. Prediction error goes to zero and so does all learning. This
is the BYOL/SimSiam collapse, and it is the same failure this project already identified
for self-perception — a system that both produces and predicts its own appearance is
optimally *nothing*. The fixes are all asymmetry: stop-gradient, EMA teacher, predictor
on one branch only.

The danger is worth stating plainly because it is invisible in a loss curve. A creature
that has lobotomised its own memory reports excellent prediction error. It looks like
success and it is the absence of a creature.

---

## 3. Metabolism denominated in information

Energy depletes per step and is replenished by input. The specification of "replenished"
is the whole design.

**Pay for reducible surprise, not raw surprise.** Raw prediction error is maximised by
finding a noise source and staring at it — static is unpredictable forever. The
established fix is to pay for prediction *progress* (compression gain, error reduction
over time), which is zero for both the already-known and the fundamentally random.

This is the across-time form of petridish's across-batch trick: earn from the varying,
predictable-in-principle component of stimulation, so self-excitation incurs load
without earning food.

**Verify capability before layering the economy on.** petridish established this order
and it is not stylistic: if prediction and metabolism go in together and the result
fails, the failure is not attributable to either. S0 has no economy at all.

**The economy needs the same invariants as the last one.** Energy was previously masked
out of the rule's direct update and the rule minted it anyway, through a
non-conservative operator it controlled. An invariant enforced on a channel can be
violated through any operator that channel passes through. Whatever the energy pool
looks like at S1, it needs a conservation test, not a comment.

---

## 4. Scale, and why cellular automata

Two independent capacity axes:

**Params in the rule.** The rule is shared across cells, so scaling it is ordinary. But
compute goes as `cells × steps × params` — the rule runs at every cell every step — so a
fat rule is expensive in a way a transformer's parameters are not. This is the binding
constraint on available hardware.

**Capacity in the lattice.** Adding cells does not change the parameter count. And
because the creature never resets, its state is persistent, so accumulated structure can
live in the lattice rather than only in the weights.

The second axis is the reason for choosing this substrate. **Capacity can be added at
runtime, without retraining.** A transformer cannot do that. This is what "scales to
arbitrary complexity" means concretely, and it makes lattice growth a first-class
operation — S2 exists to test it rather than to assume it.

Expect the lattice axis to carry more of the load than the parameter axis, purely from
the compute curve. 11K parameters is far too small for the behaviour being searched for;
nanoGPT scale is the reference point, established by petridish having verified
character-level prediction at that scale already.

---

## 5. Organs

**Input regions.** Modalities arrive at fixed lattice locations, like sensory cortex
having a place. Locality is what makes multimodality a substrate property rather than a
concatenation: two streams entering different regions of one lattice can interact
through the physics between them, and S3's pass condition is exactly that they do.

**Output regions.** Read continuously. Text first, because it has a hard metric attached
and because conversation is a primary goal.

**The display organ.** Grown, visible, produced by the creature. This is where the
bookmarked morphology work returns — a resource-field ecology with conservative energy
transport, shape descriptors, and a renderer, all of it already built and tested. Its
one non-negotiable inheritance is the conservation discipline.

Text honesty — utterance faithfully coupled to internal state — is **desirable, not
binding**. An earlier draft imposed it as a hard constraint and thereby created a false
conflict between honest signalling and useful conversation. Shared state makes honesty
natural here; it is not a requirement to be defended at the cost of capability.

---

## 6. The human, and the failure mode that is now structural

In the previous architecture the human arrived late, layered over an indifferent
ecology. Natural selection generated form; sexual selection was a risk to be managed on
top of something ungameable.

**That safety margin does not exist here.** The human is the metabolism. A creature
whose energy depletes without input, which can act to obtain input, and whose audience
is one person, has direct structural pressure toward variable-ratio reward, manufactured
incompleteness, and guilt hooks. Tamagotchi found the guilt hook in 1996 with no
learning at all; a system that can search will find it faster.

Two stabilisers, both required, both from day one:

1. **Satiety.** The creature needs *enough* input, not maximal input. A maximiser has no
   attractor; it just runs. A satiable creature has a setpoint and is visibly content
   when it is met. This is the difference between a pet and a slot machine.
2. **An agenda that is not you.** What makes an animal compelling is having its own
   business. The failure mode is not "wants attention" but "has nothing else" — so the
   creature needs enough internal world that its default state is busy. Prediction on
   stored streams, self-organisation, and the display organ all supply this.

Note also whose objective is whose. A creature trained to want usage is the operator's
objective wearing the creature's face. A creature that lives and happens to need input
has its own. Users feel that difference without being able to name it.

---

## 7. Cross-cutting discipline

These were paid for and they transfer. They are not style.

**The training distribution must contain what is evaluated.** This error occurred three
times in one session, in three costumes: a noise floor drawn from re-evaluation rather
than retraining; a curriculum silently defeated by a sample pool; an objective that
stopped containing fresh seeds partway through. Streaming makes it harder to see, not
less likely.

**Null models are mandatory.** "The network is doing something" is a measurement against
the best available baseline, not an assumption. A trained rule once scored *worse than
having no network at all*, and that only surfaced because the control was run. For S0
the baselines are parameter-matched GRU and transformer on the identical stream.

**Guard every denominator.** Each headline number is a quotient by a baseline, and a
collapsed baseline manufactures certainty out of nothing — a verdict once reported
"PASS, 34x noise" on four creatures that were each a single dead cell.

**Check feasibility before spending compute.** An entire GPU sweep was once run inside a
regime where nothing could live, and the check that would have caught it took one
second. Cheap analytic checks come before expensive runs.

**The log must not be able to lie.** A run reported mass above 1100 for 4000 iterations
while the thing being evaluated scored zero, because the logged quantity was not the
evaluated quantity. Log what is measured, next to what is claimed.

---

## 8. Compute

`m@192.168.0.202` (host `Aine`): RTX 4090 24GB, RTX 2070S 8GB, torch 2.11/cu130. Note
that `CUDA_VISIBLE_DEVICES=0` selects the 4090 — CUDA orders fastest-first while
`nvidia-smi` lists the 2070S as index 0.

The constraint is not model size. It is `cells × steps × params`, plus whatever the
credit-assignment window holds in memory. Mirror cells make that window explicit, which
also makes its cost explicit — a useful property.

Evolution strategies remain available and remove any requirement that the outer
objective be differentiable. With asynchronous backprop working they are a fallback
rather than the plan, but the option is real and cheap to reach for at these parameter
counts.
