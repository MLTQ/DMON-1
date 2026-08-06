# `test_wiki_memory.py`

## Purpose

Provides the CPU gate for L0-C1 data provenance, prompt construction, causal state
transitions, and gradient ownership before using the real wiki or 4090.

## Coverage

- The frozen manifest contains 12 unique source families and 24 questions split into
  six meta-training, two development, and four held-out documents.
- Source hashes reject changed content.
- Answer permutations and paraphrase pairing are deterministic.
- All four answer labels are exactly one token in the fixture tokenizer.
- Passage exposure advances persistent state while baseline scoring does not.
- Wiki-episode loss recruits the continuous graft and never creates backbone gradients.
- Pre-transformer prefix episodes recruit the output decoder through the frozen
  backbone and the matched causal evaluator preserves all eight intervention arms.
- Non-finite organism gradients are rejected before mutation, with the named parameter
  and its pre-step value preserved.
- No-exposure and reset causal arms traverse the same exact scorer.
- Aggregate metrics remain bounded and auditable.
- Modern-backbone binding summaries retain accuracy, correct-label probability and
  margin, and both mean and worst paired full-logit separation.
- Counterfactual targets vary deterministically by epoch.
- Paired counterfactuals retain identical question wording and choices while assigning
  incompatible passage-bound targets.
- Incompatible passages produce nonzero control separation from the same starting state
  when the attentive recall path is active.
- The paired binding margin has nonzero opposing gradients at identical logits and
  becomes exactly zero once both branches prefer their own targets by the margin.
- Passage-causal contrast has nonzero gradients only in exposed branches, detaches the
  no-exposure and incompatible-passage baselines, and vanishes only when all four
  target-log-probability advantages exceed the declared margin.
- Full-vocabulary reverse KL is zero for matching distributions, detaches the teacher,
  and sends gradients into the student; the control-energy term measures expressed
  delta amplitude exactly.
- Passage-effect targets cancel arbitrary common teacher priors within floating
  tolerance, reduce exactly to the detached student baseline for zero teacher effect,
  and send gradients only into the conditioned student.
- Passage-visible teacher inputs contain distinct incompatible temporary bindings while
  returned teacher logits remain frozen and detached. Real-logit separation is a GPU
  preflight gate rather than a toy-tokenizer assumption.
- Question-only teacher input contains the identical formatted question and excludes
  both incompatible passage annotations.
- Frozen-teacher helpers expose detached final contextual features as well as logits.
- Seeded semantic projections reproduce exactly; paired local semantic credit detaches
  Qwen targets, gives identical cellular endpoints opposing gradients, is invariant to
  branch order, and reaches zero on an aligned differential state.
- Wiki episodes expose live post-passage memory tissue before question/reset/lesion
  operations and return no such tensor for no-exposure arms.
- Opt-in recall capture is behaviorally exact, equals the vector used by the existing
  transition, and direct paired semantic credit reaches recall query/key/value/output
  plus the live memory view while leaving teacher and effector gradients absent.
- Compact bindings remove the shared wiki paragraph while retaining the exact question
  and designated answer, and gradient telemetry exposes all causal subsystem groups.
- Plasticity groups cover every organism parameter exactly once and apply the declared
  recall, sensor, and effector learning-rate multipliers.
- Lifetime lane policy preserves continuous state exactly or creates a zero-cursor,
  zero-hidden fresh episode with the same anatomy, dtype, device, and weight version;
  invalid policies fail and checkpoints serialize the selected treatment.
- Atomic checkpoints reproduce organism/graft weights, optimizer metadata, continuing
  state, corpus identity, update cursor, language-control mode, distillation settings,
  effect-distillation weight, local semantic-credit configuration, and
  reference-centering treatment.
- The matched evaluator emits all eight preregistered causal arms.
- Reference-centered prefix evaluation makes no-exposure and explicit zero-control
  arms exact zero deltas.
- Fixed output codes are deterministic and orthonormal; incompatible paired targets
  deliver distinct gradients into live output tissue and aligned states satisfy the
  training-only probe without adding evaluation fields.
- Eligibility-gated transport credit gives identical output endpoints nonzero opposing
  gradients, is invariant to paired branch order, detaches presynaptic eligibility,
  and becomes gradient-silent when relay states do not differ.

## Decisions

- The tiny tokenizer is a deterministic interface fixture, not a language-quality
  model. The 4090 preflight separately validates Llama's real label tokenization.
- Learning success is not asserted on CPU here; the existing context-erasure gate
  already proves the substrate can learn a dense persistent distinction. This gate
  protects the richer experiment's mechanics.

## Run

```bash
.venv/bin/python -m sol2.test_wiki_memory
```
