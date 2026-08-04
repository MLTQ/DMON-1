# L0-C1: held-out wiki memory through a frozen language organ

Status: protocol frozen 2026-08-04 before Llama baseline screening or organism training.

Baseline screening completed later on 2026-08-04; no optimizer update had occurred.

## Question

Can SOL2 learn a general operation for absorbing information through its frozen
language sensory graft, retain that information after the language-model workspace is
erased, and use it to control the same frozen Llama-3 8B when answering questions about
previously unseen articles?

Success here establishes a trainable persistent memory layer over an LLM. It does not
by itself establish agency, autonomous turn-taking, understanding, or organism-level
reasoning.

## Source boundary

Use only non-sensitive research summaries in `/home/m/wiki` whose underlying work was
published after the frozen Llama checkpoint's likely pretraining period. Exclude
personal prose, health claims, current political material, copyrighted books, raw
transcripts, and speculative collections.

Every record preserves:

- relative wiki path and SHA-256 hash;
- title, creation/publication date, type, and source URL or source-family identifier;
- the exact bounded passage shown to the language sensor;
- question, answer, distractors, and paraphrase family; and
- train/evaluation split assigned at the source-family level.

The frozen records are stored in `l0c1-wiki-memory-corpus.json`.

No source family may occur in both meta-training and held-out evaluation.

## Baseline screening

Before organism training, query the frozen Llama-3 8B without the passage using the
same post-erasure question and answer format. A fact is admitted only if deterministic
Llama does not already supply the correct answer. Report this selection explicitly:
the resulting benchmark measures acquisition of missing information, not general QA.

The initial gate uses four-way multiple choice with answer-order permutations. Chance
is 25%. Multiple choice provides an exact metric and avoids grading prose with another
model. A subsequent short-answer gate is required before claiming a general memory
interface.

### Frozen-Llama result

`l0c1-wiki-memory-baseline-result.json` records the committed 24-question screen using
Llama-3 8B Instruct and permutation seed 41. Baseline accuracy was 9/12 (75%) on
meta-training, 3/4 (75%) on development, and 4/8 (50%) on held-out. Strict admission
therefore retained three meta-training questions, one development question, and four
held-out questions.

This disproves the casual assumption that post-checkpoint wiki summaries are
automatically unknown to the model: answer choices made many facts inferable. The
held-out questions and success thresholds remain unchanged. Before training, the
meta-training curriculum is amended to add episode-specific counterfactual annotations
to the existing meta-training memory cards. The annotation binds the same question to
a deterministically selected alternative choice for that episode. Llama's static prior
cannot solve this varying binding, so learning pressure must cross the exposure/state
path. Unmodified wiki facts remain mixed into meta-training, and all development and
held-out passages remain unmodified. This amendment trains the write/query operation;
the unchanged held-out gate still asks whether it transfers to natural wiki facts.

### L0-C1a diagnostic and frozen repair

The first 50-update run completed on 2026-08-04 before the repair below. Its full
artifact is `l0c1-wiki-memory-u50-result.json`. Training accuracy improved from 10% in
updates 1-10 to 36% in updates 26-50, and held-out normal rose from 37.5% at update 2
to 50%. However, held-out `normal`, `no_exposure`, `reset_after_exposure`,
`wrong_passage`, `memory_lesion`, and `internal_lesion` were exactly equal at 50% with
equal mean loss. `zero_control` was 37.5%. The organism had learned nonzero LLM steering
but no passage-dependent memory.

Code inspection identified a concrete mechanism: every token wrote to the finite
stream-memory ring, so the question prompt completely replaced all 16 passage slots
before the answer was scored. Before any optimizer update under a repaired protocol,
L0-C1b is frozen with two changes:

1. Passage exposure opens the stream-memory write gate, while the question closes it;
   question tokens still drive input, recurrent dynamics, and the output organ.
2. Every meta-training update presents two incompatible temporary bindings for the
   identical question and choice order from the identical starting state, averages
   their loss, and carries only the primary branch into the continuing lifetime.

The second change makes a question-only, lifetime-position, or static output policy
incapable of satisfying both branches. L0-C1b starts from a fresh initialization and
retains the original development/held-out sources, causal arms, and success gates.

### L0-C1b result

The repaired 96-memory-cell, 50-update run completed on 2026-08-04; its full artifact
is `l0c1b-paired-m96-u50-result.json`. Mean paired training accuracy rose from 5% in
updates 1-10 to 38% in updates 26-50, with 16.80 GiB peak CUDA allocation. Held-out
`normal` reached 62.5% and `zero_control` remained 37.5%, proving that the trained
organism exerted behaviorally relevant control over frozen Llama.

It did not establish memory: `normal`, `no_exposure`, `reset_after_exposure`,
`wrong_passage`, `memory_lesion`, and `internal_lesion` were all 62.5%. Passage exposure
nevertheless changed internal activity substantially (0.1821 RMS versus 0.0586 without
exposure) and populated the preserved memory bank (0.7604 RMS versus zero), while
normal and no-exposure control RMS remained nearly equal (0.07830 versus 0.07805).
Thus the repair retained and propagated passage-dependent state, but the learned query
readout ignored that distinction and converged on a static steering policy.

The next protocol revision must target credit routing rather than organism size or
update count: remove the within-episode gradient stop on stream-memory writes and test
a bounded content-addressed recall head that injects retrieved memory into input or
recurrent tissue. It must not connect memory directly to the LLM effector, so the
existing output-sink/no-bypass invariant remains intact. A matched paired-branch
separation measurement is required before another held-out run.

### L0-C1c frozen routing repair

Before any L0-C1c optimizer update, the routing repair is fixed as follows:

1. Stream-memory writes remain differentiable within one exposure/query episode, while
   the continuing state is still detached at every optimizer boundary.
2. During the closed-write query phase, the language organ uses the current contextual
   feature to content-address valid memory slots with bounded query/key/value operators.
3. The bounded recalled vector is added to input sensory drive. It cannot enter output
   tissue or the frozen LLM effector directly; all behavioral influence must still pass
   through recurrent compute/relay tissue and the output sink.
4. The paired incompatible-binding curriculum and 96-cell memory bank remain unchanged.
5. Every paired update records RMS separation of its control banks and four-label logits.

The first decision gate is routing, not held-out accuracy. By update 25, the trailing
ten-update mean control separation must exceed `0.005`, and paired accuracy must exceed
the 25% four-way floor. If either misses, stop before a longer held-out interpretation.
If both pass, continue to update 50 and apply the original causal memory gates.

### L0-C1c result and L0-C1d frozen reward repair

L0-C1c stopped at its preregistered update-25 routing gate on 2026-08-04. The artifact
is `l0c1c-recall-m96-u25-result.json`. Peak CUDA allocation was 16.80 GiB. Recall made
memory presence behaviorally visible at update 2 (`normal` control RMS 0.00680 versus
0.00156 for no exposure), but it did not learn content identity. Across updates 16-25,
mean paired control separation was `0.00000699` and paired accuracy was 10%, missing
the respective `0.005` and 25% gates. Four-label logit separation remained exactly zero
at BF16 resolution. The checkpoint was not extended to update 50.

The failure identifies cancellation in the reward rather than another physical routing
absence: averaged cross-entropy permits two incompatible, nearly identical branches to
settle on one static answer. Before any L0-C1d optimizer update, reward shaping is fixed:

1. Both incompatible branches are built from the same starting state before one joint
   backward pass.
2. Ordinary mean four-label cross-entropy remains the primary task loss.
3. For branch targets `a` and `b`, an auxiliary hinge requires branch A's preference
   `logit(a)-logit(b)` and branch B's preference `logit(b)-logit(a)` each to reach 1.0.
4. The two hinge terms are averaged with weight 1.0 and reported separately from task
   loss. No hidden state, cell, route, or control direction is prescribed.
5. L0-C1d starts from a fresh initialization with the same recall architecture,
   96-cell memory, paired curriculum, routing gate, development set, and held-out arms.

This auxiliary reward is causal but anatomy-neutral: because question and start state
are identical across branches, the passage/state path is the only information capable
of satisfying both opposing preferences.

## Episode

For one lifetime lane:

1. Start from its continuing cellular state.
2. Present a bounded article passage token by token through the frozen comprehension
   path. No question or answer loss is applied during exposure.
3. Run zero or more unrelated distractor utterances and cellular ticks.
4. Destroy the LLM token/KV workspace while preserving `OrganismState`.
5. Present a question and four answer choices, with no passage text in the context.
6. Let SOL2 produce position-specific continuous controls while frozen Llama scores the
   answer token.
7. Apply loss only to the answer, never to prompt tokens.

Meta-training may update organism/graft weights after an episode. Held-out evaluation
never updates weights before, during, or after exposure to its articles; only cellular
state changes.

## Splits

- `meta_train`: at least four source families used to learn the write/query operation.
- `development`: at least one separate family for choosing update count and geometry.
- `heldout`: at least three source families and eight admitted facts never used for an
  optimizer step.

Question paraphrases and answer-order permutations are generated before training.
Development choices cannot inspect held-out normal-versus-control results.

## Causal arms

Evaluate identical held-out examples under:

1. `normal`: passage exposure, continuing state, active controls;
2. `no_exposure`: no passage, otherwise normal;
3. `zero_control`: passage exposure but exact-zero LLM controls;
4. `reset_after_exposure`: cellular state cleared after the passage;
5. `wrong_passage`: state exposed to a matched article containing a different answer;
6. `memory_lesion`: stream-written memory cleared after exposure while internal state
   remains;
7. `internal_lesion`: compute/relay state cleared and frozen for the query; and
8. `question_paraphrase`: normal state with a held-out wording and answer permutation.

The experiment does not train against cell rewiring or within-tissue shuffling. Those
are inappropriate survival requirements for biological-style learned anatomy.

## Measurements

- exact four-way accuracy and answer log-probability;
- normal improvement over frozen Llama/no-exposure;
- normal-minus-zero, reset, wrong-passage, and lesion effects;
- accuracy by retention delay and passage length;
- internal/memory activity and state change during exposure, distractors, and query;
- organ control RMS and output-cell attention allocation;
- per-cell gradient participation during meta-training;
- frozen-backbone parameter and gradient integrity;
- exact checkpoint/resume of weights, optimizer, lifetime states, sample cursor, and
  RNG; and
- later, short-answer exact/substring accuracy without an LLM judge.

## Gates

The pilot is mechanically valid only if the frozen backbone remains byte-stable, no
answer appears in the post-erasure context, and checkpoint resume reproduces the exact
next episode.

Memory success requires held-out normal accuracy of at least 62.5% (5/8) and at least
25 percentage points above each of `no_exposure`, `zero_control`, and
`reset_after_exposure`. Normal paraphrase accuracy must be at least 50%. Internal
attribution additionally requires an internal lesion drop of at least 25 points.

These thresholds are pilot gates, not population estimates. A miss identifies the
failed mechanism or scale and does not reject cellular memory from a single run.

## Compute boundary

All result-bearing GPU processes expose only physical RTX 4090 UUID
`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. The RTX 2070S remains unavailable to this
project. The passing L0-B smoke leaves roughly 9 GB of 4090 memory headroom, so no
larger rented device is justified before measuring this pilot.
