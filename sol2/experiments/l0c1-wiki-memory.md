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

### L0-C1d result and L0-C1e frozen compact curriculum

L0-C1d stopped at update 25 on 2026-08-04; the artifact is
`l0c1d-binding-m96-u25-result.json`. Joint graphs fit at 18.50 GiB peak allocation. The
trailing ten-update mean control separation rose only to `0.0000279`, paired accuracy
remained 10%, and label-logit separation remained zero. Binding loss worsened from a
1.14 mean over updates 1-10 to 1.46 over updates 16-25. The margin correctly penalized
the static policy but could not amplify the few differing answer tokens within roughly
100 retained passage tokens. The checkpoint was not extended.

Before any L0-C1e optimizer update, the next curriculum is frozen:

1. Architecture, differentiable recall, paired targets, joint binding margin, optimizer,
   geometry, and all development/held-out data remain unchanged.
2. Meta-training exposure contains only a compact temporary record with the exact
   question and designated answer. The shared wiki paragraph is omitted initially.
3. Development and held-out exposure remain the original natural wiki memory cards.
4. Every update reports gradient RMS and participating tensors/elements for sensor,
   recall, connectome, tissues, cell identity, and effector groups.
5. The update-25 routing gates remain control separation above `0.005` and paired
   accuracy above 25%. Only a passing compact run may progress to a curriculum that
   gradually restores wiki text as distractor context.

This is a curriculum simplification, not a reduced success claim. It asks first whether
the organism can learn a clean binding operation, then whether that operation survives
irrelevant context and finally transfers to natural unseen passages.

### L0-C1e result

L0-C1e stopped at update 25 on 2026-08-04; the artifact is
`l0c1e-compact-m96-u25-result.json`. Compact exposure increased trailing mean control
separation to `0.0001326`, about 4.8 times L0-C1d, and produced the first nonzero mean
four-label logit separation (`0.00625`). It nevertheless missed the `0.005` routing
gate, and paired accuracy remained 10%. The checkpoint was not extended.

Gradient telemetry localizes the remaining imbalance. Over updates 16-25, mean RMS
gradients were `0.0000115` for recall, `0.0000607` for the sensor, `0.00242` for the
connectome, `0.00119` for tissues, `0.00111` for cell identity, and `0.00623` for the
effector. Thus the static answer route received over 500 times the recall head's RMS
credit. The next intervention should rebalance plasticity or recall gain while retaining
the compact gate; additional organism size is not indicated by this result.

### L0-C1f frozen plasticity treatment

Before any L0-C1f optimizer update, the treatment is fixed from the measured L0-C1e
gradient imbalance:

1. Recall gain increases from 0.25 to its bounded maximum of 1.0.
2. Recall parameters use `20x` the base learning rate, sensor parameters use `4x`, and
   effector/output-organ parameters use `0.1x`. Connectome, tissues, cell identity, and
   all other organism parameters retain the base rate.
3. With base AdamW learning rate `0.002`, the resulting rates are `0.04` recall,
   `0.008` sensor, `0.0002` effector, and `0.002` elsewhere. Weight decay remains zero
   and global gradient clipping remains 1.0.
4. Compact paired bindings, the unit binding margin, continuing lifetime lane,
   96-memory-cell geometry, frozen Llama, development/held-out sets, and all causal arms
   remain unchanged.
5. Optimizer group names, rates, and parameter counts are serialized with raw gradient
   telemetry. The update-25 gates remain trailing control separation above `0.005` and
   paired accuracy above 25%; a miss stops the checkpoint.

The multipliers approximately balance the measured effective sensor/recall/effector
update scales without normalizing individual gradients or prescribing representations.
The recall drive remains bounded and still enters only sensory/recurrent tissue, so the
no-effector-bypass invariant is unchanged.

### L0-C1f result and L0-C1g frozen effector-amplitude treatment

L0-C1f stopped at update 25 on 2026-08-04; the artifact is
`l0c1f-balanced-m96-u25-result.json`. The learning-rate treatment briefly balanced
effective updates, but did not pass the frozen gate. Over updates 16-25, mean paired
accuracy was 10%, control separation was `0.00000365`, and label-logit separation was
zero. Held-out normal accuracy was 37.5%, exactly matching no exposure, reset, wrong
passage, memory lesion, and internal lesion; paraphrase accuracy was 75%.

The post-stop tissue diagnostic in `l0c1f-tissue-gain-diagnostic.json` localizes the
collapse. Across four compact incompatible pairs from matched fresh states, mean RMS
separation was `0.28899` in memory, `0.01569` in input, `0.00699` in compute,
`0.00829` in relay, and `0.00729` over all internal cells. Separation then fell to
`0.000616` in output cells and `0.0000243` in emitted controls, yielding exactly zero
four-label logit difference at BF16 precision. The organism therefore retained and
transported branch information; the present effector expressed it below the frozen
backbone's numerical resolution.

A read-only gain sweep on the same checkpoint found zero label-logit separation at
gains 1, 4, and 16. Gain 64 produced the first nonzero separation (`0.015625` RMS),
while gain 256 changed the static answer policy without producing paired separation.
Post-hoc scaling did not align either branch with its own target and is not counted as
a task success.

Before any L0-C1g optimizer update, the next isolated treatment is frozen:

1. Start from a fresh deterministic initialization and change only language-effector
   `control_gain` from 1 to 64.
2. Retain recall gain 1, base learning rate `0.002`, recall/sensor/effector multipliers
   20/4/0.1, compact paired bindings, unit bidirectional margin, 96 memory cells, all
   organism geometry, frozen Llama, and every evaluation arm from L0-C1f.
3. The effector remains bounded to `[-64, 64]`, reads only dedicated output tissue,
   and still enters the frozen backbone as four rank-eight continuous control tokens.
   This changes neither topology nor parameter count and creates no memory-to-LLM
   bypass.
4. Run a two-update capacity/gradient pilot, then extend the same checkpoint only to
   update 25 if it fits the RTX 4090 and remains numerically healthy.
5. The update-25 gates remain trailing control separation above `0.005` and paired
   accuracy above 25%. Causal held-out separation remains required before any memory
   claim.

This treatment tests a forward-pass precision bottleneck independently of larger
organisms, new losses, or direct internal readout.

### L0-C1g update-25 diagnostic and longer-horizon interpretation

L0-C1g completed its frozen update-25 diagnostic on 2026-08-04; the artifact is
`l0c1g-gain64-m96-u25-result.json`. The two-update pilot cleared the mechanical gate:
paired control separation was `0.000447`, paired four-label logit separation was
`0.08838`, every causal gradient group was finite, and peak allocation was 17.25 GiB.
Thus gain 64 repaired the BF16 forward-pass no-op without changing anatomy.

The short run did not learn content-dependent control. Over updates 1-10, mean control
and label-logit separation were `0.000447` and `0.00884`; over updates 16-25 they fell
to `0.000106` and zero. Trailing paired accuracy was 10% and binding loss was 1.44.
Held-out normal accuracy remained 37.5%, equal to no exposure, reset, wrong passage,
memory lesion, and internal lesion. Normal loss improved slightly over no exposure
(1.185 versus 1.197), but wrong-passage loss was identical to normal, so this is an
exposure effect rather than evidence of retrieved content.

Post-stop matched-state diagnostics still found large incompatible-branch separation
in memory (`0.2816` RMS) and measurable internal separation (`0.00551`), collapsing to
`0.000393` in output cells. Gain 64 amplified the resulting control difference but did
not train it into the correct answer direction within this horizon.

The update-25 gate was intentionally an early routing diagnostic, not a fair learning
horizon. With 12 meta-training questions it provides only about two visits per question,
while the organism must learn memory addressing, recurrent transport, output-tissue
expression, and the frozen Llama head's control geometry jointly. The next run therefore
continues this exact checkpoint exploratorily to 300 updates with evaluations every 50.
Because this extension was chosen after seeing the update-25 result, it is reported as
hypothesis-generating and cannot retroactively count as passing the frozen short gate.

### L0-C1g exploratory continuation execution

The exact update-25 checkpoint was resumed on 2026-08-04 toward update 300 on physical
RTX 4090 UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`; the RTX 2070S was not exposed
to the process. The result root remains
`data/dmon-l0/l0c1g-gain64-m96-u25` on the GPU host. Before resume, immutable
`checkpoint-u25.pt` and `metrics-u25.json` copies were created. Their SHA-256 digests
match the original files at launch: `87c312b2e0032cbb1bbe70672fdc0990dd11a08bbaf61d2289ae0dcd5e8e6070`
for the checkpoint and `567188adfb9bad006a3f21cc10412c66a0693a3e4c693dfdf254569cb88e0e9b`
for the metrics.

Only the declared exploratory horizon and reporting cadence changed: `updates=300`,
`eval_every=50`, and `checkpoint_every=50`. The saved optimizer state, compact paired
curriculum, geometry, control/recall gains, per-group learning rates, RNG state, sample
cursor, and continuing cellular state were restored from update 25. The first emitted
post-resume marker was update 30, confirming forward continuation rather than restart.
The live log and PID files are `continue-u300.log` and `continue-u300.pid` in the result
root. No capability interpretation is made until the full trajectory and final causal
evaluation are available.

The first resumed interval reached its atomic update-50 checkpoint successfully. Across
updates 26-50, mean paired accuracy was 38%, mean control separation was `0.000261`,
four-label logit separation remained zero, and mean binding loss was 2.1425. Development
accuracy rose to 75%, but normal, no-exposure, reset, wrong-passage, memory-lesion, and
zero-control arms were all the same 75%; the admitted-question normal score was zero.
This is a changed static answer policy, not evidence of wiki-memory use. The run continues
because update 50 is only the first declared observation on the exploratory trajectory.

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
