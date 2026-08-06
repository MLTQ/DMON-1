# L0-C1u: direct semantic credit at content-addressed recall

Status: completed at update 25 on 2026-08-05; representation and causal gates failed,
while direct placement reduced aggregate recall-gradient collapse and exposed lifetime
FIFO saturation as a likely confound.

## Question

Can question-conditioned semantic credit preserve learning in the organism's actual
content-addressed recall vector, before recurrent and relay dilution? C1t weakly shaped
post-exposure memory but its relay target did not teach retrieval, and recall gradient
RMS collapsed by roughly 60x from the early to late matched window.

## Preserved inference pathway

```text
passage text -> frozen Qwen contextual states -> sensors -> stream memory
question text -> frozen Qwen contextual states -> content-addressed recall
recall -> input/recurrent/relay/output tissue -> continuous prefix
continuous prefix -> frozen Qwen transformer/head -> answer
```

The new observation is the exact live recall vector already injected into input tissue
on the final question token. Capturing it must not recompute recall, alter state, add a
readout, or expose a teacher at inference. The passage remains erased from Qwen's answer
context; only organism state crosses that boundary.

## Differential targets

Retain C1t's post-exposure memory target with fixed projection seed `307` and target
RMS `0.25`. Replace the post-question relay target with a direct recall target using
projection seed `309`:

```text
teacher_effect(P,Q) = qwen_feature(P + Q) - qwen_feature(Q)
teacher_recall_delta = project_309(effect(left,Q) - effect(right,Q))
student_recall_delta = recalled(left,Q) - recalled(right,Q)
recall_loss = MSE(student_recall_delta,
                  normalize(teacher_recall_delta, RMS=0.25))
```

The teacher features and projected axis are detached. Only the paired difference is
constrained. The final-token recall vectors remain connected to the existing recall
query/key/value/output parameters and to the exposure memory cells; the target never
enters recurrent state or the frozen language model.

## Fixed treatment

- Fresh seed 7 organism and optimizer; do not resume C1t weights.
- Frozen `Qwen/Qwen3.5-4B`, BF16, local files only, on physical RTX 4090 UUID
  `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39` only. The 2070S is excluded.
- Preserve C1t's 8/96/64/16/8 cells, hidden 96, 12/8 dendrites, three microsteps,
  four rank-8 anchored prefix controls, attentive relay tract, and initial tract gate
  `0.25`.
- Preserve compact paired counterfactuals, permutation seed 101, LR `0.001`, clip 1,
  and recall/sensor/effector LR multipliers 20/4/1.
- Preserve reference-centered prefix controls.
- Objective weights: passage-effect reverse KL 1, causal contrast 1, memory semantic
  credit 4, direct recall semantic credit 4. Relay semantic credit, absolute task NLL,
  absolute teacher KL, control energy, binding, output-code, and eligibility losses are
  0.
- Semantic target RMS is `0.25`; base semantic seed is `307`, making direct recall seed
  `309`.
- Run all CPU contracts, then a fresh one-update GPU preflight. A pass licenses one
  fresh 25-update run. Checkpoint and log every update; evaluate development and heldout
  only at update 25.

## Gates

### Mechanics and one-update preflight

- Recall capture is observational: enabled and disabled scoring produce exactly equal
  controls, logits, state, and health; capture returns the exact vector injected by the
  existing recall path and is absent when recall is not used.
- Fixed vector credit is deterministic, target-detached, branch-swap invariant, sends
  opposing gradients to paired recall vectors, and is zero for an aligned difference.
- A direct recall loss sends finite nonzero gradients to each of recall query, key,
  value, and output parameters and to live exposure memory; it does not recruit the
  effector through a shortcut.
- Qwen has zero trainable parameters and zero gradient tensors. All optimizer state,
  continuing state, metrics, and checkpoints remain finite.

### Update-25 representation gate

- Across recurrence-matched positions 1-12 versus 13-24, both memory and direct recall
  semantic losses improve in aggregate and their trailing mean alignments exceed
  `0.25`.
- Paired memory and recall separation remain nonzero. Recall query/key/value/output
  gradients remain finite and nonzero in the late window; early/late ratios are
  reported rather than replaced by aggregate gradient norm.
- All four late paired causal advantages are positive more often than not.
- Correct passage beats no exposure, zero control, reset, wrong passage, memory lesion,
  and internal lesion on held-out target likelihood. Eight-question accuracy remains
  descriptive and cannot override this continuous causal ordering.

Failure at recall alignment means the content-addressed operation or its memory-slot
representation needs redesign before scale. Recall alignment without downstream causal
control instead isolates recurrent transport as the next bottleneck. Passing both
licenses a longer persistence run and annealing of developmental credit; it is not yet
a general language-capability claim.

## Result

The exact-trace CPU contract and fresh one-update 4090 preflight passed. Capture-on and
capture-off scoring were behaviorally equal, direct loss reached live memory and all
four recall transforms without reaching the effector, and preflight query/key/value/
output gradient RMS values were respectively `1.01e-5`, `1.10e-5`, `1.21e-4`, and
`1.17e-4`. Qwen retained zero trainable parameters and zero gradient tensors.

The fresh update-25 run completed on the 4090 only. Peak CUDA allocation/reservation
was 14.16/16.40 GiB. The raw artifact is
`l0c1u-direct-recall-s7-u25-result.json` with SHA-256
`d260146f753740d716348cb2974a6a6b57adf1bf4a79a61f0e090035511c2598`.

Across recurrence-matched updates 1-12 versus 13-24:

- memory semantic loss improved from mean `0.06286` to `0.06186` on 7/12 examples,
  and memory alignment rose from `0.0434` to `0.1319` on 8/12;
- direct recall loss moved only from `0.062523` to `0.062497`, despite improving on
  9/12 examples, while recall alignment rose from `-0.0552` to only `0.0211`;
- recall separation collapsed from `7.78e-4` to `1.19e-4`; its trailing-update-16-25
  mean was `1.47e-4`, with trailing alignment `0.0444`, far below the `0.25` gate;
- passage-effect KL improved on 8/12 examples but worsened in mean from `0.9830` to
  `0.9976`; causal satisfaction moved from `41.7%` to `43.8%`, still below a majority.

Direct placement did improve gradient retention relative to C1t, but did not preserve
addressing. Aggregate recall gradient RMS fell from `1.27e-5` to `1.39e-6` (about 9x,
versus roughly 60x in C1t). Query and key still fell from `1.30e-6`/`1.56e-6` to
`3.67e-8`/`8.95e-8`, about 36x/17x. Value and output fell about 8x/11x. Sensor,
connectome, tissue, and transport gradients also declined while effector gradient rose.
Thus the direct target remains present, but the easiest late optimization continues to
be downstream expression rather than content selection.

Held out, normal exposure achieved 75% accuracy and mean loss `0.5549`. It beat wrong
passage in mean loss (`0.5631`) and memory lesion (`0.5773`), which is a more favorable
ordering than C1t, but it remained worse than no exposure/zero/reset at 87.5%/`0.5511`
and internal lesion at 75%/`0.5520`. Normal strictly beat the erased-context floor on
only 4/8 questions, wrong passage on 2/8, memory lesion on 4/8, and internal lesion on
3/8; it beat all six causal comparators on none. This does not establish useful recall.

The run also reveals a concrete training-state mismatch. The continuing lane ended at
memory cursor `1222` with only 96 FIFO slots—about 49 passage writes per update, enough
to fill the store after roughly two updates. Current-episode branch differences were
thereafter a small fraction of an undifferentiated full store, while held-out evaluation
starts fresh. Near-uniform content attention would therefore dilute the supervised
binding and naturally starve query/key gradients. This is an inference from the cursor,
separation, and gradient curves rather than proof that capacity is the only cause.

C1u stops at update 25. The next isolated diagnostic should train the identical
treatment with a fresh matched organism state per episodic update (or a small population
of fresh lifetime lanes), separating learned developmental rules from one state carried
across optimizer weight versions. If direct recall separation and query/key gradients
survive there, the next architectural step is learned recency/utility-based allocation,
forgetting, and growth for a genuinely continuing creature. If they still collapse,
recall should be rebuilt in one coherent coordinate frame: identity/residual value and
output maps, slot-wise semantic storage credit, and an addressing-specific objective.
