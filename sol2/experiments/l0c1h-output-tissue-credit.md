# L0-C1h: target-conditioned output-tissue credit

Status: protocol frozen 2026-08-04 before implementation or optimizer updates.

## Question

Can a training-only target code at dedicated output tissue prevent L0-C1g's global
steering-vector shortcut and make the same question under incompatible passages produce
distinct, correctly directed output states?

## Evidence boundary

L0-C1g already established that memory and internal tissue carry passage differences,
that a frozen Llama can be steered causally, and that 300 updates do not connect those
facts. The final steering diagnostic found a global control vector with only `0.00777%`
question variation. Correct exposure changed controls by about 1%, but correct-versus-
wrong passage changed them by only 0.1% and no held-out answer logits separated.

L0-C1h changes credit assignment at output tissue only. It is not evidence that larger
organisms, longer training, or the broader organism architecture are ineffective.

## Frozen treatment

Start from the same fresh seed-7 deterministic initialization as L0-C1g and retain:

- compact paired counterfactual bindings with the identical question and incompatible
  passage-bound labels;
- 96 memory, 64 compute, 16 relay, and 8 output cells at hidden width 96;
- bounded operators, three microsteps, four rank-eight control tokens, control gain 64,
  and recall gain 1;
- base learning rate `0.002` with recall/sensor/effector multipliers `20/4/0.1`;
- ordinary four-label task loss, unit bidirectional binding margin, frozen Llama,
  complete checkpoints, and all causal evaluation arms.

Add one training-only output-tissue objective. Mean-pool the final query-state of the
eight output cells, normalize it, and score it against a fixed deterministic four-code
orthonormal codebook at scale 4. Apply cross-entropy to each paired branch's incompatible
bound label with weight 1:

`loss = task_loss + binding_loss + output_credit_loss`.

The codebook has no trainable parameters and is regenerated from serialized seed 211.
The auxiliary scores may be logged during training but are never used during development
or held-out evaluation. At inference the only route remains memory/recurrent tissue to
output tissue to the detachable low-rank Llama effector.

## Frozen execution

1. CPU gates must prove deterministic codebook construction, opposing paired gradients,
   output-state telemetry, checkpoint configuration, and exact absence from evaluation.
2. Run two updates on the physical RTX 4090 only. Require finite gradients in every
   previously participating causal group, zero Llama gradients, and finite auxiliary
   logits/loss.
3. If healthy, continue the same checkpoint to update 25 and evaluate development.
4. Over updates 16-25, require mean auxiliary accuracy above 60% and mean paired output-
   state separation above `0.005` RMS before extending to update 100.
5. At update 100, require output-state separation to remain above `0.005` and at least
   six of eight held-out questions to produce non-identical normal-versus-wrong-passage
   label logits. A wiki-memory capability claim additionally requires normal exposure
   to beat wrong passage, reset, and memory lesion on matched loss and accuracy.

## Interpretation ladder

- Auxiliary gate fails: output-tissue credit or recurrent transport remains inadequate;
  do not deepen the Llama interface.
- Output states separate but Llama logits do not: promote deeper zero-residual injection
  as the next isolated effector treatment.
- Llama logits separate without correct causal ordering: the route exists but the target
  code or task loss remains misaligned.
- Correct held-out passage beats every causal control: passage-conditioned organism
  memory is supported and should be replicated across seeds and natural distractors.
