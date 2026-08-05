# L0-C1h: target-conditioned output-tissue credit

Status: stopped at the frozen update-25 gate on 2026-08-04; the update-100
extension was not run.

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

## Result

The implementation and two-update health pilot passed their mechanical gates, so the
same checkpoint was resumed through update 25 on the RTX 4090. The frozen Llama remained
untouched (`0` trainable parameters and `0` gradient tensors), all reported organism
gradient-group RMS values were finite, and peak CUDA allocation was 18,519,761,920
bytes. The final metrics artifact is
`l0c1h-output-credit-m96-u25-result.json` (SHA-256
`b25bfb0d5a7c92ed61555cec3fa133a53d0825ac28ac1f9675df4c6434b69815`).

The update-25 representation gate failed:

- mean output-code accuracy over updates 16-25 was `35%`, below the required `>60%`;
- mean paired output-state separation was `0.000791875` RMS, with a trailing maximum
  of `0.001652745`, both below the required `0.005` mean;
- mean paired branch task accuracy was `10%`, and mean control separation was only
  `0.000576883` RMS;
- the four-way auxiliary loss averaged `1.47072` over the trailing window, worse than
  the `ln(4) = 1.38629` uniform baseline.

The causal evaluation also provides no evidence that the passage-bound state controlled
the answer. Development accuracy was `0.75` for normal, reset-after-exposure, and wrong
passage. Held-out accuracy was `0.50` for all three. Held-out normal loss (`1.08911`)
was slightly better than reset (`1.14001`) and wrong passage (`1.09888`), but the normal
and wrong-passage predictions were identical in aggregate and their control magnitudes
were effectively equal. This small loss movement is insufficient to establish memory
use.

Per the frozen protocol, L0-C1h stops at update 25. This rejects the specific strategy
of applying unit-weight four-way credit to a mean-pooled output state under the current
paired training dynamics. It does **not** show that output organs, larger organisms,
longer training under a mechanism that passes the early gate, or the broader DMON
architecture are ineffective.

## Mechanistic reading and next treatment

The result sharpens the bottleneck. Both paired branches begin with nearly the same
query-time output state but receive incompatible auxiliary targets through shared
parameters. Cross-entropy can therefore lower its average by producing a shared label
mixture; it can only create passage-specific states through the already weak recurrent
passage-conditioned Jacobian. That is the very route the auxiliary was intended to
strengthen, so this objective supplies a destination without supplying an effective
transport or local learning mechanism.

Before attempting a deeper Llama injection, the next treatment should make target
credit local and temporally assignable inside the organism. A compact next step is an
output-organ eligibility trace: record passage-conditioned presynaptic activity during
exposure/query evolution, then use the training-only answer signal to reinforce the
specific relay-to-output interactions that were active. The inference graph remains
unchanged and receives no answer code. Its early gate should require the same question
under incompatible passages to separate at output tissue before any Llama score is
considered. This directly tests whether biological-style local credit can establish the
missing memory-to-output route, rather than asking ordinary shared backpropagation to
discover that route from two nearly identical endpoints.
