# L0-C1i: eligibility-gated differential output transport

Status: protocol frozen 2026-08-05 before implementation or optimizer updates.

## Question

Can a paired, locally measured three-factor credit signal preserve passage identity
from relay tissue into dedicated output tissue without injecting an answer into the
organism or adding a route around its anatomy?

## Evidence boundary

L0-C1h's saved-lifetime diagnostic found mean same-question paired separation of
`0.230999` RMS in stream memory, `0.004388` in relay tissue, and only `0.000412` in
output tissue. Fresh-state relay/output separation was `0.008242/0.002995`. The next
treatment therefore targets the observed relay-to-output collapse and the symmetry of
shared endpoint cross-entropy. It does not alter the Llama effector, increase organism
size, or claim that relay activity already represents the correct answer.

## Frozen treatment

Start from the same fresh seed-7 L0-C1g/h organism and retain the paired incompatible
compact bindings, 96/64/16/8 memory/compute/relay/output cells, hidden width 96, bounded
operators, three microsteps, four rank-eight control tokens, control gain 64, recall
gain 1, base learning rate `0.002`, `20/4/0.1` recall/sensor/effector multipliers, unit
binding loss, frozen Llama, checkpoints, and every causal evaluation arm.

Remove L0-C1h's absolute four-way output cross-entropy from the objective. Reuse its
fixed seed-211 orthonormal label codes only to define an incompatible target axis for
the matched pair:

`axis = normalize(code[left_label] - code[right_label])`.

Mean-pool each branch's final output cells and form

`projection = dot(mean(output_left) - mean(output_right), axis)`.

Measure presynaptic eligibility as the detached paired RMS difference in final relay
tissue and map it to `[0,1]` with reference `0.005`:

`eligibility = clamp(relay_separation / 0.005, 0, 1)`.

The training-only local transport loss is

`eligibility * 0.005 * softplus((0.005 - projection) / 0.005)`

with objective weight 4. Unlike independent endpoint classification, its gradient is
nonzero and oppositely directed at identical output endpoints. The three factors are
passage-dependent relay availability, the incompatible answer axis, and postsynaptic
output difference. Relay eligibility is detached so the auxiliary cannot improve by
manufacturing a larger gate. Ordinary task and binding losses retain their existing
gradients.

No code, target, auxiliary score, or teaching drive is written into cellular state.
Development, held-out evaluation, and inference use the unchanged organism and frozen
LLM graph.

## Frozen execution and gates

1. CPU tests must prove zero endpoint difference receives nonzero opposing gradients,
   swapped branch order is invariant, a zero relay eligibility gate produces zero
   auxiliary gradient, relay eligibility itself is detached, configuration resumes
   exactly, and evaluation never invokes the loss.
2. Run two updates on only the physical RTX 4090. Require finite auxiliary telemetry,
   finite gradients in every previously participating organism group, and zero Llama
   gradients.
3. If mechanically healthy, continue the same checkpoint to update 25.
4. Over updates 16-25 require mean output-state separation above `0.005` RMS, mean
   target-axis projection above `0.003`, and mean output/relay separation ratio above
   `0.8` before extending to update 100. Report the measured relay gate so a pass cannot
   be attributed to a silently absent presynaptic signal.
5. At update 100 require the representation gates to persist and at least six of eight
   held-out questions to produce non-identical normal-versus-wrong-passage label logits.
   A memory result additionally requires normal exposure to beat wrong passage, reset,
   and memory lesion on matched loss and accuracy.

## Interpretation

- Eligibility or relay separation absent: the upstream representation is not stable
  enough under the actual training lane; target upstream transport or lifetime
  interference next.
- Relay remains distinct but output gates fail: this differential local objective is
  insufficient; add a learned/local synaptic plasticity rule rather than merely more
  optimizer time.
- Output separates but Llama logits do not: deepen the zero-residual Llama injection as
  an isolated effector treatment.
- Causal controls order correctly: replicate across seeds and replace compact bindings
  with natural unseen passages before expanding conversational generation.
