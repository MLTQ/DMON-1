# L0-C1j: attentive relay-to-output tract

Status: protocol frozen 2026-08-05 before implementation or optimizer updates.

## Question

Does the organism need an explicit differentiated white-matter tract that gives every
output cell adaptive access to relay tissue before passage-conditioned state can steer
its language organ?

## Evidence boundary

L0-C1i retained `0.008796` RMS paired distinction in relay tissue but only `0.000810`
in output tissue, despite a 95.2% active eligibility gate and non-cancelling credit.
The seed-7 anatomy has only 10 active relay-to-output dendrites among 64 active output
dendrites, and three of eight output cells have no direct active relay source. This
does not prove sparse anatomy is the cause, but it identifies a concrete transport
constraint that more training of the same endpoint loss cannot remove.

## Frozen treatment

Retain the complete L0-C1i training protocol, including fresh seed 7, paired compact
bindings, 96/64/16/8 tissue geometry at hidden width 96, frozen Llama, control gain 64,
recall gain 1, optimizer rates, unit binding loss, and weight-4 eligibility-gated
differential output credit.

Add one optional internal `RelayOutputAttention` tract. For each output cell it owns a
learned query and bounded scalar gate. At every cellular microstep it cross-attends to
all relay cells using bounded key/value/output operators and adds the bounded result to
that output cell's ordinary sparse-connectome message before the output tissue rule.
The tract:

- reads relay tissue only;
- writes output-tissue message only;
- leaves the sparse connectome in place;
- cannot read stream memory, input cells, Llama features, label logits, or target codes;
- cannot emit Llama controls directly.

Every tract gate is initialized exactly zero, so enabling the module is an exact
behavioral no-op before learning. Learned per-output queries and gates allow output
cells to differentiate rather than share one pooled router. Bounded tract gain is 1.

## Frozen execution and gates

1. CPU tests must prove the enabled zero-gate organism is bit-exact to the disabled
   organism under matched initialization; the tract reads all and only relay cells;
   output cells remain sinks; nonzero gates change output tissue but cannot directly
   change another tissue; gradients recruit gates first and tract projections after a
   gate opens; configuration and checkpoints preserve the optional module exactly.
2. Run two updates on only the physical RTX 4090. Require finite gradients, zero Llama
   gradients, a nonzero learned tract gate, and no regression in bounded state/topology
   contracts.
3. If healthy, resume the same checkpoint to update 25.
4. Over updates 16-25 require mean output separation above `0.005` RMS, target-axis
   projection above `0.003`, and output/relay retention above `0.8` before update 100.
   Tract gate magnitude and gradient participation must be reported.
5. At update 100 retain L0-C1i's six-of-eight non-identical normal/wrong-passage logit
   gate and require normal exposure to beat reset, wrong passage, and lesions before a
   memory claim.

## Interpretation

- Tract remains closed: initialization/credit cannot recruit the route; fix that
  mechanism without interpreting language performance.
- Tract opens but output still collapses: relay state is not organized for attentive
  transport or output dynamics erase the message; inspect relay specialization and
  output time constants next.
- Output separates but Llama logits do not: the internal route works and the next
  isolated change belongs at the zero-residual Llama injection depth.
- Causal controls order correctly: replicate, move to natural unseen passages, and
  then extend the same organism into multi-turn generation.
