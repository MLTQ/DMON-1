# L0-C1j: attentive relay-to-output tract

Status: stopped at the frozen update-25 gate on 2026-08-05; update 100 was not run.

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

## Result

The two-update pilot and exact resume to update 25 completed on the physical RTX 4090.
The tract recruited a finite dedicated gradient group, opened from exact zero to
`0.002844` RMS by update 2, and reached `0.011439` at update 25. The frozen Llama again
had `0` trainable parameters and `0` gradient tensors. The final metrics artifact is
`l0c1j-attentive-tract-m96-u25-result.json` (SHA-256
`e56cfd3007bdb6444d7e407fb8e0c8aeaafcd6fec05314861ef3688b07eef747`).

The trailing update 16-25 representation gates failed:

- relay separation averaged `0.00873367` RMS;
- output separation averaged `0.000808172` RMS;
- output/relay retention averaged `0.0969600`;
- target-axis projection averaged `-0.000198968`;
- tract gate RMS averaged `0.00648050` and ended at `0.0114395`.

Development accuracy was `0.75` in every causal arm and held-out accuracy was `0.375`
in every causal arm. L0-C1j therefore provides no memory evidence and stops at update
25.

The read-only gate sweep (`l0c1j-gate-sweep-diagnostic.json`, SHA-256
`54525b5d4c3b8f5a3b56c0dfdd6118c5581d9006864a18bb3afb7aae3b45302b`)
replayed four matched incompatible pairs from the saved lifetime lane. Counterfactually
forcing every tract gate from zero to `0.90` raised output/relay retention only from
`0.09556` to `0.10629` and output separation from `0.0004686` to `0.0005446`. Target
projection did become more positive (`0.0000283` to `0.0001688`) but remained far below
the `0.003` gate. Increasing amplitude after training is therefore insufficient.

This result constrains the exact-zero developmental treatment, not attentive transport
in general. At a zero gate, only gate parameters receive first-order gradients; query,
key, value, and output projections remain silent until the route opens. The learned
gate stayed small long enough that these content parameters never organized a useful
tract. The next isolated treatment should start the same bounded tract partially open,
allowing its attention content to learn from the first optimizer update. That test must
remain fresh and matched rather than mutating this failed checkpoint.
