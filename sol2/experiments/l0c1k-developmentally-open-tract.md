# L0-C1k: developmentally open attentive relay tract

Status: protocol frozen 2026-08-05 before implementation or optimizer updates.

## Question

Can the L0-C1j relay-to-output tract learn useful content-addressed transport when its
attention projections receive gradients from the beginning of development rather than
waiting behind an exact-zero gate?

## Evidence boundary

L0-C1j's gate opened from zero but remained small and did not improve representation
transport. A post-hoc amplitude sweep cannot test attention learning because its
projection parameters developed behind the small gate. L0-C1k changes only the tract's
initial developmental availability; it does not enlarge the organism, change the loss,
or claim that an always-open tract is biologically mandatory.

## Frozen treatment

Use a fresh seed-7 L0-C1j organism and the identical wiki-memory, paired binding,
eligibility loss, optimizer, tissue geometry, Llama graft, and causal controls. Change
only the attentive tract gate initialization from exact zero to `0.25` for every output
cell. Store the initial gate in `Sol2Config`; valid values lie in `[0,1)`. Bounded tract
gain remains 1 and gates remain learnable through the same `tanh` parameterization.

This is an internal developmental prior, not a training-only bypass. The tract reads
relay only and writes output message only during both training and inference. Targets,
label codes, memory slots, and LLM controls remain inaccessible to the tract.

## Frozen execution and gates

1. CPU tests must prove configured gate initialization is exact, bounded, serialized,
   and immediately recruits query/key/value/output projections while all topology and
   output-sink contracts remain intact.
2. Run two updates on only the physical RTX 4090. Require finite gradients in the tract
   and every prior causal group, zero Llama gradients, tract gate RMS above `0.1`, and
   finite bounded state.
3. If healthy, resume exactly to update 25.
4. Over updates 16-25 retain the same preregistered gates: output separation above
   `0.005` RMS, target projection above `0.003`, and output/relay retention above `0.8`.
5. Only a representation pass may extend to update 100 and the existing six-of-eight
   normal/wrong-logit and causal-control criteria.

## Interpretation

- Open tract still fails transport: the attention pool is erasing distributed relay
  distinctions or output tissue dynamics suppress incoming differences; do not tune
  gate amplitude again.
- Transport improves without target alignment: organize relay specialization or add a
  local predictive objective to tract attention rather than the output endpoint.
- Output aligns but Llama logits do not: move the isolated treatment to injection depth.
- Causal controls order correctly: replicate, then advance to natural passage memory
  and multi-turn communication.
