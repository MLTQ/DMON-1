# L0-C1aa: normalized staged coherent delta

Status: preregistered after the full program review and before any C1aa optimizer
update.

## Question

C1z made correct passage better than wrong passage in mean likelihood, but its exact
target delta varied by question, live recall retained only about 9--14% of that delta,
and language-effect credit competed from the beginning. Can a fixed-strength paired
developmental signal first form a stable differential internal route, and can that
route survive when behavioral language credit returns?

## Why this is the bounded next test

This is not a scale-up, a duration test, or a new inference path. It isolates the
measured weak/variable differential signal while preserving the only treatment that
moved the correct-vs-wrong comparison in the right direction. Raw target magnitude
continues to be logged. The exact detached target direction, anatomy, Qwen interface,
and held-out causal tests do not change.

## Frozen lineage and normalization

- Clone the exact C1z seed-7 update-150 checkpoint, optimizer, RNG, schedule, 736-cell
  geometry, coherent top-16 recall, reference-centered prefix, and fresh-episode paired
  counterfactual policy. Use only the physical RTX 4090.
- Rescale each detached paired answer-value difference to RMS `0.05`, preserving its
  midpoint, direction, branch sign, and swap symmetry. Zero differences are rejected.
- Continue logging raw answer-target separation as
  `coherent_value_target_separation_rms`; log the trained fixed target as
  `coherent_delta_target_rms`. No normalized target exists at inference or evaluation.
- Preserve addressing KL weight `1`, temperature `0.1`, coherent transport temperature
  `0.05`, LR `0.001`, recall/sensor/effector multipliers `20/4/1`, exact-zero legacy
  losses, frozen Qwen, and all architecture settings.

## Stage A: form the route

Continue updates 151--175 with normalized paired recall-delta weight `64` and paired
differential relay/output transport weight `64`. Set task, binding, passage-effect,
causal-contrast, absolute coherent, semantic, and output-code credits to zero. The
effector receives no direct objective; the purpose is to establish recall, relay, and
output-tissue differentiation before it can be traded away for an easier language
policy.

At update 175, Stage A passes only if:

- raw and fixed target telemetry are finite and exact, addressing stays healthy, and
  every intended local/recurrent gradient group remains live;
- recall delta retention reaches at least `0.50` late and does not collapse in the
  final five updates;
- recall, relay, and output delta alignments are positive and stable; differential
  effective-cell counts show a nontrivial subset rather than a compulsory all-cell
  representation.

If Stage A does not materially improve retention over C1z, stop. Stage B cannot rescue
an internal route that was not formed.

## Stage B: reconnect behavior

Only after a Stage-A pass, continue the same checkpoint through update 200 with the
same normalized target and `64/64` local weights. Restore passage-effect distillation
weight `1` and causal-contrast weight `1`; keep ordinary task and binding weights zero.

Stage B succeeds if the internal retention and differential route survive, late causal
satisfaction exceeds C1z, control/effect separation remains live, and correct passage
beats floor, wrong passage, memory lesion, and internal lesion in held-out mean target
likelihood with a majority of strict per-question wins.

A local pass with a behavioral fail promotes the multi-depth differentiated language
organ described in `program-review-2026-08-06.md`. A behavioral pass promotes scaffold
withdrawal and continuous-lifetime retention. Neither outcome licenses blind scaling.
