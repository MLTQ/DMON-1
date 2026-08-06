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

## Stage-A result and stop

Stage A completed update 175 on the physical RTX 4090 and failed its route-formation
gate, so Stage B was not run. Qwen remained at zero trainable parameters and zero
gradient tensors. Peak allocation/reservation was 21.97/22.29 GiB. The raw artifact is
`l0c1aa-normalized-stage-a-s7-u175-result.json`, SHA-256
`41ab76d7ad8b82e714465cd139598bf22458772c5d89c7b8de9cbd6047260ea9`.

Normalization and optimization were mechanically healthy. Across updates 151--162
versus 164--175:

- raw target RMS varied `0.0343 -> 0.0365`, while the trained target remained exactly
  `0.0500` in both blocks;
- addressing KL improved `0.0601 -> 0.0385`, selected teacher mass stayed near
  `0.539 -> 0.529`, and every intended recurrent/local gradient group remained live;
- recall delta RMS was `0.00324 -> 0.00311`; retention therefore stayed
  `0.0648 -> 0.0623` and fell to `0.0581` in the last five updates, far below the
  required `0.50` and below C1z's late raw-scale retention;
- recall alignment rose only `0.082 -> 0.125` and fell to `0.066` in the tail. Relay
  alignment rose `0.168 -> 0.213`, but output alignment fell `0.0766 -> 0.0570`;
- carried differential transport RMS grew `0.00107 -> 0.00300`, while relay selection
  used about `29.7 -> 24.0` effective cells and output selection broadened
  `10.9 -> 12.4` of 16 cells. This is downstream response without recall-value capture.

The inherited language interface remained active even without behavioral reward.
Held-out normal loss was `0.56598`, better than the no-control floor `0.57526` and
internal lesion `0.61682`, but worse than wrong passage `0.56219` and memory lesion
`0.56897`. Strict normal wins/ties were `4/0` versus floor, `1/5` versus wrong passage,
`3/1` versus memory lesion, and `6/0` versus internal lesion. Development normal
`0.29175` was worse than floor `0.27222`, memory lesion `0.28833`, and wrong passage
`0.29443` only by a small margin in the favorable direction.

Fixed scale and removal of competing behavioral credit therefore do not repair this
exact same-coordinate copy route. The result localizes the failure more strongly than
C1z: one final-token content recall and a moving organism-derived sensory target do not
form a stable differentiated value representation under this curriculum. Escalating
the loss, extending the run, or enlarging the creature is not licensed. The next
architecture should allow learned representational transformation and give the
language organ several bounded, output-tissue-only injection sites, while retaining
paired behavioral and lesion attribution.
