# L0-C1v: separate training lifetimes from organism memory

Status: completed at update 25 on 2026-08-05; fresh lifetimes preserved substantially
more plasticity but failed recall-alignment and held-out causal gates.

## Question

Did C1u's content-addressed recall collapse because one cellular state was carried
across optimizer updates until 1,222 writes repeatedly saturated its 96-slot FIFO?
Test the identical organism, objectives, data schedule, and optimizer while starting
each meta-training episode from a fresh matched cellular state.

This is a training-population diagnostic, not the desired deployed behavior. Passage
exposure and the erased-context question remain one continuous within-episode lifetime.
Only state inheritance between separate optimizer examples changes; learned anatomy and
weights continue across every update.

## Preserved pathway and ownership

```text
fresh episode state
  -> passage text -> frozen Qwen states -> organism memory
  -> erase Qwen context
  -> question text -> frozen Qwen states -> exact content-addressed recall
  -> recurrent/relay/output tissue -> continuous prefix -> frozen Qwen answer
  -> optimizer updates shared developmental weights
  -> next training episode receives a new cellular state
```

Teacher features remain detached training targets and never enter cellular state. Both
incompatible branches and the no-exposure control begin from the same per-update state.
Development and heldout evaluation are unchanged and already use matched fresh states.

## Fixed treatment

- Fresh seed 7 weights and optimizer; do not resume C1u.
- Add one serialized lane policy, `fresh_episode`; C1u's existing behavior remains the
  backward-compatible `continuous` default.
- Preserve frozen `Qwen/Qwen3.5-4B`, BF16, local files only, and physical RTX 4090 UUID
  `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. Exclude the 2070S.
- Preserve C1u's 8/96/64/16/8 cells, hidden 96, 12/8 dendrites, three microsteps, four
  rank-8 prefix controls, attentive relay tract, and initial tract gate `0.25`.
- Preserve compact paired counterfactuals, permutation seed 101, LR `0.001`, clip 1,
  recall/sensor/effector LR multipliers 20/4/1, and reference-centered prefix controls.
- Preserve objective weights: passage-effect KL 1, causal contrast 1, memory semantic
  credit 4, direct recall credit 4; all other losses 0. Semantic RMS/seed remain
  `0.25`/`307`, with direct recall projection seed `309`.
- Run CPU contracts and a fresh one-update preflight before one fresh 25-update run.
  Checkpoint/log every update; development and heldout evaluation only at update 25.

## Mechanics gates

- `continuous` returns the supplied lane unchanged; `fresh_episode` returns zero hidden
  state and cursor 0 while preserving batch, device, dtype, anatomy, and weight version.
- Both paired branches and every training-only control use the same selected episode
  start. Passage and question remain continuous within each branch.
- Checkpoints serialize the policy and last completed episode state. Resume restores
  weights, optimizer, schedule, RNG, and update count exactly; `fresh_episode` does not
  accidentally resume cellular contents on the next update.
- Per-update telemetry records episode start/end cursor. Every C1v start cursor must be
  0 and end cursor must be positive but no larger than the exposure token limit.
- Qwen remains zero trainable/zero gradient tensors; all metrics and optimizer state are
  finite, and only the 4090 is exposed.

## Update-25 comparison gates

- Compare recurrence-matched updates 1-12 versus 13-24 and C1u directly.
- Trailing recall alignment exceeds `0.25`, recall separation does not collapse across
  cycles, and query/key/value/output gradients remain finite and nonzero late.
- Fresh lifetimes materially improve late recall separation and query/key retention over
  C1u; otherwise FIFO saturation is not an adequate explanation.
- More than half of late updates make all four paired causal advantages positive.
- Held-out correct passage beats no exposure, zero control, reset, wrong passage,
  memory lesion, and internal lesion on mean target likelihood, with per-question strict
  counts reported.

A pass licenses the large juvenile organism with learned recency, utility protection,
allocation, and growth. A representation pass without causal output licenses transport
work. Failure sends the large design toward coherent sensor/memory/recall coordinates
and addressing-specific supervision rather than merely increasing cell count.

## Result

CPU contracts and the fresh one-update 4090 preflight passed. The preflight began at
cursor 0, wrote 54 passage tokens, reproduced C1u's first-update local metrics exactly,
and gave nonzero query/key/value/output gradients. Qwen remained zero trainable and
zero gradient tensors.

The fresh update-25 run completed on only the 4090. Every episode began at cursor 0 and
ended between 44 and 57 writes, so no cross-update cellular contents entered training.
Peak CUDA allocation/reservation was 14.15/16.39 GiB. The raw artifact is
`l0c1v-fresh-lifetime-s7-u25-result.json` with SHA-256
`d9f899d63013a504085826493c3ea4b26049f481bc7894881d54cf2dc0037877`.

Across recurrence-matched updates 1-12 versus 13-24:

- memory loss improved from `0.06350` to `0.06205` on 10/12 examples, and alignment
  rose from `0.0186` to `0.1224` on 9/12;
- direct recall loss moved from `0.062522` to `0.062500`, but alignment worsened from
  `-0.00794` to `-0.01451` and improved on only 5/12;
- recall separation still collapsed from `9.91e-4` to `2.21e-4` and trailing alignment
  was `0.00327`, far below the `0.25` gate;
- causal satisfaction was `56.25%` then `54.17%`; trailing per-advantage positive
  fractions were `0.50`, `0.50`, `0.50`, and `0.60`, not a joint majority result.

Fresh lifetimes did validate part of the C1u diagnosis. Aggregate recall gradient RMS
fell only from `3.53e-5` to `2.10e-5` (1.7x), versus 9x in C1u, and late recall
separation was about 1.85x C1u. Connectome, tissue, transport, and effector gradients
grew rather than collapsed. Query and key nevertheless fell about 9.4x and 5.6x, while
value/output fell 2.6x/1.5x. Removing old state therefore preserves plasticity but does
not teach stable content selection.

The preserved plasticity chiefly produced larger language intervention. Mean control
RMS rose from `0.00829` to `0.01432` across matched cycles and student effect RMS from
`0.678` to `0.749`; gradient clipping was frequently active. This was not useful
control. Held-out normal exposure scored 75%/`0.6002`, worse than no exposure, zero,
reset, and memory lesion at 87.5%/`0.5511`, internal lesion at 87.5%/`0.5527`, and even
wrong passage at 75%/`0.5933`. Normal lost strictly to the erased-context floor, memory
lesion, and internal lesion on all 8/8 questions, and beat wrong passage on only 2/8.

C1v therefore rejects cross-update FIFO saturation as a sufficient explanation. It was
a real gradient-suppression amplifier, but the current recall objective and coordinate
system remain the root bottleneck. The large juvenile should not multiply this anatomy
unchanged. Its storage and recall values should share the sensory substrate's coordinate
frame, addressing should receive a direct sparse/recency-aware objective, and expansion
should add utility-protected plastic capacity with demand-driven growth.
