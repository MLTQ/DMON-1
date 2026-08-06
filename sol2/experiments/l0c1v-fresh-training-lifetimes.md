# L0-C1v: separate training lifetimes from organism memory

Status: preregistered before implementation and optimizer initialization on 2026-08-05.

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
