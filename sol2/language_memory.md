# `language_memory.py`

## Purpose

Defines the first training task whose solution requires DMON-L0's persistent cellular
state rather than ordinary visible-context language modeling.

## Components

### `ContextErasureTask`

Specifies a small paired protocol: repeatedly expose one of two modes, allow
intervening tokens, erase the language context, present an identical query, and
require a mode-dependent response.

### `run_context_erasure_episode`

Runs that protocol through one continuing organism state and reports loss, accuracy,
control magnitude, and internal activity.

### `train_context_erasure`

Trains balanced persistent batch lanes with truncated lifetime gradients. Cellular
state continues between updates by default; computation graphs do not.

### `evaluate_context_erasure_controls`

Runs four causal arms over identical balanced modes:

- normal organism control,
- exact-zero language control,
- cellular reset only at the erased-context query, and
- silenced internal tissue.

## Decisions

- The visible post-erasure query is identical across modes, so the frozen language
  backbone cannot solve the balanced task from context alone.
- Repeated mode exposure gives the initially anonymous, leaky substrate a learnable
  developmental signal without handing the answer to the post-erasure language
  context.
- Output tissue is cleared each transition and reads only internal tissue; retaining
  the mode therefore requires the organism's sensory, memory, and internal route.
- The task does not train against shuffled or rewired anatomy. Those are inappropriate
  biological insults for this goal. The meaningful controls are loss of external
  influence, loss of lifetime state, and targeted tissue lesion.
- Persistent lanes model long-lived creatures while detaching state between optimizer
  updates keeps memory bounded.
- Control magnitude is observed, not rewarded directly. A later energy term can price
  excessive control once successful dependence exists.

## Contracts

- Mode and response tokens are distinct and inside the backbone vocabulary.
- Modes are balanced so a context-only strategy is capped near chance.
- Reset affects cellular state, not model weights.
- Internal-lesion evaluation uses the organism's existing zero-and-freeze mechanism.
- The trainer updates organism and graft parameters only; the backbone remains frozen.
