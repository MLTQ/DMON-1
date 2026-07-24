# checkpoint.py

## Purpose

Persist a trained rule together with the physics it was trained under.

Separated from both `train_m0.py` and `substrate.py` so that anything wanting to *load*
a rule — the probe, the renderer, a future contingency analysis — does not have to
import a training loop, and so the physics module stays free of file formats.

## Components

### `save`
- **Does**: writes `{format, cfg, state_dict, meta}`.
- **Rationale**: `meta` carries the training geometry, grid, steps, iters and seed. The
  probe defaults its geometry from it, so a probe run cannot silently be conducted under
  the wrong ecology.

### `load`
- **Does**: rebuilds `SubstrateConfig`, constructs a `Substrate`, loads weights,
  returns `(sub, meta)` with the module in eval mode.

## Decisions

- **The config is stored with the weights, not alongside them.** A rule is only
  meaningful with respect to the ecology it was trained in, and a bare `state_dict`
  silently loses that. Loading a rule under different physics is a real experiment —
  cross-evaluation is the M0 verdict — but it has to be a *deliberate* one.
- **Unknown config keys warn and are dropped; missing keys warn and take defaults.**
  Raising would make old checkpoints unloadable every time the config gains a field,
  which in practice means people stop checkpointing. Both cases print, because a missing
  key means the rule is now running under physics it was not trained under, and that is
  exactly the kind of silent change that produces a confusing result three days later.
- **`weights_only=False`** because the payload is a config dict, not just tensors. Only
  load checkpoints you produced.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train_m0.py` | `save(path, sub, meta)` | Signature |
| `probe.py`, `render.py` | `load(path, device) -> (Substrate, meta)` | Return arity |
| `probe.py` | `meta["geom"]` present for trained checkpoints | Key name |

## Notes

- `FORMAT` is currently unused on read. It exists so that a future incompatible change
  has somewhere to branch rather than needing to guess from the payload shape.
