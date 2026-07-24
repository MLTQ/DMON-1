# render.py

## Purpose

Diagnostic visualisation. **For intuition only — never for the verdict.**

`PROJECT.md` is explicit that descriptors decide pass/fail and renders do not, and this
file must not be allowed to quietly become the thing runs get judged by. Its real job is
that a starvation transition or a failed reach, read off four scalars, is exactly the
situation where you convince yourself of a trend that is not there.

It paid for itself on first use: the first render of a smoke checkpoint showed the body
sitting in the centre with conductance high everywhere and **not reaching toward the
source at all**. The descriptors said "mass 12, box_dim 1.12" and would not have said
that.

## Components

### `render_state`
- **Does**: one RGB frame from a batch of states, rendering element 0.
- **Rationale**: colour mapping is chosen so the three distinct failure modes are
  separable at a glance:

  | Channel | Shows | Reads as |
  |---|---|---|
  | R | transport conductance (gated, body only) | did it build a vasculature? |
  | G | energy / `e_max` | the body, and how fed it is |
  | B | resource field / `field_cap` | where the food actually is |

  Food is blue, body is green, a body sitting on food is cyan, a conducting network is
  yellow-white. A body that never turns yellow never built transport.

### `rollout_frames`
- **Does**: no-grad rollout collecting frames.

### `to_gif`
- **Does**: PIL animated GIF.
- **Rationale**: GIF over a video container because it needs no codec and previews
  everywhere, including in a terminal-adjacent workflow.

## Decisions

- **Rendering element 0 only.** A grid of samples would tempt cherry-picking; the probe
  and contingency test are where batch statistics belong.
- **Nearest-neighbour upscaling.** Cells are the unit of reality here — smoothing
  invents structure between them, which is precisely the failure this file guards
  against.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `probe.py` | `render_state(x, r, sub, scale)`, `to_gif(frames, path)` | Signatures |
| `substrate.py` | channel 2 is transport, `gate_bias[1]` is its bias | Channel assignment |

## Notes

- Channel assignment is duplicated here from `substrate.py`. If the layout changes, this
  file breaks silently — it will render *something* regardless, which is the dangerous
  kind of breakage. The `substrate.md` contract table lists channel assignment as a
  breaking change for exactly this reason.
