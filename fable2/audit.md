# audit.py

## Purpose

G0: the interface audit that must pass before any training — the one-second check
before the forty-minute sweep (`PROJECT.md` Sharp Edges). Verifies the graft is
exactly silent at birth, every preregistered depth is causally reachable, and
gradient recruitment follows the designed two-stage pattern.

## Components / gates

1. `gate_zero_init_noop` — injected logits versus bare logits, max |Δ| must be
   exactly 0.0. Runs with gradients enabled on purpose: the no-grad path
   short-circuits zero controls by contract, so only the live-graph math path can
   prove the claim.
2. `gate_depth_sensitivity` — bounded random control banks at each depth alone must
   move the fp32 label logits; per-depth magnitudes and their max/min ratio are
   recorded (a degenerate ratio flags a dead or dominant site before training can
   hide it).
3. `gate_init_heads_live` / `gate_recruited_after_one_step` — through the actual
   paired loss: at exact zero-init only depth heads can receive gradient; after one
   optimizer step, trunk, sensor, and organism tissues/graph must come live.
4. `gate_backbone_frozen` — zero gradient tensors on the backbone in every stage.

## Decisions

- Gates run on the development split (never heldout), and the audit performs one
  real optimizer step — the report says so; a G1 run starts from a fresh system.
- `native_logit_error` is recorded so head/hidden-state mismatches on a new
  backbone surface here, not as a silent scoring bias.

## Contracts

- Emits `gates_all_pass`; the launcher treats a false as a hard stop for G1.
