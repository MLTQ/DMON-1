# liquid.py

## Purpose

The F3 candidate cell rule: CfC-style leaky integration with learned,
bounded, input-dependent time constants. Stability by construction where the
GRU rule needed (and outgrew) a guard stack: five fatal blowups in five
attempts at sustained high LR (E1 3/3, F7 2/2).

## Components

- `LiquidRule` — `h' = (1−a)h + a·tanh(target)`, `a = 1/τ`,
  τ = τ_min + (τ_max−τ_min)·σ(gate), computed from a shared tanh backbone
  over `[messages, drive, h]`. Drop-in for `SharedRule` (same forward
  signature). Exposes `last_alpha_mean` for health telemetry.
- `backbone_width` — solves the backbone size so the rule's parameter count
  matches the GRU rule's (9H²+6H); whole-organism match 0.03% at H=128.
- `gru_rule_param_count` — the closed form being matched.

## Decisions

- **τ_min = 1** so a ≤ 1: the h-backbone of the per-step map is a strict
  leak; state bounded by max(|h|, 1) and abuse decays (smoke: |h|=5 → <1 in
  40 tokens). τ_max = 100 bounds the slowest cell at ~100-tick memory.
- **Gate bias −2 at init** (τ ≈ 13): the organism starts responsive and
  learns to slow down; starting slow starves early learning.
- **Δt fixed at 1.** Δt-awareness is the asynchrony/composability step and
  gets its own test; conflating it with the stability question would make
  an F3 result unattributable.
- **Honesty**: backbone contraction does not prove gradient stability (the
  W-paths through target and τ can still grow). F3's arms therefore include
  every regime that killed the incumbent — the claim is tested, not assumed.

## Contracts

- Enforced in smoke_test.py: param match < 1%, contraction after state
  abuse, gradient flow to backbone/target/tau_gate, α within
  (1/τ_max, 1/τ_min].
- Selected via `cfg.cell_rule = "liquid"` — a field that exists WITH a
  preregistration (F3), per config.md's no-flag-graveyard rule.
