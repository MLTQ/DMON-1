# system.py

## Purpose

The closed loop: one persistent SOL2 organism perceives frozen language features,
and its output tissue modulates the frozen backbone at several depths. Owns episode
scoring with the fp32 instrument fix.

## Components

- `EpisodeScore` — one scored question; `label_logits` is always fp32.
- `BrocaSystem` — organism + backbone + resolved depths; `observe_feature_sequence`
  (exposure; memory writes on), `_evolve` (organism over features → final-step
  controls), `depth_controls` (stacked controls → per-depth dict, with lesioning),
  `score_from_features` (question → injected or bare logits → fp32 label scoring).
- `build_broca_system(backbone, cfg)` — fresh organism, deterministic organ attach
  via `attach_organ_module` (fork_rng seeded), depth resolution.

## Decisions

- **fp32 label scoring, structurally.** C1l reported controls below the BF16
  forward-resolution floor and C1y logged exact ties; label logits here are cast to
  fp32 before any log-softmax, in one place all arms share.
- **Final-step controls, injected over the whole question.** The organism reads the
  question through frozen features (the sensing path is `encode`, no controls);
  scoring reruns the backbone under injection. Same separation as the C1 prefix
  flow, so exposure-state, not prompt engineering, is the only channel.
- **`bare=True` skips the organism entirely** — the floor arm shares no computation
  with the graft. `inject=False` runs the organism but injects detached exact
  zeros; the backbone contract makes that bitwise-identical to the floor, and it is
  asserted as an invariant, never reported as an independent control (the C1
  control-arm-collapse fix).
- **Lesioned depths inject detached zeros** rather than being omitted, so a lesion
  run exercises the identical hook structure.

## Contracts

- Scoring raises on multi-lane states (episodes are one lifetime lane).
- `depth_controls` rejects lesions of unselected depths and misshaped stacks.
- Backbone parameters must all be frozen at construction (raises otherwise).
