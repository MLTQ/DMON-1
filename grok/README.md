# grok/ — lean S0 organism (SOL-informed)

A clean experimental line for the **character organ**: continuous stream, sparse
directed dendrites, persistent state, event eligibility, reverse vector credit, and
honest matched-budget comparison against a GRU.

Not a port of `sol/` — a **thin reimplementation of findings** that earned status there.
See `findings.md`.

## What is on by default

| Mechanism | Source | Role |
|-----------|--------|------|
| Directed dendrites + local attention | petridish / prior grok | No Moore neighborhood |
| Mirror ring (stream-written only) | DMON architecture | Explicit event memory slots |
| Eligibility + delayed reward | SOL | Cross-window credit |
| Decoder-shaped reverse credit | SOL S15/S18 | Channel credit along real axons |
| Fast edge efficacy | SOL | Reward × edge eligibility |
| Multi-lane chunk BPTT | SOL stream | Continuous, GPU-efficient |
| Reset / shuffle ablations | SOL eval | Prove state is causal |

**Default off:** metabolism, structural rewiring (enable with `--structural`).

## Run

```bash
# Contracts + learning smoke
.venv/bin/python -m grok.smoke_test

# Train on Tiny Shakespeare (90/10 split, held-out eval + ablations)
.venv/bin/python -m grok.train --updates 2000 --device cpu --baseline

# Matched S0 benchmark writeup
.venv/bin/python -m grok.benchmark --updates 2000 --device cuda --out-dir grok/runs/s0-bench

# Optional morphology organ (S17-style global fitness when available)
.venv/bin/python -m grok.train --updates 2000 --structural --baseline
```

## Pass bar (S0)

- Held-out BPC competitive with parameter-matched GRU on the same stream.
- `reset_delta_bpc` and `shuffle_delta_bpc` substantially positive.
- Gradients reach dendrites, rule, and message value (reverse path).

## Layout

| File | Role |
|------|------|
| `findings.md` | Hard constraints from SOL |
| `model.py` | Organism tick + credit |
| `graph.py` | Dendrites + reverse transport |
| `stream.py` | Multi-lane continuous corpus |
| `train.py` | Chunk trainer + baseline |
| `evaluate.py` | Held-out + ablations |
| `structure.py` | Optional probe rewiring |
| `benchmark.py` | Comparison report |
| `smoke_test.py` | Gate |

## Relation to sol/

`sol/` remains the full scientific stack (routing trials, randomized structural
inference, energy, UI). `grok/` is for **fast iteration toward the GRU gap** with a
smaller surface area. Promote ideas into SOL-level rigor only after they move BPC here.
