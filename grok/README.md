# grok/ — S0 streaming creature substrate

Dominion of a clean-slate implementation toward DMON-1's S0 goal:

> An asynchronously stimulated NCA-like network that learns **character-level
> language modeling on a continuous stream**, competitive with a parameter-matched
> GRU (and later a small transformer) — without episodic resets.

This package **does not** use the bookmarked `dmon/` morphology code. It is a
fresh line informed by `PROJECT.md`, `ARCHITECTURE.md`, and petridish insights.

## Design commitments (from petridish + architecture)

1. **No Moore/Neumann neighborhood CA.** Cells communicate over **directed
   dendrites** (fixed slots of specific sources). Connectivity is a graph, not a
   convolution. Growth/pruning can come later; the abstraction is axons/dendrites
   from day one.
2. **Continuous input ⇒ continuous credit.** The stream never pauses for a
   "training phase." Forward messages and reverse-mode gradients both flow through
   the same retained state every token.
3. **Reward needs a memory of the event.** Dedicated **mirror cells** hold a
   stream-written ring of recent stimulus (write-only from the stream; the rule
   cannot overwrite them). Dendrites can read them, so credit can reach cells that
   stored the past when the future loss arrives.
4. **Capability before economy.** S0 has no energy ledger. Bits-per-character
   first; metabolism is S1.

## Layout

| Module | Role |
|--------|------|
| `config.py` | Hyperparameters for model + stream trainer |
| `corpus.py` | Character stream (Tiny Shakespeare cache) |
| `graph.py` | Directed dendrite connectome |
| `cell.py` | Shared local rule (GRU) |
| `model.py` | Streaming creature: ports, mirrors, readout |
| `baselines.py` | Parameter-matched GRU null model |
| `train.py` | Continuous online training loop |
| `smoke_test.py` | Fast correctness + learning smoke |

## Quick start

From the repo root (after `uv venv .venv && uv pip install torch numpy`):

```bash
# Smoke: contracts + loss drops
.venv/bin/python -m grok.smoke_test

# Train (defaults: 128 cells, h=128, dendritic attention, batch 16)
.venv/bin/python -m grok.train --steps 5000 --device cpu

# Creature vs parameter-matched GRU (S0 comparison)
.venv/bin/python -m grok.train --steps 10000 --baseline --device cuda \
  --out-dir grok/runs/s0_compare

# Larger field
.venv/bin/python -m grok.train --n-cells 256 --hidden 192 --batch-size 32 \
  --steps 20000 --baseline --device cuda
```

## Pass / fail (S0)

- **Pass**: online next-char learning; bits/char improves vs chance; competitive
  with the matched GRU on the same stream; state is **not** reset between tokens.
- **Fail**: only learns when the stream is paused / batched into independent
  episodes; or cannot beat a dead baseline while looking good on a broken metric.

## GPU notes (Aine)

- `CUDA_VISIBLE_DEVICES=1` (or torch order with 2070S alone) for the free card when the 4090 is busy.
- BPTT memory ≈ `B × N × H × K × steps_per_token × truncate_every`. On 8GB: prefer
  `--truncate-every 32 --steps-per-token 3 --batch-size 16` at 128/128; T=64 OOM'd.
- Probe with `python -m grok.mem_probe` before long jobs.

## What is deliberately deferred

- Energy / satiety (S1)
- Runtime lattice growth (S2)
- Multimodal ports (S3)
- Display organ / morphology (S4)
- Lifecycle birth/death (petridish full stack)

Those belong later. S0 is the machine that must work first.
