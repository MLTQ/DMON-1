# SOL2

SOL2 is the next experimental line for DMON-1: a persistent neural organism made from
typed tissues on a sparse directed connectome. It keeps SOL's scientific discipline and
Fable's lean capability path while removing the accidental assumption that every cell
must share one universal rule.

## What is new

- Separate input, compute, relay, and output tissue rules.
- Optional bounded private target/time-constant expression for every mutable cell.
- Organ-specific learned queries that attend only to dedicated output tissue.
- Parameter-count-neutral bounded graph operators: damp-only spectral rescaling and
  unit-ball attention, bounded values, and bounded edge bias.
- Stream-written sensory memory named honestly as short-term memory, not learned credit.
- Dormant dendrite slots and append-only relay growth that never deletes a working edge.
- Freeze and within-tissue shuffle probes reported together.
- Complete process checkpoints and matched GRU/transformer controls.
- A versioned background-learning runtime that keeps foreground ticks separate from
  backward computation.

## Architecture

```text
character stream
    ├── input tissue (fast typed rule)
    └── sensory-memory tissue (stream-written FIFO)
                │
       sparse directed connectome
                │
      compute tissue ↔ relay tissue
                │
          output tissue (sink)
                │
       attentive character output organ
```

Input, compute, and relay cells own persistent state; output tissue is cleared at each
token boundary so it cannot become an autonomous language model. A rule is shared only
within its tissue. In the identity treatment, each mutable cell additionally owns
bounded private expression. Relay growth appends both state and private parameters
while preserving the rules as tissue genomes.

Output targets read only compute/relay cells. The architecture therefore permits
low-traffic reserve cells but cannot solve the task through a direct sensory-to-decoder
shortcut. Gradient participation is logged as effective, material, and reserve
fractions rather than optimized as an activity reward.

## CPU gate

```bash
.venv/bin/python -m sol2.test_sol2
```

The suite is deliberately executable without pytest so a fresh project environment
can validate the architecture before optional test-runner dependencies are installed.

This gate must pass before GPU work.

## Short local smoke

```bash
.venv/bin/python -m sol2.train --model creature --device cpu \
  --updates 20 --eval-every 20 --eval-tokens 64 \
  --out-dir /tmp/sol2-smoke
```

## GPU experiment

The preregistration is `experiments/s0r-stable-differentiation.md`. Generate its
manifest without beginning training:

```bash
.venv/bin/python -m sol2.benchmark --root sol2/runs/s0r
```

`--execute --device cuda:0` starts the recorded arms after a GPU is allocated.
The preregistered first launch is a four-creature, seed-7, 2,000-update diagnostic via
`--creature-only`; a single weak differentiation result is not treated as a conclusive
architectural rejection.

## Deliberate omissions

- No claim that chunked training is asynchronous; `runtime.py` is the separate
  concurrency prototype.
- No metabolism yet. SOL's conserved ledger will return only with an externally audited
  prediction-progress inflow.
- No content-addressed memory organ yet. Retrieval first needs a dense curriculum and a
  shared write/query key contract.
- No random structural search. Growth is append-only until the informed-proposal
  experiment has a stable, differentiated substrate.
