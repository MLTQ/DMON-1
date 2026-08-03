# S0-R: stable differentiated kernel

Status: preregistered before GPU allocation, 2026-08-03.

## Question

Can typed tissue with bounded graph operators remain plastic at a constant learning
rate long enough to use additional budget, and does bounded private cell identity create
causal specialization without sacrificing character-model capability?

This is deliberately narrower than the full project S0. It validates the kernel on
which true asynchronous training, continual memory, growth under demand, and metabolism
will depend.

## Factors

Four creature arms per seed:

| Arm | Graph operators | Private cell identity |
|---|---|---|
| anonymous-unbounded | raw linear/dot attention | off |
| anonymous-bounded | spectrally rescaled, unit-ball attention, bounded values | off |
| identity-unbounded | raw linear/dot attention | bounded per-cell expression |
| identity-bounded | spectrally rescaled, unit-ball attention, bounded values | bounded per-cell expression |

Parameter-matched GRU and transformer controls are trained separately for the anonymous
and identity-sized budgets. Bounded/unbounded pairs have identical raw parameters.

## Protocol

- Tiny Shakespeare, contiguous 90/10 split.
- Three seeds: 7, 13, 21.
- 24,000 updates, 12 lanes × 32 characters.
- Constant AdamW learning rate `1e-3` after 200-update warmup.
- 8 input, 16 stream-memory, 64 compute, 16 relay, and 8 output cells.
- Hidden width 96, 12 dendrite slots with 8 active, 3 microsteps per character.
- Held-out evaluation every 500 updates; complete curves retained.
- Health: total and per-module gradient norms, effective operator norm, hidden/message/
  logit scale, rejected updates, and tissue update rates.
- Causal reads: reset, freeze internal, freeze compute, freeze relay, within-compute and
  within-relay state shuffle, and memory-zero.

## Primary decisions

### Stability

Bounded treatment passes if all three bounded seed runs of an identity condition complete
with fewer than 0.1% rejected updates and no rising late operator/gradient pathology.
It earns a positive comparative claim if the matched unbounded condition either fails
or has materially worse late gradient health without better held-out BPC.

### Differentiation

Private identity passes if its within-tissue shuffle penalty exceeds the corresponding
anonymous arm by at least `0.10 BPC` on the three-seed mean, while internal freeze cost
remains positive. Freeze establishes work; shuffle establishes private specialization.

### Capability guard

The identity-bounded organism must remain within `0.10 BPC` of its parameter-matched GRU
at the 24k endpoint and must beat its headless/null interpretation through positive
internal freeze cost. The transformer is a secondary matched-stream baseline.

## Interpretation constraints

- A surviving single seed is not a stability pass.
- A freeze cost without shuffle sensitivity means useful but interchangeable tissue.
- A shuffle cost without positive freeze cost is not credited as useful differentiation.
- Failure at constant `1e-3` does not reject typed tissue; it rejects this stability
  contract and sends operator/gradient telemetry back to design.
- Success is not the full asynchronous-learning milestone. It licenses the versioned
  runtime experiment in `runtime.py`.

## Launch

Prepare the immutable manifest without allocating a GPU:

```bash
.venv/bin/python -m sol2.benchmark --root sol2/runs/s0r
```

After GPU allocation:

```bash
.venv/bin/python -m sol2.benchmark --root sol2/runs/s0r \
  --device cuda:0 --execute
```
