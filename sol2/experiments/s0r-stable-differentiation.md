# S0-R: stable differentiated kernel

Status: amended before GPU allocation or results, 2026-08-03.

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
and identity-sized budgets. They are closed-world capability/opportunity-cost references,
not arbiters of whether an evolvable substrate succeeds. Bounded/unbounded pairs have
identical raw parameters.

## Protocol

Before the full factorial, run a 2,000-update seed-7 creature-only preflight. It is a
plumbing and resource diagnostic: inspect learning, rejects, utilization distribution,
organ attention, freeze costs, and topology rewiring. A weak single-seed causal metric
does not reject an approach; only a mechanical bypass, numerical failure, or wholly
unused internal substrate blocks the replicated run.

- Tiny Shakespeare, contiguous 90/10 split.
- Three seeds: 7, 13, 21.
- 24,000 updates, 12 lanes × 32 characters.
- Constant AdamW learning rate `1e-3` after 200-update warmup.
- 8 input, 16 stream-memory, 64 compute, 16 relay, and 8 output cells.
- Hidden width 96, 12 dendrite slots with 8 active, 3 microsteps per character.
- Four learned character-organ queries attend only to eight per-token output cells;
  output dendrites read compute/relay tissue only.
- Held-out evaluation every 500 updates; complete curves retained.
- Health: total and per-module gradient norms, effective operator norm, hidden/message/
  logit scale, rejected updates, tissue update rates, organ-attention allocation, and
  material/effective/reserve private-gradient fractions.
- Causal reads: reset, freeze internal, freeze compute, freeze relay, within-compute and
  within-relay state shuffle, private-expression shuffle, degree-preserving topology
  rewire, and memory-zero.

## Primary decisions

### Stability

Bounded treatment passes if all three bounded seed runs of an identity condition complete
with fewer than 0.1% rejected updates and no rising late operator/gradient pathology.
It earns a positive comparative claim if the matched unbounded condition either fails
or has materially worse late gradient health without better held-out BPC.

### Differentiation

Substrate use requires positive internal freeze cost in every seed and at least `0.10
BPC` on the three-seed mean. Structure requires a positive degree-preserving rewiring
cost in every seed and at least `0.05 BPC` on the mean. Private identity earns a
specialization claim only if expression shuffle and within-tissue state shuffle are
both more damaging in the identity arm without a material normal-BPC regression.

Internal effective participation must exceed 10% on the three-seed mean. There is no
upper occupancy target: low-gradient reserve cells are permitted and are not penalized.

### Capability report

Normal held-out BPC must improve materially from initialization and remain finite. GRU
distance is reported as the cost of choosing the evolvable substrate on this closed
character task, not used as a hard pass/fail boundary. The tiny transformer is a
descriptive secondary reference; no claim about transformer scaling follows from it.

## Interpretation constraints

- A surviving single seed is not a stability pass.
- A freeze cost without shuffle sensitivity means useful but interchangeable tissue.
- A shuffle cost without positive freeze cost is not credited as useful differentiation.
- A freeze cost alone is partly enforced by the no-bypass anatomy; topology rewiring is
  required before claiming that learned structure matters.
- Uniform activity is not a goal. Some unused tissue is interpreted as reserve capacity,
  provided a nontrivial internal subset is causally useful.
- Failure at constant `1e-3` does not reject typed tissue; it rejects this stability
  contract and sends operator/gradient telemetry back to design.
- Success is not the full asynchronous-learning milestone. It licenses the versioned
  runtime experiment in `runtime.py`.

## Launch

Prepare the immutable manifest without allocating a GPU:

```bash
.venv/bin/python -m sol2.benchmark --root sol2/runs/s0r
```

GPU preflight:

```bash
.venv/bin/python -m sol2.benchmark --root sol2/runs/s0r-preflight \
  --updates 2000 --seeds 7 --creature-only --device cuda:0 --execute
```

After GPU allocation:

```bash
.venv/bin/python -m sol2.benchmark --root sol2/runs/s0r \
  --device cuda:0 --execute
```
