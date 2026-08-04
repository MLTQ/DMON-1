# S1-P2: true organ attachment and retention

Status: preregistered before implementation and result inspection.

## Question

Can a mastered differentiated SOL2 substrate accept a distinct sensor/effector organ,
reuse its learned procedure, and retain the original capability through the original
organ? This replaces S1-P's contradictory remapping of one shared interface with two
independently parameterized physical ports.

## Starting organism

- Seed 7 larger/deeper S1-P2 checkpoint at update 5,000.
- 208 cells: 8 input, 32 memory, 128 compute, 32 relay, and 8 output.
- Hidden width 192, 12 of 16 dendrites active, 5 microsteps, 8 organ queries.
- Acquisition fixed-length-four checks: 97.66% and 98.76%; zero rejected updates.
- Checkpoint path on the GPU host:
  `/home/m/Code/DMON-1-sol2-preflight/runs/s1p2-acquisition/large-s7/acquisition.pt`.

## Organ boundary

Organ A retains the legacy embedding, sensory gain/bias, attentive output reader, and
decoder from acquisition. A newly attached B owns an independent copy of exactly those
interface parameter types. The graph, tissue rules, private cell expression, output
tissue, and living electrical state are the shared substrate. Every step names the
active organ explicitly; attaching B cannot overwrite A's parameters.

## Matched branches

All adaptation branches use batch 24, constant learning rate 1e-3, full program lengths
one through four, 2,000 updates, and evaluations every 200 updates. Intermediate curves
use 16 batches and terminal estimates/lesions use 64 batches. Full, organ-only, and
scratch B consume the same generated B examples; A-control and A-return terminal probes
also consume identical A examples.

1. `control`: continue A from the acquired checkpoint.
2. `full`: attach fresh B and train B plus the entire shared substrate.
3. `organ_only`: attach the identically initialized B, freeze the substrate and A,
   and train only B.
4. `scratch`: rebuild the exact seed-7 initial substrate, retain an unused A port,
   attach the identically initialized B, and train B plus the fresh substrate.

At every B checkpoint, score B by lengths one through four and score A through its
unchanged organ from a clone of the current living state. Report both the first
fixed-length-four batch after the switch and the full evaluation, so transient mode
recovery is not confused with weight retention.

## Removal and reattachment

After the full branch's 2,000 B updates:

1. Evaluate B and save the exact B module object.
2. Remove B from the organism without changing its parameters.
3. Exercise A for 500 updates with the shared substrate plastic.
4. Reattach the same B module object.
5. Score B immediately, then after 25, 100, and 250 B recovery updates.

The optimizer continues to own the off-body B parameters while detached; they receive
no gradients or decay because the inactive organ is absent from the forward path.

## Measurements

- Per-update answer loss and exact accuracy.
- Per-length A and B accuracy at every checkpoint.
- Full-minus-scratch early and late B accuracy.
- Full-minus-organ-only B accuracy.
- Immediate and sustained A return accuracy during B adaptation.
- B accuracy before removal, immediately after reattachment, and through recovery.
- Compute freeze, relay freeze, whole-internal freeze, and topology-rewire penalties on
  cloned final states. These are causal diagnostics, never robustness objectives.
- Rejected updates and per-cell/edge telemetry.

## Decision rules

The experiment is a strong success if:

1. B-full reaches at least 50% fixed-length-four accuracy.
2. B-full exceeds B-scratch by at least 10 percentage points in the first 10% of
   adaptation updates.
3. Warm A return remains within 15 percentage points of matched A-control.
4. Reattached B returns within 10 percentage points of its pre-removal accuracy by
   recovery update 250.

B-organ-only above 22.5% fixed-length-four accuracy is modular-reuse evidence, not a
mandatory biological constraint. A single seed can promote the architecture and expose
mechanisms, but capability claims require seeds 13 and 21.

## Performance boundary

The result-bearing launch remains batch 24 unless a compiled execution path passes
output, gradient, checkpoint-resume, and convergence parity. Larger-batch throughput
measurements do not retroactively alter this protocol.

All branches run exclusively on physical RTX 4090 UUID
`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. The 2070S is reserved for the separate
`jewels` program and must not receive SOL2 processes. Branches run in two memory-safe
waves rather than spilling onto the reserved card.
