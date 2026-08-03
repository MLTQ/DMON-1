# `test_sol2.py`

## Purpose

Acts as the mandatory CPU gate before any SOL2 GPU allocation.

## Coverage

- Shape, bounded-state, reachability, output-sink, and dormant-slot invariants.
- Stream-only sensory-memory writes.
- Parameter-neutral bounded/unbounded treatment and effective spectral rescaling.
- Zero-initialized private identity behavioral equivalence.
- Gradient flow through every tissue, graph, identity, embedding, and output layer.
- Tiny repeated-stream learnability.
- Append-only relay growth with exact anatomy/state/optimizer preservation.
- Matched GRU and transformer parameter budgets.
- Exact next stream chunk, state, loss, optimizer moments, and update after checkpoint
  resume.
- Foreground ticks continuing while the background learner publishes an update.

## Contracts

Every test is deterministic on CPU. A failed contract blocks GPU work; it must not be
waived because a long experiment appears to learn.
