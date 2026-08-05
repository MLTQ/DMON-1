# `relay_output_transport.py`

## Purpose

Implements an optional differentiated white-matter tract from relay tissue into output
tissue. It supplies adaptive internal transport without letting an external organ read
past its dedicated output sink.

## Components

### `RelayOutputAttention`

- Gives every output cell its own learned query and bounded scalar gate.
- Cross-attends over all relay cells with the same bounded operator conventions as the
  sparse connectome and attentive organs.
- Emits a per-output-cell message that is added before the ordinary output tissue rule.
- Reports tract gate magnitude and operator norms.

## Contracts

- Relay tissue is the only readable source; output-tissue message is the only write.
- Gates initialize to the serialized developmental value; the default is exactly zero
  and therefore a behavioral no-op.
- At zero gate, learning first reaches gate parameters; projections become recruitable
  after a gate opens.
- A nonzero initial gate immediately recruits attention projections while remaining
  bounded below one.
- Each output cell owns a distinct query and gate rather than sharing a pooled router.
- Bounded mode damps queries/keys to the unit ball and bounds values and final summary.
- The module has no access to memory slots, sensory input, targets, logits, or LLM
  controls.
