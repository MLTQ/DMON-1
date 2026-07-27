# S18: Reverse scalar credit in the adaptive connectome

## Question

Does a persistent output-originating scalar credit wave improve a continuously learning,
globally gated adaptive connectome beyond decoder-shaped output-error credit alone?

S17 becomes the matched control. It already transports a channel-wise correction
backward over directed axons and uses exact global-loss evidence to gate structural
candidates. S18 adds the complementary scalar signal implemented in S8: signed surprise
reward enters the output cells, travels sourceward over the live connectome, persists
across characters, and meets each cell's memory of recent events.

## Isolated intervention

The treatment changes only:

```text
--backward-credit-gain 0.25
```

The control's gain is zero. Both arms keep output-error credit gain `0.5`, decay `0.8`,
global-loss structural gating, live ABBA candidate traffic, variable fan-in, ordinary
backpropagation, and every S17 optimizer and topology setting.

This is intentionally not a frozen-organ comparison. Reverse credit changes local cell
dynamics while the whole organism continues receiving characters, adapting its body,
testing axons, and emitting predictions.

## Protocol

- Tiny Shakespeare character stream;
- 16 cells × 16 channels;
- four dendrite slots, two active per target at birth;
- four sensory and four output cells;
- two message steps per character;
- seeds 7, 13, and 21 run concurrently on the RTX 4090;
- 4,096-character validation at updates 250, 500, 750, 1,000, 1,250, 1,500, 1,750,
  and 2,000;
- decision window: the five aligned evaluations from 1,000 through 2,000.

## Decision rule

Report treatment minus S17 BPC for every terminal seed/checkpoint, per-seed and pooled
ordering relative to residual noise, reverse-credit telemetry, anatomy events, and
active fan-in. Nonzero terminal learning is allowed; the comparison must maintain a
directional effect that is large relative to observed noise.

Promote the reverse scalar wave only if its capability benefit replicates. A nonzero
credit metric without improved held-out prediction is mechanistic activity, not success.
