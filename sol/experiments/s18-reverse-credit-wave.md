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

## Results

All three matched runs completed update 2,000 on the RTX 4090. Reverse credit remained
bounded and active, with final mean absolute cell credit of `0.0519`, `0.0685`, and
`0.0535` for seeds 7, 13, and 21.

Capability did not replicate:

| Seed | Mean treatment − control BPC | Terminal wins | Effect / residual noise | Endpoint treatment − control |
|---|---:|---:|---:|---:|
| 7 | -0.0083 | 4 / 5 | 0.64x | -0.0220 |
| 13 | -0.0030 | 3 / 5 | 0.13x | -0.0024 |
| 21 | +0.0453 | 1 / 5 | 0.96x | +0.0301 |

Across all seed/checkpoint pairs, reverse credit wins 8 of 15. The pooled terminal mean
regresses by `0.0113` BPC and the mean endpoint regresses by `0.0019` BPC. Its pooled
effect/noise ratio is only `0.65x`, well below the `2x` comparison gate. The large seed
21 regression at update 1,000 recovers partially but remains adverse at four of five
terminal measurements.

Anatomy remains live rather than frozen:

| Seed | Active edges at 2,000 | Spawns | Prunes |
|---|---:|---:|---:|
| 7 | 35 / 64 | 15 | 12 |
| 13 | 33 / 64 | 14 | 13 |
| 21 | 29 / 64 | 4 | 7 |

Persistent state remains important: final reset-each-token BPC is `4.7304`, `4.7802`,
and `5.3634`, while shuffled-cell BPC is `4.7152`, `4.6157`, and `4.9043`.

## Decision

Reject scalar reverse-credit gain `0.25` for the working organism. The mechanism is
functioning, but its signal is not selective or stable enough to improve capability on
top of decoder-shaped credit and global-loss-gated morphology. Retain S17's zero scalar
gain as the adaptive baseline.

The next backward-credit intervention should improve selectivity rather than merely
reduce this scalar gain: credit should be channel-specific, normalized at branch points,
and tied to the remembered event that actually influenced the output. This points toward
learned or eligibility-normalized vector transport over axons, not another broadcast
amplitude sweep.
