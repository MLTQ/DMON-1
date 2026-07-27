# S14: Cell-count scaling on the RTX 4090

## Motivation

The 64-cell SOL organism reaches `2.33179` held-out bits per character while retaining
strong dependence on its persistent state and differentiated cell arrangement. It uses
only about `2.3 GiB` of the RTX 4090 during full training, leaving enough headroom to
test whether a substantially larger recurrent body preserves or improves that behavior.

Cell count is not a conventional width multiplier. The transition rule is shared across
cells, so additional cells primarily add persistent state, learned cell identity, and
directed workspace rather than proportionally more rule parameters. Scaling must
therefore earn its cost through measured language capability and state dependence.

## Hypothesis

A larger directed cell field can use additional persistent tissue as recurrent workspace
without losing trainability, viability, or dependence on streamed state. If the
hypothesis is wrong, held-out BPC will stagnate or regress even though the model fits in
memory.

## Capacity sweep

Keep every setting fixed except `cells`:

- 64 channels, eight directed dendrites, eight sensory cells, eight output cells;
- three message-passing steps per streamed character;
- batch 16, chunk length 32, seed 7;
- energy held at one and fast efficacy disabled;
- Tiny Shakespeare continuous train/validation streams;
- short runs at progressively larger cell counts, beginning at 128 and doubling while
  safe, with an intermediate size near the memory boundary if needed.

For every candidate, record:

- peak allocated and reserved CUDA memory during training and evaluation;
- training tokens per second after warm-up;
- parameter count and topology reachability;
- whether forward, backward, checkpoint, evaluation, and generation all complete.

Do not intentionally run to an out-of-memory fault. Stop scaling when extrapolated peak
use would leave less than roughly `3 GiB` free on the 24 GiB card, or when throughput
makes a full comparison impractical. The selected size is the largest candidate that
passes the complete path with that margin, not merely the largest allocation that can be
constructed.

## Capability run

Train the selected organism for 5,000 updates with the S12 optimizer policy:

- AdamW at `3e-3` through update 2,500;
- cosine decay to `3e-4` at update 5,000;
- no metabolism, fast efficacy, or structural rewiring, isolating cell-count scale.

Compare it with the 64-cell S12 checkpoint at aligned evaluations. Report best and final
BPC, post-best regression, next-character accuracy, reset-each-token BPC,
shuffled-cell BPC, topology, throughput, and peak VRAM.

## Gate

Call larger tissue promising only if it:

1. completes with full sensory/output reachability and numerical stability;
2. retains a meaningful penalty when persistent state is reset or cell identity is
   shuffled;
3. matches or improves the 64-cell capability trajectory rather than merely fitting in
   memory;
4. has enough throughput to complete replications or supplies a clear basis for renting
   larger hardware.

Do not replace the local live checkpoint unless the larger organism also passes the
existing promotion and UI compatibility gates.

## Capacity result

All probes completed 50 train updates followed by persistent, reset, and shuffled-state
evaluation, generation, and checkpointing. CUDA peak statistics were reset after model
and optimizer construction and recorded inside each run artifact.

| Cells | Parameters | Peak reserved | Train tokens/s | Reachable |
|---:|---:|---:|---:|---:|
| 128 | 127,426 | 2.67 GiB | 1,675 | 100% |
| 256 | 137,666 | 5.61 GiB | 1,682 | 100% |
| 512 | 158,146 | 10.84 GiB | 1,616 | 100% |
| 896 | 188,866 | 18.99 GiB | 1,607 | 100% |

The 896-cell probe retained `4.53 GiB` between PyTorch's peak reserved memory and the
card's reported total. A 1,024-cell run was not attempted: the measured linear slope
projects roughly `21.3 GiB` reserved before process/driver overhead, violating the
planned 3 GiB safety reserve. Scaling from 128 to 896 cells reduced measured throughput
by only about 4%, showing that the 4090 was underutilized by the smaller fields.

## Full-run status

The selected 896-cell organism is training for 5,000 updates on the RTX 4090 with fixed
anatomy, energy held at one, no fast efficacy, and the S12 decay from updates 2,500 to
5,000. It is a 14-fold increase in persistent cells over the local-live checkpoint but
only a 1.54-fold increase in learned parameters because the cell transition rule is
shared.
