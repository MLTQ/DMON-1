# SOL

SOL is an experimental line for the central creature: a persistent neural cellular
organism that is continuously stimulated, metabolizes input, remembers events long
enough for delayed reward, and continuously emits multimodal outputs.

This first slice is the **character organ**. It asks a deliberately narrow question:

> Can a homogeneous cellular rule running on a sparse directed connectome learn
> next-character behavior while its internal state, energy, and event memory persist
> across an uninterrupted stream?

It is “NanoGPT-like” only at the behavioral boundary—characters in, next-character
probabilities out. It is not a transformer imitation.

## Architecture

- Characters stimulate a small sensory population.
- Every cell owns private recurrent state but uses the same GRU rule.
- Targets read only their named dendrite slots. Synapses are directed, signed, and
  trainable; there is no convolutional neighborhood exchange.
- Designated output cells emit character logits after every input.
- Exact truncated BPTT sends task credit backward through the recurrent graph.
- Persistent eligibility traces remember which cells participated in preceding events;
  delayed scalar reward modulates those traces on the following tick, including across
  optimizer boundaries.
- An optional persistent credit wave enters output cells and travels backward through
  the signed directed axons, meeting event eligibility at upstream cells without
  introducing a separate teaching network.
- A complementary channel-shaped decoder correction follows the transpose of installed
  message paths. An experimental router can favor dendrites whose source cells remember
  an event aligned with that correction while equal evidence preserves the historical
  transport scale. Its explicit alignment gain changes routing selectivity without
  adding parameters or changing the decoder-error amplitude.
- An alternative reward-plastic router remembers which branches carried aligned
  correction, then lets later signed reward reinforce or suppress those branch choices.
  This closes a local forward-event/backward-credit/future-fitness loop with persistent
  state and no new model parameters.
- One weak exploratory source per target is measured by running the shared cell rule
  with and without that candidate message. Between differentiable windows, a bounded
  structural phase may replace a mature low-credit dendrite when the candidate is
  better, both endpoints can pay, and full sensory/output reachability survives.
- Externally caused stimulation propagates along measured edge flow. Stimulation and
  energy decay when input stops.
- Novel input is the only metabolic inflow. Energy is conserved while moving through
  named axons and exploratory probes, pays basal/activity/growth costs, and gates each
  cell from quiescent to fully active. A quiet cell can recover when later input or a
  funded neighbor reaches it.

Rewiring is disabled by default. The matched structural control receives the same
rotating exploratory traffic and continues ordinary learning, but never installs a
candidate into its permanent dendrite table. Every live traffic trial is checkpointed
with its on/off reward evidence and energy state, then aligned with validation before
and after the intervention so a candidate cannot count as successful while the body
collapses around it.

## Run

```bash
python -m pytest sol/test_sol.py
python -m sol.train --updates 300
python -m sol.train --file path/to/corpus.txt --device cuda --updates 10000
python -m sol.benchmark --model sol --out-dir sol/runs/sol-main
python -m sol.benchmark --model gru --out-dir sol/runs/gru-control
python -m sol.benchmark --model transformer --out-dir sol/runs/transformer-control
python -m sol.benchmark --model sol --freeze-edges --out-dir sol/runs/fixed-edge-control
python -m sol.benchmark --model sol --no-metabolism --out-dir sol/runs/capability-control
python -m sol.benchmark --model sol --no-reward --out-dir sol/runs/reward-control
python -m sol.benchmark --model sol --output-error-credit-gain 0.5 \
  --eligibility-routed-output-credit --eligibility-routing-gain 100 \
  --out-dir sol/runs/routed-credit
python -m sol.benchmark --model sol --output-error-credit-gain 0.5 \
  --reward-plastic-output-credit-routing --eligibility-routing-gain 100 \
  --out-dir sol/runs/reward-plastic-routing
python -m sol.benchmark --model sol --no-fast-plasticity \
  --structural-plasticity --out-dir sol/runs/growing-connectome
python -m sol.benchmark --model sol --no-fast-plasticity \
  --structural-probes-only --out-dir sol/runs/probes-only-control
python -m sol.serve --checkpoint sol/runs/sol-main/best.pt
python -m sol.report \
  --run sol=sol/runs/sol-main \
  --run gru=sol/runs/gru-control \
  --output sol/runs/S0-COMPARISON.md
python -m sol.promote \
  --run sol/runs/sol-main \
  --run sol/runs/fixed-edge-control
python -m sol.serve
```

The checkpoint bridge binds to `127.0.0.1:8765` by default. It keeps one organism lane
alive across prompts, measures real prompt gradients for the console's credit telemetry,
and never exposes the service beyond the local machine unless explicitly reconfigured.

`sol.benchmark` uses a fixed 90/10 contiguous split, records held-out bits per
character, state ablations, and directed sensory/output reachability, and writes complete
resumable checkpoints. The intended dual-GPU arrangement is SOL on the 4090 and matched
conventional controls on the 2070S; synchronous data parallelism across mismatched cards
would idle the faster device.

## Experiment horizon

Every meaningful SOL comparison must include the complete held-out trajectory, not only
best/final BPC. Completed summaries fit the final validation window and record its slope,
noise, and 95% interval. Paired reports additionally measure the aligned gap, win
fraction, effect relative to residual noise, and relative slope.

Reports must graph every arm and seed, disclose mixed-device controls, and compare the
between-arm gap with terminal movement and seed variance. A comparison horizon is
informative when both arms support a practical plateau or when one ordering remains
consistent and large relative to terminal noise. Continued noisy learning is reported,
but mathematical flatness is not required to recognize a robust treatment effect.

## Falsification

The prototype has failed if any of these are true:

1. A tiny repeated corpus cannot be overfit.
2. Directed synapses or early retained states receive zero gradient.
3. Resetting state has no effect on predictions.
4. Delayed reward produces the same update with and without an eligibility trace.
5. Energy does not fall during unstimulated ticks.
6. A structural candidate can bypass endpoint energy payment or disconnect the output
   organ.
7. Rewiring changes fan-in, duplicates a source, drops successful probe traffic at
   graft time, or contaminates untouched optimizer slots and persistent edge state.

## Deliberate omissions

- No cell birth or irreversible death; topology mutation preserves a fixed number of
  cells and dendrite slots, while energy can cause reversible quiescence.
- No visual or audio organs yet.
- Energy governs reversible viability but not reproduction or permanent tissue loss.
- Eligibility is neural event memory, not a full online parameter-gradient algorithm.
  Exact gradients remain truncated to the current optimizer window.
