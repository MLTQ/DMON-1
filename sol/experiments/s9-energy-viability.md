# S9: Conserved directed metabolism and reversible viability

## Problem

SOL previously added energy from propagated stimulation during every internal message
step. A recurrent signal could therefore pay for itself repeatedly, making the economy
decorative and allowing a body to remain active without an attributable external source.
Energy also entered the recurrent rule but never made a cell stop computing.

## Hypothesis

Novel streamed input should be the sole metabolic inflow. Existing energy should travel
with directed axonal traffic, pay computation costs, and determine whether a cell can
update or emit. Starvation should be reversible so the initial experiment measures
quiescence and recovery without prematurely adding irreversible tissue death.

## Mechanism

1. One character event contributes a bounded novelty-proportional energy budget to the
   sensory population exactly once.
2. Named dendrite and candidate-probe flow request source-to-target energy transfer.
   Total requests are normalized per source, so no source can spend more than it owns.
3. Targets accept only the transfer they have capacity to store; rejected transfer
   remains at its source, so transport conserves energy exactly.
4. Basal and activity costs are the only ordinary sinks. Recurrent stimulation remains
   an information trace and supplies no energy.
5. A parameter-free ramp maps cell energy to viability. Below the lower threshold a
   cell freezes its recurrent state and emits no message; above the upper threshold it
   is fully active. Sensory or incoming energy can reactivate it.
6. Output features are viability-gated, so an unfunded output organ cannot pretend to
   function from stale hidden state.

## Integrity gate

- parameter count is unchanged;
- adversarial directed traffic cannot increase total energy;
- an isolated named source loses exactly the energy its target receives;
- internal stimulation with no token cannot mint energy;
- reported external input and transport drift account for the full no-cost balance;
- a zero-energy body is quiescent and sensory input reactivates its sensory cells;
- old checkpoints gain safe default transport/viability constants;
- training, held-out evaluation, and the local bridge expose provenance and viability.

Irreversible cell death and reproduction remain out of scope until a calibrated
reversible economy learns without collapsing language behavior.

## Calibration variables

The first smoke held the inherited external gain at 0.025. It conserved energy and
stabilized mean energy, but measured input remained below spending and many cells lost
partial viability. Lowering the quiescence threshold exposed that some cells were
actually reaching zero; thresholds cannot compensate for an underfunded body.
`--external-energy-gain` therefore records the input budget explicitly for matched
intake sweeps alongside transport rate and viability thresholds.

## CPU calibration result

The selected calibration uses external gain 0.05, transport rate 0.50, quiescence at
0.01, and full activity at 0.05. The high transport rate is safe because source demand
and target capacity constrain accepted transfer; it describes delivery speed, not energy
creation.

Matched 16-cell Tiny Shakespeare runs used 400 updates, batch two, 16-character windows,
two message steps, no fast efficacy, and validation every 50 updates:

| Seed | Conserved metabolism final BPC | Energy-one control | Delta |
|---:|---:|---:|---:|
| 7 | 4.11596 | 4.19911 | -0.08315 |
| 13 | 4.33256 | 4.43384 | -0.10128 |
| 21 | 4.29217 | 4.40613 | -0.11396 |

Lower is better. The candidate won all three endpoints and 17 of 24 paired evaluations.
Mean endpoint improvement was 0.09946 BPC; mean all-evaluation improvement was 0.07422.
Across final held-out streams, mean energy was 0.97997, viability 1.0, quiescent fraction
zero, external input 0.08092, and spending 0.07696 per tick. Mean transport drift was
-3.5e-8 energy units, consistent with floating-point summation around exact
conservation.

This is a replicated short-budget win and a calibration gate, not a full-scale
capability claim. Healthy continuous input funds the complete body; finite silence still
drives it through the viability ramp to quiescence, and later input can recover it. The
next matched GPU experiment must determine whether the benefit survives 64-cell,
5,000-update training.
