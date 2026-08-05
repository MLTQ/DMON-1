# L0-C1l: frozen-effector causal passage contrast

Status: complete negative 2026-08-05.

## Question

Can the C1k organism align its existing passage-dependent internal signal to a fixed,
already calibrated language effector when reward is defined exclusively by causal
improvement over no-exposure and incompatible-passage controls?

## Motivation

C1k's open tract doubled relay and output separation by update 1000, but target
projection remained negative and held-out normal equaled no exposure. More duration
therefore increased transport without making the transported distinction useful.

## Frozen treatment

Copy the exact seed-7 C1k update-1000 checkpoint. Preserve its organism, continuing
state, optimizer moments, open relay-output tract, frozen Llama, corpus, permutation,
geometry, and learning rates. Make only these training interventions:

1. Freeze the language effector parameters, including its attentive output reader and
   low-rank control basis. Gradients may pass through the fixed effector into cells.
2. Set ordinary task, paired-binding, output-code, and local eligibility weights to
   zero.
3. For each paired counterfactual, run a matched no-exposure branch from the same state.
4. Optimize each exposed branch's target log-probability advantage over both the
   no-exposure arm and its incompatible-passage mate. Baselines are detached.

The Llama remains frozen and passage text remains absent from its question context.

## Pilot and gate

Run 25 additional updates, 1001-1025, on only physical RTX 4090 UUID
`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.

Require:

- zero trainable or gradient-bearing Llama parameters;
- exactly zero effector gradient tensors and nonzero gradients in internal causal
  groups, including transport;
- finite optimization, state, and control telemetry;
- trailing updates 1016-1025 mean causal advantage above zero, positive-advantage
  fraction above 0.5, and label-logit separation above its update-1000 floor;
- no claim of memory unless normal evaluation also beats no exposure and wrong passage.

Only a mechanical/representation pass may extend this arm. Held-out accuracy alone on
eight questions is descriptive and cannot override failed causal controls.

## Interpretation

- Training advantages rise but evaluation controls do not order: the compact
  counterfactual operation fails to transfer; redesign the exposure curriculum.
- Effector is gradient-free and advantages do not rise: the fixed head does not expose
  a learnable cellular control direction; calibrate a passage-neutral readout first.
- Natural development/held-out controls order correctly: replicate from an earlier
  checkpoint and a fresh seed before expanding corpus or organism size.

## Result

The 4090-only continuation completed through update 1025 without rejects or numerical
instability. The intervention was exact: all 138,784 effector parameters were frozen,
the effector reported zero gradient tensors, the frozen Llama remained 0 trainable / 0
gradient-bearing parameters, and sensor, cell identity, connectome, tissue, transport,
and recall groups all received gradients.

The representation gate failed unambiguously over updates 1016-1025:

- mean causal loss remained exactly `0.1000000015`;
- all four mean target-log-probability advantages remained exactly `0.0`;
- causal positive-advantage fraction and label-logit separation remained `0.0`;
- control separation was only `1.13e-5` RMS;
- output separation was `1.05e-4` RMS and target projection was `-3.55e-5`;
- relay-output gate RMS fell from about `0.191` to a trailing mean of `0.165`.

Held-out normal, no-exposure, wrong-passage, reset, memory-lesion, and internal-lesion
arms were all exactly 50% accuracy with mean loss `1.18738`; zero control remained
37.5%. Development controls were likewise identical at 75% accuracy. The organism
still supplies generic steering, but passage content remains causally unused.

This rules out the narrow hypothesis that C1k already contained a sufficiently
sensitive calibrated effector and needed only passage-exclusive internal credit. It
does not rule out causal passage training with a deliberately well-conditioned fixed
readout: C1k's passage-dependent controls remained below the BF16 forward-resolution
floor, so this particular frozen head exposed no usable direction for internal tissue
to amplify. Do not extend C1l; the next treatment must establish readout sensitivity
without restoring a trainable question-only shortcut.
