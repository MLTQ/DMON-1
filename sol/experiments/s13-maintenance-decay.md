# S13: Maintained physiology with late optimization decay

## Motivation

S11's directed maintenance floor kept every cell viable and produced the new live best
of `2.41516` BPC, but constant-rate training damaged that representation after update
3,750. S12's three-seed preflight removed the same late-regression pattern in an
energy-one organism. The mechanisms affect different layers: maintenance changes
conserved energy routing, while the schedule changes only optimizer step size.

## Hypothesis

An organism trained with S11 physiology and S12 decay can retain the maintenance-enabled
capability gain through late development. If the effects compose, the candidate should
follow S11 exactly before decay and finish closer to its best checkpoint afterward.

## Guarded protocol

Do not launch unless full-scale S12:

1. completes all 5,000 updates and passes its stability gate;
2. beats its exact constant-rate control at best BPC;
3. beats the same control at final BPC.

The S13 candidate changes only S11's optimizer schedule:

- seed 7, 64 cells/channels, eight directed dendrites;
- 5,000 updates, batch 16, chunk 32, three message steps;
- conserved metabolism with external gain `0.05`, transport `0.50`, and directed
  maintenance floor `0.001`;
- no fast efficacy;
- AdamW `3e-3` through update 2,500, cosine-decayed to `3e-4` at update 5,000.

The completed constant-rate S11 run is the matched control. The schedule is
mathematically inactive through update 2,500, so a same-device rerun must be bit-for-bit
identical over that interval. The first composition run uses the RTX 2070 Super while
S11 ran on the RTX 4090; recurrent numerical divergence across GPU architectures means
that run is a seeded robustness comparison, not a bitwise causal pair.

## Gate

The combined candidate must:

- retain full directed reachability, full healthy-stream viability, zero quiescence,
  and numerical-noise transport drift;
- beat S11 at second-half mean, best, and final BPC;
- reduce final and worst post-best regression;
- beat the current local live checkpoint before promotion.

Passing this seed-7 composition gate earns replication. It does not by itself establish
that maintenance or decay generalizes across topology seeds.

## Launch status

S12 completed with `2.33179` best/final BPC, zero post-best regression, and improvements
over its exact constant-rate control at both best and final checkpoints. The guarded
launch therefore passed. S13 began on the RTX 2070 Super while the RTX 4090 started the
independent seed-7 live exploratory-traffic replication.

The first S13 evaluations were `3.32914` and `3.09256` BPC at updates 250 and 500,
versus `3.34721` and `3.07293` for S11. The manifests match in every training setting
other than the inactive future schedule and output directory. This confirms that the
different GPU architecture prevents the planned bitwise pre-boundary identity. Continue
the run as a useful robustness candidate, but require a 4090 same-device replication
before attributing any final difference causally to decay.

## Seed-7 robustness result

The RTX 2070 Super run completed all 5,000 updates at its best checkpoint:

- best and final held-out BPC: `2.32164`;
- final and worst post-best regression: `0.00000`;
- next-character accuracy: `50.93%`;
- reset-each-token BPC: `7.91480`;
- shuffled-cell BPC: `6.14728`;
- mean healthy-stream energy: `0.97856`;
- viability: `1.0`; quiescent fraction: `0.0`;
- full sensory and output reachability.

Against the RTX 4090 S11 constant-rate control, S13 won all ten post-boundary aligned
evaluations. Mean second-half BPC improved from `2.53423` to `2.42483`, a `-0.10940`
delta. Best BPC improved by `0.09352`, final BPC improved by `0.19718`, and the final
checkpoint narrowly beat the S12 local-live checkpoint by `0.01015` BPC.

This passes the robustness and physiology gate, but the cross-architecture
pre-boundary differences prevent a bitwise causal attribution to optimizer decay. A
seed-13 replication is running on the same RTX 2070 Super. Promotion remains gated on a
same-device 4090 comparison or sufficiently strong replication evidence.
