# S11: Directed metabolic maintenance

## Motivation

S9 conserved metabolism accelerated full-scale seed-7 learning through update 2,500 but
did not improve best or final BPC. One of 64 cells became chronically quiescent from
update 2,000. Before quiescence, metabolism averaged 0.02296 BPC better than its
energy-one control; afterward the advantage collapsed to 0.00247 BPC.

The final checkpoint was globally well funded at mean energy 0.94252, but cell 34 held
only 0.00878 energy and essentially zero stimulation despite retaining eight named
incoming axons. This points to localized transport starvation rather than insufficient
total input.

## Hypothesis

A small maintenance request along installed directed axons can keep silent tissue
recoverable while preserving the early regularization benefit of a conserved economy.
The mechanism must not add parameters, introduce geometric neighbor exchange, subsidize
speculative probes, or create energy.

## Mechanism

- Add a configurable constant to activity-measured flow on installed non-self axons.
- Apply the existing per-source outbound normalization, source-owned transfer, target
  capacity limit, and exact conservation accounting to the combined request.
- Exclude self edges because they cannot redistribute energy.
- Exclude candidate probes because uninstalled anatomy must earn its own traffic.
- Default the maintenance floor to zero for exact checkpoint and live-model behavior.
- Force it to zero under `--no-metabolism`.

## Integrity gate

The deterministic suite must prove that zero activity can move existing energy only
from a named source to its installed target, that the source loses what targets receive,
and that total transport drift remains numerical noise. Old checkpoints must load with
zero maintenance flow.

## Capability protocol

The first full-scale candidate keeps the completed S9 seed-7 configuration and changes
only `--energy-maintenance-flow`. It is compared against the already completed
energy-one control and the zero-floor S9 organism at all 20 held-out checkpoints.

The gate requires:

1. no chronically quiescent cells under healthy streamed input;
2. exact energy provenance and full directed reachability;
3. lower second-half, best, and final BPC than the zero-floor S9 run;
4. lower best and final BPC than the matched energy-one control before replication;
5. no live checkpoint promotion unless the result also beats the existing 2.43925 BPC
   checkpoint and passes stability validation.

## Inference-only floor calibration

The completed zero-floor seed-7 checkpoint was evaluated on the same 2,048-token
held-out window while changing only the maintenance floor. This tests immediate
viability and behavioral perturbation before paying for retraining.

| Maintenance floor | BPC | Mean energy | Mean viability | Quiescent fraction |
|---:|---:|---:|---:|---:|
| 0 | 2.56175 | 0.94252 | 0.98438 | 0.015625 |
| 0.00025 | 2.56184 | 0.95157 | 0.98445 | 0.008781 |
| 0.0005 | 2.56185 | 0.95411 | 0.98467 | 0.002121 |
| 0.001 | 2.56209 | 0.95729 | 0.99524 | 0 |
| 0.002 | 2.56203 | 0.96123 | 0.99998 | 0 |

Transport drift remained at numerical-noise scale for every floor. `0.001` is the
smallest tested value that eliminates complete quiescence, and its immediate BPC change
is only +0.00035. It is therefore the first full-training candidate. The sweep does not
establish capability: weights trained with zero maintenance may not use the restored
cell, and the floor must earn its effect through a complete continuous run.

## Full-scale result

The seed-7 `0.001` candidate completed all 5,000 updates with 20 aligned held-out
evaluations. Every scored checkpoint retained full viability and zero quiescent cells.
Final mean energy was `0.96548`; external input and spending were `0.47440` and
`0.47455` per scored tick, with transport drift `1.42e-7`.

| Run | Best BPC | Final BPC | Best update |
|---|---:|---:|---:|
| maintenance `0.001` | **2.41516** | **2.51882** | 3,750 |
| zero-floor metabolism | 2.45957 | 2.56175 | 3,750 |
| energy-one control | 2.44573 | 2.52740 | 3,750 |

Candidate-minus-control evidence:

| Comparison | All-eval mean | Second-half mean | Best delta | Final delta |
|---|---:|---:|---:|---:|
| zero-floor | -0.00780 | -0.01823 | -0.04441 | -0.04293 |
| energy-one | -0.01744 | -0.01285 | -0.03057 | -0.00858 |

The full gate passes. Maintenance improved the second half, best, and final checkpoint
against zero-floor metabolism and beat the stronger energy-one control at best and
final. The run retained complete directed reachability. Resetting state at the final
checkpoint worsened BPC from `2.51882` to `8.91099`; shuffling cell identity worsened it
to `5.73405`.

Constant-rate optimization still regressed after the new best: final regression was
`0.10367` BPC and worst post-best regression was `0.14617`. Both are slightly better
than the prior live source (`0.11201` and `0.15154`) and pass a stricter `0.15` promotion
limit. The checkpoint was therefore promoted locally at `2.41516` BPC, improving the
previous live checkpoint by `0.02410`. S12 separately tests whether late learning-rate
decay can preserve this kind of gain.
