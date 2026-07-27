# S6: Within-organism exploratory traffic

## Question

Can SOL decide whether to grow one axon from evidence gathered inside the same living,
continuously adapting organism, rather than comparing a growing organism with another
organism whose topology is frozen?

## Protocol

1. Ordinary causal probes, delayed eligibility, reward, streamed hidden state, and
   backpropagation remain active.
2. The existing confirmation filter nominates one target, incumbent slot, and candidate
   source. The incumbent anatomy remains installed.
3. For an even number of streamed optimizer windows, only that target's candidate probe
   follows a deterministic ABBA schedule: on, off, off, on. Every other candidate probe
   remains on.
4. Candidate-on and candidate-off windows both update the same hidden state, body
   parameters, edge parameters, optimizer, energy, and reward memories.
5. Signed prequential reward is accumulated separately for the two traffic conditions.
6. If mean candidate-on reward minus mean candidate-off reward clears the configured
   margin and both endpoints can still pay, the candidate becomes a dendrite and pays
   the growth cost. Otherwise the candidate is rejected without anatomical mutation or
   energy expenditure.
7. Every decision is retained in a checkpointed ledger with its candidate identity,
   ABBA rewards, decision update, outcome, and body/endpoint energy. Held-out validation
   immediately before and after the trial is aligned to that record until the next
   intervention.

The fixed-topology probes-only run is now a diagnostic rather than the primary fitness
criterion.

## Integrity checks

- Target masking silences exactly one candidate input while all other probes continue.
- The ABBA schedule balances candidate-on and candidate-off observations and survives
  checkpoint resume exactly.
- Shared body parameters change during both traffic conditions.
- Incumbent anatomy and energy remain untouched until a positive decision.
- Rejection leaves anatomy, optimizer slots, stream-local edge state, and energy live;
  no rollback of the body is attempted.
- A committed candidate initializes the graft from its probe-equivalent coefficient,
  resets only the reused edge slot, increments morphology, and pays growth energy once.
- Full repository gate: 53 tests passed, with the same four pre-existing
  `PytestReturnNotNoneWarning` warnings in `dmon/test_conservation.py`.

## Natural CPU smoke

Configuration: Tiny Shakespeare, 16 cells by 16 channels, four dendrites, no metabolism,
no fast plasticity, 400 updates, two confirmation phases, 40-window exploratory trials,
and 20 candidate-on plus 20 candidate-off observations per completed trial.

Results:

- completed trials: 2
- commits: 2
- rejections: 0
- active trial at completion: no
- first within-organism reward advantage: `+0.04057`
- second within-organism reward advantage: `+0.03349`
- total rewires: 2
- best held-out BPC: `4.36290`
- final held-out BPC: `4.42543`

This is a mechanism smoke, not a capability claim. The small model and short stream show
that real exploratory traffic can drive balanced, checkpointable decisions while the
whole organism learns. They do not establish that the resulting morphology improves
language modeling.

## Decision-aligned survival smoke

After adding the permanent trial ledger, a fresh 16-cell, 400-update run used the same
live-body protocol with validation every 50 updates. It recorded three complete trials:

| Trial | Decision | ABBA advantage | Pre-trial BPC | Post-trial BPC | Survival |
|---:|---|---:|---:|---:|---|
| 1 | reject | -0.05386 | 5.44351 | 5.24031 | pass |
| 2 | reject | -0.02793 | 5.24031 | no isolated evaluation | pending |
| 3 | commit | +0.00571 | 4.74495 | 4.55362, then 4.32248 | pass |

The accepted graft charged exactly 0.005 mean energy to each endpoint and reduced
whole-body mean energy from 1.0 to 0.999375. The organism continued improving afterward
and the completed run was stable. This demonstrates that the artifact can distinguish a
positive local traffic decision inside a surviving body from two rejected organs and can
admit missing follow-up evidence instead of manufacturing a verdict. It still does not
show that the graft caused the later language improvement: the ABBA reward difference is
the local causal evidence, while decision-aligned held-out BPC is the survival gate.

Two matched seeds extended this into a three-seed preflight:

| Seed | Trials | Commits | Rejections | Evaluable | Survived | Pending | Final BPC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 3 | 1 | 2 | 2 | 2 | 1 | 4.32248 |
| 13 | 4 | 3 | 1 | 4 | 4 | 0 | 4.20667 |
| 21 | 1 | 0 | 1 | 1 | 1 | 0 | 4.35458 |

Across eight trials, exactly four advantages were positive and committed and four were
negative and rejected. Mean advantage was +0.00200, median was -0.00257, and the range
was -0.05386 to +0.10234. All seven trials with an isolated follow-up validation passed
the body-survival threshold; every completed run reached its best BPC at the final
evaluation. This validates sign selectivity and evidence coverage at smoke scale, not a
language-capability benefit.

## Full GPU protocol

The full experiment uses three independent living organisms rather than assigning one
GPU to a frozen-topology primary control:

- RTX 4090: seed 7, followed by seed 21.
- RTX 2070 Super: seed 13 concurrently.
- 64 cells/channels, eight dendrites, eight sensory and output cells, three message
  steps, batch 16, chunk 32, 5,000 updates.
- Three structural confirmation phases, 100-update phases, 500-update warmup, 250-update
  minimum edge age, and 300-window exploratory trials.
- No fast efficacy and no ordinary metabolic drain, preserving the earlier capability
  comparison while retaining real growth-energy payment.
- Held-out evaluation and checkpointing every 250 updates. Each 300-window trial
  therefore receives decision-aligned follow-up unless it resolves at the run boundary.

The gate requires balanced ABBA evidence, a mix of commits and rejections, complete
reachability, no trial-level body collapse, and stable late held-out BPC across seeds.
Only after that within-organism gate passes is a probes-only run useful as secondary
context; it cannot decide whether a graft lives.

## GPU status

The first S5 launch reached update 500 before the RTX 2070 Super disappeared from the
NVIDIA driver with an unknown-error device handle. The fault then caused CUDA unknown
errors in the concurrent RTX 4090 process and in new processes even when the 4090 was
selected explicitly by UUID. No graft had begun, so there is no partial scientific
result to retain.

The full S6 run should begin only after CUDA service is restored. Its primary gate is
the distribution of within-organism exploratory advantages and decision-aligned
whole-body survival. A candidate win is insufficient if subsequent held-out BPC exceeds
the configured regression threshold before the next intervention. A frozen-topology
organism may still be recorded for context, but it must not decide whether the living
body accepts an organ change.

## Full GPU seed-13 result

After GPU recovery, seed 13 completed the 64-cell protocol on the RTX 2070 Super. The
original 5,000-update boundary landed 100 windows into an eighth trial, so the same
checkpointed organism continued to update 5,200 rather than abandoning a candidate
organ mid-evaluation.

The clean final ledger contains eight resolved trials:

- five commits and three rejections;
- exactly 150 candidate-on and 150 incumbent-only windows per trial;
- mean within-organism decision advantage `+0.002494`;
- eight of eight decision-aligned whole-body survival passes;
- zero pending or unstable trials;
- five real rewires with exact fixed fan-in and complete directed reachability.

Individual advantages were:

`-0.001695, +0.001276, +0.009552, +0.000784, -0.000194, -0.000733,
+0.003963, +0.007000`.

Every rejection left topology and body energy unchanged. Each commit paid the configured
growth cost once; live body energy after the last commit was `0.99922`, with full
viability. The organism
remained stable at `2.45638` best and `2.53128` final BPC, with final regression
`0.07490`. Resetting state worsened final BPC to `10.93587`, and shuffling cell identity
worsened it to `5.05249`.

This seed passes the living-organism morphology gate: exploratory traffic can accept and
reject candidate axons selectively while the shared body learns continuously, and every
resolved grafted/rejected body survives. It does not establish a global language benefit
from rewiring; seed-7 and seed-21 replications remain queued.

## Full GPU seed-7 result

Seed 7 reached the nominal 5,000-update boundary with an eighth trial only 100 windows
into its 300-window ABBA observation period. The same checkpointed organism therefore
continued to update 5,200, preserving its state, optimizer, anatomy, stream position,
and reward memories until the trial resolved.

The final ledger contains eight resolved trials:

- five commits and three rejections;
- exactly 150 candidate-on and 150 incumbent-only windows per trial;
- mean within-organism decision advantage `+0.002020`;
- eight of eight decision-aligned whole-body survival passes;
- zero pending or unstable trials;
- five rewires with fixed fan-in and complete directed reachability.

The last candidate committed with advantage `+0.007272`, after which mean body energy
was `0.99922` and all cells remained viable. Language quality was weaker and less stable
than the S12 fixed-topology decay checkpoint: seed 7 reached `2.49942` best and `2.63140`
final BPC, a `0.13198` final regression. Resetting state worsened final BPC to `9.71051`;
shuffling cell identity worsened it to `6.47580`.

Together, seeds 7 and 13 establish two independent full-scale living bodies with a mix
of accepted and rejected organs and 16/16 trial-level survival. They do not show a
language benefit from rewiring. Seed 21 is deferred while the 4090 tests the higher
priority cell-count scaling hypothesis.
