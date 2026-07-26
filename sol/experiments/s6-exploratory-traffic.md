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

## GPU status

The first S5 launch reached update 500 before the RTX 2070 Super disappeared from the
NVIDIA driver with an unknown-error device handle. The fault then caused CUDA unknown
errors in the concurrent RTX 4090 process and in new processes even when the 4090 was
selected explicitly by UUID. No graft had begun, so there is no partial scientific
result to retain.

The full S6 run should begin only after CUDA service is restored. Its primary gate is
the distribution of within-organism exploratory advantages and subsequent post-commit
stability. A frozen-topology organism may still be recorded for context, but it must not
decide whether the living body accepts an organ change.
