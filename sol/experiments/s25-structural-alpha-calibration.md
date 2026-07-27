# S25: Structural inference calibration

## Question

Did S24 lose capability because randomized structural traffic is harmful, or because
the exact `p <= 0.10` gate suppressed too much useful developmental growth?

S24 removed the fixed corpus phase but cut growth commits from 35 to 10 and all
mutations from 69 to 35 across three seeds. Its mean capability remained `0.077 BPC`
behind S23 at 2,000 updates even though it learned faster late in the run.

## Intervention

Keep the complete S24 organism and change one manifest field:

```text
structural_probation_randomization_alpha: 0.10 -> 0.25
```

Candidate-specific checkpointed ABBA/BAAB assignments, every raw reward, exact null
enumeration, positive advantage, reachability, endpoint energy, ordinary learning,
randomized routing, and structural/routing phase separation remain unchanged.

The S24 rank distribution suggests `0.25` could admit roughly 21 of the 95 observed
trials, between S24's 10 commits and S23's 35. Anatomy will diverge after the first
different decision, so this is a calibration target rather than a promised count.

## Controls

- S24 `alpha = 0.10` is the direct reliable-but-conservative control.
- S23 fixed ABBA with positive-advantage-only commitment gives the phase-biased
  high-mutation reference.
- Seeds 7, 13, and 21 run on the same RTX 4090 and use the same 4,096-token validation
  every 250 updates.

## Reduced gate

Stop first at 1,000 updates and graph:

1. all seed loss curves and paired gaps against both S24 and S23;
2. growth commits, prunes, active edges, and total mutations;
3. structural schedule/advantage decorrelation;
4. full cell/output reachability and intervention-aligned body stability;
5. complete terminal movement, without calling a moving curve converged.

Extend exact checkpoints to 2,000 only if the mutation/capability tradeoff remains
unresolved. Promotion requires capability recovery without restoring fixed-phase
correlation or materially exceeding the S23 mutation rate.

## Results

Pending.
