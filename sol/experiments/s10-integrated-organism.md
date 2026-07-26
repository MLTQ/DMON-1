# S10: Integrated metabolism and exploratory anatomy

## Question

Do the two independently validated mechanisms compose inside one organism? S6 measured
candidate axons under live ABBA traffic with ordinary metabolism disabled. S9 measured
conserved directed metabolism without structural exploration. Neither result proves that
a fed, energy-conserving body remains stable while it considers and pays for new organs.

## Protocol

The seed-7 integration smoke used the same 16-cell, 400-update Tiny Shakespeare budget
as the earlier preflights:

- S9 defaults: external gain 0.05, transport 0.50, quiescence 0.01, full activity 0.05;
- S6 policy: structural phase every 20 updates, two confirmations, 40-window balanced
  ABBA trials, and growth cost 0.01;
- no fast efficacy; all body parameters, state, energy, reward, and backprop remained
  live in both traffic arms;
- held-out validation every 50 updates with decision-aligned survival analysis.

## Result

The organism completed with full reachability, final/best BPC 4.17593, held-out mean
energy 0.97758, viability 1.0, and quiescent fraction zero. Mean external input was
0.08131 energy units per scored tick, spending was 0.07584, and transport drift was
2.2e-8.

All three complete candidate trials were rejected:

| Trial | Candidate advantage | Decision | Body survival |
|---:|---:|---|---|
| 1 | -0.05410 | reject | pass |
| 2 | -0.04067 | reject | pending follow-up |
| 3 | -0.00465 | reject | pass |

No anatomy or growth energy changed. The two trials with isolated follow-up validation
both survived and improved. Relative to earlier seed-7 smokes:

- integrated metabolism + exploration: 4.17593 BPC;
- metabolism without structural probes: 4.11596 BPC;
- exploratory anatomy with energy held at one: 4.32248 BPC.

## Interpretation

The mechanisms compose safely: exploratory traffic did not starve or destabilize the
body, and the live selector refused every non-beneficial candidate. Structural probing
cost 0.05998 BPC relative to the metabolism-only smoke at this endpoint, so integration
does not yet establish a morphology benefit. It does show that the next full experiment
can use a genuinely fed organism rather than assuming a fixed-energy body is an adequate
proxy for organ survival.
