# S1-P: procedural transfer across changing interfaces

Status: architecture and CPU gate complete; interpretation amended 2026-08-03 before
result-bearing runs to distinguish adaptation from lesion tolerance.

## Goal

Test whether SOL2 can preserve and reuse a *way of computing* when its sensors and
effectors change. This deliberately demotes language BPC from project objective to a
separate language-organ capability diagnostic.

Each episode supplies an initial latent value and a short program composed from four
transformations. Only the final answer is scored. Values, programs, and answers are
freshly generated, so memorizing facts cannot solve the task.

## Exact fork design

1. Acquire the canonical procedure through interface A.
   Acquisition begins with one transformation and progressively admits lengths two,
   three, and four; all scored forks begin immediately at the full training range.
2. Snapshot the exact weights, optimizer moments, and living recurrent state.
3. Fork three matched continuations:
   - A-control: unchanged interface and procedure.
   - B-interface: new state, operation, and answer symbols; same procedure.
   - C-procedure: same symbols and primitive meanings; compose the program in reverse
     rather than forward order.
   - B-scratch: learn B from the exact pre-acquisition initialization on the same B
     examples, providing a transfer baseline for the unrestricted organism.
   - SOL2-only B/C organ forks: repeat both changes while freezing graph, tissue rules,
     and private cell expression parameters; the recurrent core remains operational,
     while only sensory and output organs learn.
4. Compare early and late answer accuracy, answer cross-entropy, longer-program
   extrapolation, per-length acquisition/adaptation curves, and return-to-A retention.

The primary organism capability is useful unrestricted B-interface adaptation. Its
advantage over B-scratch measures transfer from prior procedural acquisition. The
organ-only B branch diagnoses how much reuse is possible without internal parameter
change; it is not a demand that biological neurology remain immutable. The matched
organ-only C branch tests whether interface organs can fake a new procedure. A measures
ordinary continued improvement. Raw B-versus-C speed is descriptive because arbitrary
symbol remapping and reversed execution need not have equal intrinsic difficulty.

## Organism evidence

Task success is reported beside reset, whole-internal, compute-only and relay-only
freezes, within-tissue state shuffle, private-expression shuffle, topology rewire, and
sensory-memory-zero interventions. These are acute lesions evaluated after learning;
the organism is never trained to withstand them. Degradation establishes that grey-like
computation, white-like routing, private identity, or topology carries the capability.
Its magnitude is not a robustness objective and is meaningful only beside normal
performance. Plastic recovery after adding or removing organs is a later experiment in
which whole-organism learning remains enabled.

## Memory interpretation

Neural Turing Machines and Differentiable Neural Computers establish useful read/write,
content-addressing, allocation, and temporal-link primitives. For SOL2, their most
interesting payload is not an isolated fact but a compact controller configuration:
which cells, timescales, routes, and organs implement a strategy.

Fable F9b did not test that strongly. Its stored value was a projected context vector
that became one channel-wise gain broadcast across every mutable cell; it did not save
or restore a differentiated per-cell expression profile. Its read-key alignment also
had no dense bootstrap curriculum. That negative result remains evidence about that
mechanism, not a rejection of procedural-mode memory.

The intended follow-up is a low-rank dynamic mode state: per-lane coefficients combine
with learned cell-by-channel bases to modulate target and time-constant expression.
DNC-like slots may later store those compact coefficients and transition links. The
read path must be live and densely rewarded before sparse lifetime retrieval is tested.

## Liquid-network interpretation

Liquid Time-Constant networks contribute input-dependent continuous dynamics and
bounded state; CfC contributes efficient closed-form time dependence without an ODE
solver. SOL2 already has leaky integration toward bounded targets, but Δt is implicit
and its telemetry does not expose differentiated per-cell timescales.

Fable F3 fixed Δt to one, shared the time-constant generator across cells, and left the
surrounding attention/readout scale paths unbounded. It therefore did not test the
combination relevant here. A future SOL2 treatment should use private baseline
timescales plus bounded context modulation and the composable update
`alpha = 1 - exp(-delta_t / tau)`. Irregular organ latency—not character BPC—is the
appropriate test.

Primary references:

- Graves, Wayne & Danihelka, *Neural Turing Machines*, 2014:
  https://arxiv.org/abs/1410.5401
- Graves et al., *Hybrid computing using a neural network with dynamic external
  memory*, 2016: https://doi.org/10.1038/nature20101
- Hasani et al., *Liquid Time-constant Networks*, 2020:
  https://arxiv.org/abs/2006.04439
- Hasani et al., *Closed-form continuous-time neural networks*, 2022:
  https://doi.org/10.1038/s42256-022-00556-7

## Promotion boundary

Do not add an external memory organ merely because the benchmark is hard. First
establish that bound-4 SOL2 can learn the dense procedure and that B/C forks separate.
Then test dynamic mode state without external slots. Add DNC-style storage only when a
mode exists that is useful while active and demonstrably worth recalling later.

## First GPU protocol and decision rules

The first result-bearing run uses the promoted identity-conditioned, operator-bound-4
creature at seed 7. It receives 4,000 acquisition updates and 2,000 updates in each
fork, batch size 24, lengths one through four, 64 evaluation batches, and the existing
constant 1e-3 optimizer schedule with 200 warmup updates. A GRU receives the same data
and update budget as a descriptive learnability/adaptation control. The full creature
is the scientific subject; the GRU is not a model-selection winner.

Outcomes are interpreted in this order:

1. **Dense learnability gate:** the creature must reach at least 50% exact accuracy on
   held-out length-four A programs, with no rejected updates. Otherwise tune training
   or the dense task before adding memory or liquid machinery.
2. **Substrate-use gate:** freezing internal cells or rewiring topology must each lower
   length-four A accuracy by at least 10 percentage points. Otherwise task performance
   has not established a procedural internal circuit.
3. **Whole-organism adaptation gate:** unrestricted B must reach at least 50% held-out
   length-four accuracy. B is compared with its matched from-scratch branch; an early
   accuracy advantage of at least 10 percentage points is strong evidence that prior
   acquisition accelerates adaptation, while useful B without that advantage establishes
   adaptability but not positive transfer.
4. **Modular-reuse diagnostic:** B organs-only above 22.5% length-four accuracy is
   evidence that an operational but parameter-frozen core can be reconnected. Failure
   does not negate successful whole-organism adaptation; it locates adaptation inside
   the integrated substrate rather than only at its interface.
5. **Specificity control:** B and C are never ranked by raw speed alone. We report the
   full-versus-organs-only gap within each regime; C checks whether organ-only learning
   can merely route around an unchanged procedure.
6. **Retention and extrapolation:** return-to-A and length-eight results are reported
   as separate capabilities, not used to retroactively redefine the first three gates.

A single seed can promote an architectural ablation but cannot establish a general
capability claim. If gates 1 through 3 pass, repeat seeds 13 and 21 before adding
complexity. If the substrate is causal and B adapts but shows neither a scratch advantage
nor organ-only reuse, the next experiment is low-rank dynamic mode state. DNC-style
slots remain deferred until that active mode is causally beneficial. Delta-aware liquid
timescales are tested separately on irregular-latency variants, rather than bundled into
this attribution run.
