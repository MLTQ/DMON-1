# `test_procedural.py`

## Purpose

Provides the deterministic CPU gate for the procedural-transfer task and its exact
branching harness before any GPU experiment is allocated.

## Coverage

- Interface remapping preserves latent operation semantics.
- Procedure changes preserve all external sensor and effector codes.
- Generated programs execute correctly in latent space and remain within vocabulary.
- Forward and reverse procedures receive identical input tokens but produce distinct
  latent answers, confirming that the C fork changes computation rather than interface.
- A tiny SOL2 organism can acquire, fork full and organs-only transfer branches, run
  interventions, and write a complete metrics artifact.
- The interface-from-scratch control is present, and acute lesions separately expose
  compute- and relay-tissue contributions.
- Mastery-gated acquisition writes a complete checkpoint, emits per-neuron telemetry,
  and resumes to the exact next interval without discarding prior evaluations.
- Attaching B preserves global RNG, leaves A logits exact and its state numerically
  identical, keeps A/B parameters disjoint, and restores the same module on reattach.
- Causal utility calibration leaves parameters exact, assigns no utility to dormant
  edges, preserves within-tissue treatment distributions when shuffled, and scales the
  realized optimizer delta exactly for shared rules and private cell rows.
- Directional edge utility, mean-matched uniform protection, cell-local adapter
  protection, and append-only reserve grafting are exercised before GPU work.
- Proximal anchors enforce exact cumulative restoration, preserve new growth rows as
  fully plastic, and expose anchored versus accessible gradient demand.
- Developmental triggers require persistent pressure, honor refractory periods and
  event caps, and resume their decision state exactly.
- Tiny full and organ-only true-attachment branches preserve inactive A parameters,
  keep the frozen organ-only substrate exact, checkpoint adaptation, and exercise the
  removal/reattachment path.
- A tiny consolidated-attachment branch calibrates utility, protects the substrate,
  retains A-organ integrity, checkpoints its graft ledger, and emits utility and
  private-reserve lesions end to end.
- A tiny S1-P4 branch crosses a forced pressure threshold, grows append-only relay
  cells, preserves A-organ integrity, emits grown-cell lesions, and resumes through
  exact geometry replay.
- Synthetic matched artifacts exercise the exact S1-P3 aggregate contrasts and frozen
  success gates without waiting for a GPU result.
- Synthetic S1-P3b artifacts keep capability and measured-allocation conclusions
  separate while exercising adapter and graft recruitment gates.
- The genome-rate selector maximizes worst-organ accuracy and applies the frozen
  higher-B tie-break inside a two-point band.
- Synthetic S1-P4 artifacts keep anchor attribution, simultaneous capability, growth
  benefit, and causal recruitment as independent frozen gates.
- Split organ-only execution resumes to the same records, model tensors, and living
  state as uninterrupted interval execution.

## Contracts

- Tests use temporary output directories and leave the repository clean.
- The smoke test is mechanical only; two updates cannot support a capability claim.
- This gate complements `test_sol2.py`; it does not replace the kernel invariants.
