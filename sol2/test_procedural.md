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
- Tiny full and organ-only true-attachment branches preserve inactive A parameters,
  keep the frozen organ-only substrate exact, checkpoint adaptation, and exercise the
  removal/reattachment path.
- Split organ-only execution resumes to the same records, model tensors, and living
  state as uninterrupted interval execution.

## Contracts

- Tests use temporary output directories and leave the repository clean.
- The smoke test is mechanical only; two updates cannot support a capability claim.
- This gate complements `test_sol2.py`; it does not replace the kernel invariants.
