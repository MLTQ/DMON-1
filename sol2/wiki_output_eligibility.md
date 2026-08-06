# `wiki_output_eligibility.py`

## Purpose

Defines L0-C1i's training-only, activity-gated differential credit at the measured
relay-to-output bottleneck. It changes optimizer credit without adding an inference
module, state write, or LLM path.

## Components

### `TransportStateResult`

- Structural contract for paired results exposing target labels and live relay/output
  tissue states.

### `eligibility_gated_transport_loss`

- Forms an answer axis from fixed orthonormal codes.
- Gives identical paired output endpoints non-cancelling opposing gradients.
- Gates that credit by detached paired relay separation and reports projection,
  tissue separation, transport ratio, and gate strength.

## Contracts

- Exactly two branches share a question and have incompatible target labels.
- Relay eligibility is detached; the auxiliary cannot reduce loss by inflating its
  own presynaptic gate.
- Zero relay difference produces zero auxiliary loss and output gradient.
- Swapping branch order leaves the scalar objective unchanged.
- The function reads tissue state only and never writes an answer or teaching signal
  into the organism.
- Development, held-out evaluation, and inference never call this function.
