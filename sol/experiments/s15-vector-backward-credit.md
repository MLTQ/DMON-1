# S15: Decoder-shaped vector backward credit

## Question

Does a channel-shaped correction from the output organ provide more reliable
cross-window credit than broadcasting one reward scalar or transporting that scalar
backward through the connectome?

## Mechanism

After each observed next-character target, compute the detached corrective decoder
signal

`W_readout^T(one_hot(target) - softmax(logits))`.

Split that vector across output cells, smoothly bound it, and retain it in the live
field state. On later ticks:

- each target transports its credit to named source cells through the signed coefficient
  of the same axon used in the forward pass;
- transport applies the transpose of the learned `message_value` transform, preserving
  hidden-state channel direction;
- credit decays but persists across characters and truncated-BPTT boundaries;
- the shared transition rule receives credit only where it aligns channel-by-channel
  with a cell's event eligibility.

The mechanism introduces no trainable parameters. Existing direct scalar reward,
scalar reverse reward, exact within-window BPTT, and one-shot fast-synapse reward remain
available as explicit controls. Older checkpoints initialize the new state to zero and
the new gain defaults to zero.

## Protected contracts

Tests require:

- the seed to equal the decoder-shaped corrective projection and occupy only output
  cells;
- exact signed sourceward transport through the transpose of `message_value`;
- channel-specific interaction with eligibility;
- no target-free tick to create credit from zero;
- equal parameter count with the disabled control;
- exact checkpoint resume, safe old-checkpoint upgrade, aligned cell-state shuffle, and
  independent training/evaluation/local-bridge telemetry.

## Reduced-scale protocol

Use the healthy RTX 4090 while the RTX 2070S is under investigation. Run Tiny
Shakespeare with 16 cells, 16 channels, four dendrites, two message steps, four lanes,
16-character windows, 800 updates, energy held at one, and fast efficacy disabled.

Compare seeds 7, 13, and 21:

1. direct scalar reward `0.25`;
2. direct scalar reward `0.25` plus scalar reverse credit `0.25`;
3. direct scalar reward `0.25` plus decoder-shaped credit, calibrated from gains
   `0.1`, `0.25`, and `0.5` on seed 7;
4. decoder-shaped credit alone at the selected gain.

Use identical topology seed, stream order, optimizer, validation anchors, and update
schedule within each seed. Report best/final BPC, all-aligned and post-boundary paired
deltas, stability, state ablations, and live credit magnitude.

## Gate

Proceed to credit-guided axon/dendrite turnover if the mechanism:

1. remains bounded, resume-safe, and causally active outside the output cells;
2. matches or improves the direct scalar control across seeds without relying on one
   favorable endpoint;
3. is at least as consistent as scalar reverse credit in aligned validation;
4. retains strong reset and shuffled-state penalties.

If no gain passes, use the diagnostics to revise the seed/transport normalization before
allowing structural growth to consume this signal.
