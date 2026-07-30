# cell.py

## Purpose

The single shared cell rule: one `nn.GRUCell(2H, H)` applied to every mutable
cell of every lane in parallel.

## Components

- `SharedRule` — input is `[messages, drive]` (2H); biases zero-init.

## Decisions

- **2H input, not grok's 3H.** The third slice was `credit_drive`, identically
  zero under no-credit yet still counted — ~9% of the scale arm's parameters
  were dead while the matched GRU was sized against the inflated count (grok
  debt #22). Deleting the slice makes the parameter match honest.
- GRU chosen (over vanilla RNN/LSTM cell) because its output is a convex
  combination of bounded candidates: cell state cannot exceed ±1, so the
  substrate itself cannot blow up — only gradients can (guarded in train.py).

## Contracts

- `forward(h, messages, drive)` shapes all `[B, Nt, H]`; returns `[B, Nt, H]`
  bounded to (−1, 1).
