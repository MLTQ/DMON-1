# train.py

## Purpose
Continuous multi-stream online trainer. Truncated BPTT: sum CE over `truncate_every`, one backward, detach state (never reset). Optional parameter-budget-matched GRU baseline on identical protocol.

## Components

### `_train_online`
- **Does**: Shared loop for creature and GRU so comparison is fair

### `eval_window`
- **Does**: Single-stream eval from a mid-corpus start (fresh state — eval only)

### `match_hidden_for_budget` (via baselines)
- **Does**: Size the GRU so param count ≈ creature

## Notes
- Detach ≠ reset.
- Prefer `--device cuda` on Aine; `CUDA_VISIBLE_DEVICES=0` selects the 4090 (CUDA fastest-first).
