# config.py

## Purpose
Single dataclass for model geometry and online-training hyperparameters. JSON-serializable for run logs.

## Components

### `TrainConfig`
- **Does**: Cell counts, dendrite width, attention flag, multi-stream batch size, BPTT truncate, lr
- **Interacts with**: `model.py`, `train.py`, `graph.py` (via model)

### Defaults (post-scale)
- 128 cells, h=128, 12 dendrites, batch 16, attention on — CPU-smoke still works if you override down

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `StreamingCreature` | geometry fields + `use_attention` | Rename/remove fields |
| `train.py` | `batch_size`, `truncate_every`, `lr`, `weight_decay` | Same |
