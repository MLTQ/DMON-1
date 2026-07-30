# model.py

## Purpose
Streaming character organism: sparse directed field, mirror memory, eligibility, reverse vector credit, fast edge efficacy.

## Components

### `CreatureState`
- **Does**: Carries h, eligibility, edge eligibility, fast weights, reverse credit, reward baseline, mirror cursor across BPTT cuts
- **Rationale**: Detach ≠ reset (SOL / ARCHITECTURE)

### `StreamingCreature`
- **Does**: Embed → mirror write → dendrite microsteps with credit drive → readout; `observe_prediction` for delayed reward + decoder correction
- **Interacts with**: `DendriteGraph`, `SharedGRURule`, `train.py`, `evaluate.py`, `structure.py`

### Credit path
1. Eligibility tags participation each microstep
2. Surprise vs EMA baseline → signed reward for next tick
3. Decoder residual maps through readout transpose into output cells
4. Credit transports sourceward via `message_value` transpose × dendrite coefficients
5. Fast weights update when reward meets edge eligibility

### Readout (`TrainConfig.readout_mode`)
- `mean` — pool output cells then `Linear(H→V)` (default)
- `concat` — flatten outputs then `Linear(O·H→V)`
- `attn` — learnable query over output cells then project

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Trainer | `forward_chunk(tokens, targets, state) → logits, state, loss` | Signature |
| Eval | `step` + `observe_prediction` + `initial_state` | State fields |
