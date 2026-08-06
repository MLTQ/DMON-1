# `procedural_task.py`

## Purpose

Defines a synthetic task that separates remembering a reusable procedure from learning
the sensor and effector codes through which that procedure is invoked. It supplies the
first SOL2 benchmark whose primary outcome is procedural transfer rather than text BPC.

## Components

### `ProcedureRegime`

- **Does**: Names the input-state, operation, answer, primitive semantics, and execution
  order for one environment.
- **Rationale**: Interface changes and algorithm changes must be independently
  controllable or rapid relearning cannot be interpreted as procedural transfer.

### `ProcedureBatch`

- **Does**: Carries encoded episode tokens and both encoded and latent answers.
- **Interacts with**: `train_phase` and `evaluate_regime` in
  `procedural_benchmark.py`.

### `ProceduralTask`

- **Does**: Samples short programs over four transformations of a finite cyclic state,
  executes them in latent space, and exposes only regime-specific surface symbols.
- **Rationale**: The useful knowledge is how to compose transformations. Arbitrary
  facts are regenerated every episode and cannot be memorized profitably.

### `base_regime`, `remapped_interface`, `changed_procedure`

- **Does**: Construct the acquisition environment, a new surface interface, and a true
  procedural change from forward to reverse-order composition.
- **Rationale**: Interface remapping preserves the procedure; procedure change
  preserves the interface. Their matched adaptation curves distinguish reusable
  internal computation from generic continued learning.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `procedural_benchmark.py` | `tokens` end in `QUERY`; final logits predict `answer_tokens` | Episode order or answer position |
| `test_procedural.py` | Interface remaps preserve semantics and procedure shifts preserve surface codes | Regime construction semantics |
| SOL2 models | All tokens are within `vocab_size` | Token ranges or vocabulary calculation |

## Notes

- Training lengths and extrapolation lengths are selected by the benchmark runner.
- No loss is assigned to scaffolding tokens; the task cannot be won by improving
  generic compression while ignoring the procedural answer.
