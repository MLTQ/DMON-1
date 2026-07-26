# `benchmark.py`

## Purpose

Runs reproducible Tiny Shakespeare experiments for SOL, a parameter-matched GRU, and a
parameter-matched causal transformer on identical continuous train/validation streams,
with checkpointing and append-only metrics.

## Components

### `_run_sol`
- **Does**: Trains or resumes SOL, logs behavioral and organism metrics, evaluates state
  ablations, and atomically promotes the best held-out checkpoint.
- **Interacts with**: `ContinuousTrainer`, `checkpoint.py`, and `evaluate.py`.

### Scientific controls
- **Does**: `--freeze-edges` trains an otherwise identical organism without modifying
  its directed edge weights or biases; `--no-metabolism` holds energy at one.
- **Does**: `--no-reward` sets reward gain to zero while leaving the event trace
  computation intact.
- **Rationale**: Learned morphology and metabolic modulation must earn their claims
  against controls rather than ride on the shared cell rule; delayed eligibility
  feedback must likewise improve prediction rather than merely sound plausible.

### Topology report
- **Does**: Records sensory reachability, output reachability, self-edge count, and mean
  sensory-to-output distance in every SOL summary.
- **Interacts with**: `analyze_topology` in `topology.py`.

### `_run_gru`
- **Does**: Trains a parameter-matched GRU with the same batch lanes, chunk length,
  optimizer family, token budget, held-out split, and evaluation window.
- **Interacts with**: `CharacterGRU` in `baselines.py`.

### `_run_transformer`
- **Does**: Trains a parameter-matched causal transformer with a rolling context on the
  same lanes, token budget, split, and evaluation window.
- **Interacts with**: `CausalCharacterTransformer` in `baselines.py`.

### CLI
- **Does**: Records a manifest, JSONL history, summary, and local checkpoint beneath an
  explicit output directory.
- **Rationale**: Large tensors stay out of Git while small evidence files remain easy to
  inspect and copy home.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dual-GPU launch | `--model sol|gru|transformer` lets each GPU run one independent job | CLI names |
| Local UI backend | `best.pt` is a complete SOL checkpoint with evaluation metadata | Checkpoint selection |
| Scientific report | Both models see the first 90% for training and final 10% for validation | Split semantics |

## Notes

- `--updates` is a final update number when resuming SOL, not an additional count.
- The 4090 should run SOL; the 2070S is better used for the GRU and ablations than for
  mismatched synchronous data parallelism.
