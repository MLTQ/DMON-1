# `wiki_causal_contrast.py`

## Purpose

Defines the L0-C1l training-only objective that rewards answer improvement caused by
the matching passage at the frozen LLM's final decision logits.

## Components

### `LogitResult`

- **Does**: Describes the minimal target-label and live-logit interface consumed by
  the objective.
- **Interacts with**: `WikiEpisodeResult` in `wiki_memory.py`.

### `passage_causal_contrast_loss`

- **Does**: Hinge-rewards each exposed branch for raising its target log probability
  over matched no-exposure and incompatible-passage baselines.
- **Interacts with**: The paired training loop in `wiki_memory_train.py`.
- **Rationale**: Detached baselines prevent success by degrading control arms, so
  optimizer credit can only improve the passage-matched branch.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `wiki_memory_train.py` | Four ordered advantages: left/right over no exposure, then left/right over incompatible passage | Order, detach policy, return shape |
| `test_wiki_memory.py` | No gradient reaches the no-exposure baseline and satisfied margins produce zero loss | Gradient ownership, hinge semantics |

The objective is training-only. It does not add parameters, write organism state, or
run during development, held-out evaluation, or inference.
