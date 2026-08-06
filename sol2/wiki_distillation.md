# `wiki_distillation.py`

## Purpose

Provides dense frozen-teacher credit, passage-effect targets, and an explicit energy
price for passage-conditioned language control.

## Components

### `full_vocab_reverse_kl`

- **Does**: Computes temperature-scaled `KL(student || teacher)` over every vocabulary
  entry and averages any leading dimensions.
- **Interacts with**: Student vocabulary logits retained by `run_wiki_memory_episode`
  and detached passage-visible logits built by `wiki_memory_train.py`.
- **Rationale**: Full-distribution credit is denser and lower variance than reducing
  the teacher to one sampled token or four answer labels.

### `control_delta_energy`

- **Does**: Returns mean squared amplitude of an effective control bank.
- **Interacts with**: Reference-centered controls emitted by `LivingLanguageSystem`.
- **Rationale**: Makes gratuitous intervention costly while allowing a small
  content-specific delta to earn its keep through task and teacher losses.

### `passage_effect_target_logits`

- **Does**: Adds detached `teacher(passage + question) - teacher(question)` logits to
  the detached matched student zero-prefix baseline.
- **Interacts with**: Passage-visible/question-only teacher helpers and the no-exposure
  student arm in `wiki_memory_train.py`.
- **Rationale**: Cancels the teacher's general language prior so dense credit cannot be
  satisfied merely by suppressing organism control.

### `full_vocab_effect_reverse_kl`

- **Does**: Applies full-vocabulary reverse KL from the live passage-conditioned
  student to the passage-effect target.
- **Rationale**: Preserves dense frozen-language credit while allowing gradients only
  through the conditioned student path.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `wiki_memory_train.py` | Teacher is detached and only student receives KL gradients | KL direction or detach semantics |
| `test_wiki_memory.py` | Matching distributions have exact-zero KL within floating tolerance | Reduction or temperature scaling |
| C1p protocol | Energy is computed on expressed deltas, not raw common controls | Input meaning |
| C1s protocol | Common teacher priors cancel within floating tolerance and all baseline/teacher logits detach | Target construction or detach semantics |

## Notes

This module contains training credit only. Neither teacher logits nor target labels
enter organism state or the inference path.
