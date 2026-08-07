# corpus.py

## Purpose

Builds the G2 paired-question corpus from the wiki vault at 4.5× the C1 corpus,
with a *behavioral* admission gate: the bare backbone must fail each question
closed-book and pass it open-book. C1's rule ("the model got it wrong") never
verified extractability, so a C1 question could be unanswerable even with the
passage in hand.

## Components

- `strip_note` / `extract_passage` — frontmatter/markdown-stripped prose,
  sentence-trimmed to ≤ `max_passage_tokens` (must fit the organism's memory
  FIFO); short notes are rejected, and the `Designated answer` marker is
  rejected at the source.
- `draft_questions` — frozen Qwen drafts exactly two four-choice questions per
  passage via its chat template (greedy, deterministic); strict JSON parsing and
  structural validation; malformed drafts reject the document.
- `bare_answers_correctly` — fp32 label scoring of the formatted question under
  the training permutation seed, with or without the passage prepended.
- `build_corpus` — deterministic seeded scan over `concepts/`, `entities/`,
  `comparisons/`, `summaries/`; admits documents until the 36/6/12
  (meta_train/development/heldout) quotas fill; writes sol2-schema-1 JSON with
  source sha256 pins and a reject-count report.

## Decisions

- **The generator is the same frozen model being evaluated.** Its knowledge sets
  the closed-book gate honestly: whatever it can already answer is inadmissible
  by construction, so generation bias toward answerable questions is filtered by
  the gate rather than trusted.
- **Split assignment happens after admission**, from a seed-shuffled scan order,
  so no property of a document's content influences which split it lands in.
- Document ids/families derive from the vault-relative path (not the bare stem),
  so same-named notes in different subdirs cannot collide; one family per
  document, which makes the split-leak guard in the sol2 loader trivial and the
  mate constraint ("different family") meaningful.

## Contracts

- Fails loudly if quotas cannot be met, with the reject breakdown.
- Output loads through `sol2.wiki_memory.load_wiki_memory_corpus` (schema,
  split, distinct-choice, and duplicate-id validation) and passes
  `verify_wiki_sources` against the vault.
