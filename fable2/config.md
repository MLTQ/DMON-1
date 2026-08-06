# config.py

## Purpose

One checkpointable dataclass for the whole Broca-graft treatment: organism geometry,
graft shape, and training knobs. Deliberately small — the C1 line accumulated an
eleven-term loss graveyard (`sol2/wiki_memory_train.py:62-116`) and fable2 does not
inherit it.

## Components

- `Fable2Config` — validated dataclass with JSON round-trip.
- `organism_config()` — derives the `Sol2Config` for the imported kernel. The
  organism's own token organ is unused (the Broca organ senses frozen features
  directly), so `vocab_size=2` keeps it minimal without a dead branch.

## Decisions

- **Memory must span the exposure**: `validate()` rejects
  `max_memory_tokens > n_memory`. The FIFO otherwise silently retains only the tail
  of a passage and every "memory" claim becomes a claim about the last few tokens.
- **Depth fractions, not indices**: depths are preregistered as fractions of the
  backbone's layer count and resolved at attach, so the same config is meaningful
  across backbones; collisions raise instead of silently merging sites.
- **Two loss knobs only** (`contrast_margin`, `task_weight`, default task off). Any
  new loss term requires a preregistration, not a flag.

## Contracts

- `validate()` raises on: empty depth list, duplicate/out-of-range fractions,
  nonpositive gains/rates, memory shorter than exposure. Round-trip
  `from_json(to_json(cfg)) == cfg` is contract-tested.
