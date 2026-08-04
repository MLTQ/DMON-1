# `living_language.py`

## Purpose

Closes the loop between a frozen language backbone and one persistent SOL2 organism.
The backbone supplies fluent language machinery; the organism supplies continuous
state, identity, development, and adaptive control.

## Components

### `LanguageStep`

Carries next-token logits, emitted control tokens, continuing organism state, and
optional SOL2 health telemetry.

### `LivingLanguageSystem`

- Extracts the newest contextual language feature from the frozen backbone.
- Evolves SOL2 exactly once for that perceived token.
- Sends SOL2's continuous output-tissue controls back through the frozen backbone.
- Computes teacher-forced language loss without unfreezing the backbone.
- Greedily generates while feeding both prompt tokens and newly generated tokens into
  the same continuing organism state.
- Absorbs exposure sequences without allocating vocabulary logits, and scores only the
  final next-token distribution for classification-style memory probes.
- Accepts already-computed frozen feature sequences for matched causal evaluations
  that reuse the exact same language representation across organism interventions.
- Exposes a per-sequence memory write gate so a read/query phase can retain an earlier
  exposure while continuing to process its own tokens.
- Supports masked teacher forcing so question/prompt tokens can evolve the organism
  without receiving a language loss.

### `graft_language_backbone`

Builds a deterministic continuous organ sized to a backbone's hidden width, attaches
it to an existing organism, and returns the closed-loop system without consuming the
caller's random stream.

## Decisions

- `advance` treats only the final context token as new sensory evidence. Callers own
  the language context, while SOL2 owns lifetime state; this prevents accidental
  replay of an entire transcript into the organism on every turn.
- Teacher forcing computes one causal frozen-backbone feature sequence, evolves SOL2
  sequentially over those features, and decodes with one control bank per position.
  It therefore avoids the quadratic repeated-prefix execution of the reference loop.
- `observe_sequence` deliberately omits the frozen vocabulary head. This keeps passage
  exposure memory-efficient and prevents an unused text loss from becoming the
  learning objective.
- `score_next_after_sequence` evolves over every prompt token but decodes only the last
  position, which is the exact surface needed by multiple-choice memory experiments.
- Stream-memory writing remains enabled by default for ordinary language flow and is
  closed only by protocols that explicitly distinguish write and query phases.
- Feature-level observation/scoring is semantically identical to token wrappers. It
  avoids repeating an 8B-parameter frozen forward for every causal arm while retaining
  separate cellular transitions and controls.
- Frozen features are computed once per transition and reused for controlled decoding;
  there is no mandatory second backbone pass.
- Sensory features are explicitly converted to the continuing cellular state's device
  and dtype. Return controls are converted to the backbone feature dtype by the adapter,
  so BF16 language machinery and FP32 organism dynamics remain a deliberate boundary.
- `control_scale=0` provides an exact language-floor intervention while allowing the
  organism to continue perceiving and evolving. Reset and tissue-lesion controls are
  expressed through the initial/frozen-state arguments, not separate model variants.
- The system rejects any backbone with trainable parameters. Training must improve the
  creature or its detachable interface, not quietly fine-tune Broca's area.
- The organism configuration's legacy vocabulary size does not need to match the
  backbone vocabulary. The continuous graft owns no vocabulary decoder.

## Contracts

- A continuous organ is attached before system construction.
- One `advance` call corresponds to one newly observed token and one SOL2 transition.
- Disabling memory writes freezes only stream-memory contents and cursor advancement;
  sensory input, recurrent evolution, and output remain active.
- Language loss has a gradient path into controls and SOL2 but never into backbone
  parameters.
- Generated tokens are sensory input on the following transition.
- Clearing the LLM context does not itself clear `OrganismState`.
- Grafting is deterministic from its private seed and preserves global RNG state.
- A loss mask changes credit assignment only; every input token still advances the
  same continuing cellular state.
- Cached features must retain `[batch, sequence, backbone_width]` shape and remain
  detached from backbone parameters.
