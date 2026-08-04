# `language_backbone.py`

## Purpose

Defines the narrow boundary that makes language a replaceable DMON organ and provides
a deterministic toy implementation for CPU integration tests. The production backend
can later wrap an open-weight pretrained model without changing SOL2.

## Components

### `FrozenLanguageBackbone`

A structural protocol exposing only:

- `width` and `vocab_size` metadata,
- contextual feature extraction for sensory input, and
- language logits under optional continuous control tokens, including decoding from
  already-computed frozen features and one distinct control bank per sequence position.

### `ToyFrozenLanguageBackbone`

- Uses a small embedding, GRU, and decoder as deterministic frozen machinery.
- Exposes contextual recurrent features to the organism.
- Lets each frozen contextual feature attend over continuous controls, then applies an
  additive residual immediately before the frozen decoder.
- Retains gradient flow from language loss into controls while every backbone
  parameter has `requires_grad=False`.

### `HuggingFaceFrozenBackbone`

- Wraps a decoder-only `AutoModelForCausalLM` without making Transformers a core
  SOL2 dependency.
- Obtains final contextual features through `output_hidden_states=True`.
- Lets each frozen feature attend to the organism's control tokens, adds the resulting
  residual, and decodes it through the model's exposed frozen output embeddings.
- Loads current Hugging Face checkpoints lazily through `from_pretrained`.
- Measures generic-head versus native-logit parity with `native_logit_error`; models
  with extra architecture-specific logit transforms require a specialized adapter.

## Decisions

- The toy backend is a contract fixture, not a claim that a GRU is an appropriate
  scientific baseline for emergent language behavior.
- `controls=None` is the canonical frozen baseline. Exact-zero controls traverse the
  controlled path but must reproduce it exactly.
- Decoding from existing features avoids a duplicate frozen forward pass. Applying the
  residual immediately before the language head also avoids storing a full frozen
  transformer graph merely to deliver gradients into SOL2 controls.
- Position-specific control banks allow a whole causal training sequence to use one
  frozen backbone pass. Each hidden position already encodes only its causal prefix;
  SOL2 still evolves sequentially and supplies a separate control for that position.
- Sensory extraction is detached because the language model is an environmental organ,
  not part of the organism's trainable persistent state.
- The first production adapter should obey this protocol while choosing a
  model-specific injection point that preserves the zero-control identity invariant.
- Final-hidden residual injection is the initial production point because it keeps the
  backbone forward under `no_grad`, avoids a second transformer pass, and still gives
  the organism a differentiable path through the frozen language head. Deeper
  zero-residual injection remains an empirical extension, not a prerequisite.

## Contracts

- Input IDs have shape `[batch, sequence]` and dtype `torch.long`.
- `encode` returns detached features shaped `[batch, sequence, width]`.
- Controls have shape `[batch, control_tokens, width]`.
- Training-sequence controls may instead have shape
  `[batch, sequence, control_tokens, width]`.
- Adapters convert controls to the frozen feature device and dtype before attention.
- Frozen parameters never receive gradients or optimizer updates.
- Language loss remains differentiable with respect to non-detached controls.
- A production checkpoint is accepted only after `native_logit_error` is within its
  numeric tolerance; otherwise its language head needs a model-specific wrapper.
