# `language_smoke.py`

## Purpose

Runs the first result-bearing compatibility and capacity probe against a real frozen
Hugging Face causal language model before any conversational training.

## Flow

1. Load a decoder-only checkpoint and tokenizer lazily through Transformers.
2. Build a moderate SOL2 organism whose unused legacy A port has vocabulary size two.
3. Attach a deterministic continuous language graft sized to the backbone width.
4. Audit the generic exposed output head against native model logits.
5. Run one zero-control prediction and backward pass into the organism.
6. Record parameter counts, loss, gradients, health, latency, and peak CUDA memory.

## Decisions

- The command does not perform an optimizer step. It proves memory capacity and the
  intended gradient boundary without changing either model.
- Model choice is a CLI argument so SOL2 is not coupled to one vendor, license, or
  hidden width.
- `bfloat16` is the default 4090 dtype; models requiring another numeric path declare
  it in the result command.
- Native-logit parity is a hard gate. A checkpoint with head-specific transforms must
  receive a specialized adapter rather than silently changing its baseline.
- Only the final hidden feature and frozen output head participate in the controlled
  path, so no full transformer backward graph is retained.

## Output contract

The JSON result must show:

- zero trainable backbone parameters and zero backbone gradient tensors,
- exact-zero controls at graft initialization,
- a nonzero control-decoder gradient,
- native-logit error within the requested tolerance, and
- measured peak allocated/reserved VRAM on CUDA.

## Example

```bash
.venv/bin/python -m sol2.language_smoke \
  --model /path/to/local/model --device cuda:0 --dtype bfloat16 \
  --out /tmp/dmon-l0-smoke.json --local-files-only
```

