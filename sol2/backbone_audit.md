# `backbone_audit.py`

## Purpose

Qualifies a frozen language checkpoint as a DMON organ before any optimizer run. The
audit separates generic model reputation from the properties the organism actually
needs: an exposed differentiable input substrate, exact decoding parity, continuous
intervention sensitivity, reliable passage-visible binding, and usable GPU headroom.

## Measurements

- Loads the checkpoint through the same `HuggingFaceFrozenBackbone` used by training.
- Checks the generic final-hidden/output-head path against native model logits.
- Backpropagates correct-label loss through an anchored four-token continuous prefix;
  the prefix must receive finite nonzero credit while the backbone receives no
  gradients.
- Applies a deterministic perturbation at one percent of token-embedding RMS and
  reports full-vocabulary and A/B/C/D response plus forward KL.
- Constructs both compact incompatible epoch-zero bindings for every meta-training
  question, scores each with the passage still visible, retains per-example adverse
  cases, and summarizes label accuracy, probability, margin, and paired full-logit
  separation.
- Records model dimensions, parameter count, wall time, and peak CUDA memory.

## Decisions

- Raw `Answer:` prompts are intentional: this is the interface used by the current
  before-mode training protocol. A later conversational graft may use a checkpoint's
  native chat template, but that is a distinct interface experiment.
- Label accuracy is normalized only across A/B/C/D. Full-vocabulary distributions are
  retained for pair separation and the perturbation KL.
- All twelve meta-training questions contribute two incompatible temporary bindings,
  so a candidate is not accepted or rejected from one convenient prompt.
- The deterministic prefix direction is a local sensitivity probe, not an estimate of
  trainability. The live gradient test is the stronger technical requirement.

## Run

```bash
CUDA_VISIBLE_DEVICES=<4090-uuid> python -m sol2.backbone_audit \
  --model Qwen/Qwen3.5-9B \
  --device cuda:0 --dtype bfloat16 --local-files-only \
  --out data/dmon-l0/l0c1q-backbone-audit/qwen35-9b.json
```
