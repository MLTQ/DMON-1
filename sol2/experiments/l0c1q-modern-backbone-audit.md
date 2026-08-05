# L0-C1q: modern frozen-language-organ audit

Status: protocol frozen 2026-08-05 before either candidate is loaded by DMON.

## Question

Which currently available small language backbone is the best *organ* for the next
before-mode experiment? Generic chat benchmark quality is insufficient: DMON needs an
interface that is frozen, differentiable with respect to continuous inputs, sensitive
to small prefix interventions, faithful to passage-visible temporary bindings, and
small enough to leave useful 4090 memory for a growing organism.

## Candidates

- `Qwen/Qwen3.5-2B-Base`: foundation checkpoint, 2,048-wide text state and 24 hybrid
  layers. It directly tests whether turn-taking and intent can eventually come from
  the organism rather than instruction tuning.
- `Qwen/Qwen3.5-9B`: post-trained checkpoint, 4,096-wide text state and 32 hybrid
  layers. It tests the same model family with stronger learned language behavior.
- `NousResearch/Meta-Llama-3-8B-Instruct` remains the historical measurement, not the
  presumed winner.

Both Qwen candidates are already present in the remote Hugging Face cache. No model
download or 2070S use is licensed by this audit.

## Fixed measurements

Run each Qwen candidate separately in BF16 on physical RTX 4090 UUID
`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`:

1. Load through `AutoModelForCausalLM` and the generic frozen adapter.
2. Confirm exposed input/output embeddings, language width, vocabulary size, frozen
   parameters, and native-final-hidden/output-head parity.
3. Backpropagate one anchored continuous-prefix label loss. Require a finite nonzero
   prefix gradient and zero backbone gradients.
4. Score passage-visible temporary counterfactual bindings across the meta-training
   questions. Report label accuracy, correct-label probability/margin, and paired
   full-logit separation; do not infer model fitness from one prompt.
5. Measure the change caused by a deterministic small continuous-prefix perturbation,
   wall time, and peak CUDA allocation/reservation.

## Decision rule

A candidate is technically admissible only if it passes adapter parity, frozen-gradient,
finite-prefix-gradient, memory, and intervention-sensitivity checks. Among admissible
candidates, prefer reliable temporary-binding likelihood and lower resource cost.
Post-trained chat quality is a secondary benefit; base-model controllability is a
scientific advantage if it does not destroy the language interface.

If neither candidate produces a label-consistent passage-visible teacher, do not call
the family unsuitable. Separate the two roles: keep the chosen language organ and
construct a deterministic label-consistent dense target, or use a stronger teacher
only during training. That becomes a new frozen treatment rather than a silent change
to C1p.
