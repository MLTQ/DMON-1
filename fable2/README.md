# fable2/ — the Broca graft (multi-depth differentiated language organ)

The fourth experimental line. Written 2026-08-06, immediately after the SOL2 program
review and the C1aa stop. Its charge is the one experiment that review licenses: prove
that a persistent SOL2 organism can **meaningfully use a frozen LLM as a Broca's
area** — carry information in living tissue across context erasure and change frozen-
model behavior *for the right reason* — through a differentiated, bounded,
output-tissue-only language organ injecting at several transformer depths.

## The evidence this line synthesizes

Every claim has a recorded run behind it. Sources: `sol2/experiments/`,
`sol2/experiments/program-review-2026-08-06.md`, `fable/experiments/`.

1. **The substrate carries skills causally.** S1-P/P2/P3b: internal, compute, relay,
   and topology lesions cost 47–86 accuracy points on a mastered deep procedure;
   private per-cell adapters became jointly causal; a mastered organism learned a new
   organ ~1,000 updates faster than scratch. The organism is real. This line does not
   need to re-prove it.
2. **Exact same-coordinate value transport is dead.** C1g→C1aa: coherent addressing
   learns, Qwen can be steered, but copying the organism's own stored answer-value
   through one final-token recall never survived as usable behavior. Fixed target
   scale (C1aa), isolated local training, 4× tissue, and longer duration are all
   individually falsified. **Do not relitigate any of them here.**
3. **The single pre-transformer prefix is functional but poorly conditioned** (C1n–C1x):
   it learns static interventions, harmful passage effects, or silence. The successful
   direction (C1z) was *paired differential* credit — correct passage beat wrong
   passage in dev and held-out mean loss — with an effect too small to survive.
4. **The C1aa close-out is this line's design mandate**: *allow learned
   representational transformation and give the language organ several bounded,
   output-tissue-only injection sites, while retaining paired behavioral and lesion
   attribution.*
5. **Fable's structural lessons hold**: differentiation, not capacity, is the binding
   constraint (F1/F8); stability lives in bounded shared-scale paths (F3+F8 → SOL2's
   damp-only operators); matched schedules go to every arm; contracts run before GPU.

## What fable2 is

**The organism is imported, not rewritten.** `sol2` remains the validated kernel —
typed tissues, bounded operators, private expression, output-as-sink anatomy, and its
CPU contract suite all carry over by import. Rewriting a tested kernel to relitigate
solved problems is how sessions get lost. fable2 contains only the *new treatment
surface*:

- **`BrocaOrgan`** — one sensor (frozen-LLM feature → bounded organism drive), one
  attentive trunk that reads **only output tissue**, and K per-depth zero-initialized
  low-rank heads. Each head emits a small bank of bounded control vectors for one
  preregistered frozen-transformer depth. Representational transformation is learned;
  nothing is copied coordinate-for-coordinate.
- **`MultiDepthBackbone`** — a frozen HF causal LM with additive cross-attention
  control residuals injected at the selected depths via forward hooks. **Scaffold-free
  by construction**: zero controls produce bitwise-identical logits to the bare model,
  so the no-control floor *is* the bare-LM floor (the C1n prefix could not say this).
- **A paired causal assay with honest instruments.** The five instrument defects found
  in the 2026-08-06 review of the C1 line are fixed structurally here, not by
  discipline: fp32 label scoring; a true bare floor; control arms that are either
  genuinely distinct computations or asserted-equal invariants (never quoted as
  independent corroboration); exposure is the passage prose itself (no
  "Designated answer:" cards — the C1 training/eval distribution shift); every
  denominator guard raises instead of returning a plausible number.

Deleted (they live in `sol2/` if ever needed again): coherent recall residuals,
same-coordinate value/transport/semantic credits, reference-centered controls,
compact binding cards, the eleven-term loss graveyard. The fable2 loss is paired
causal contrast plus an optional task term. Two knobs, both preregistered.

## Experiments

### G0 — interface audit (before any training)

On frozen Qwen3.5-4B (the admitted backbone, l0c1q): attach an untrained graft and
verify (a) exact zero-init no-op against the bare model, (b) nondegenerate label-logit
sensitivity to bounded random controls at **every** selected depth individually,
(c) finite nonzero gradients reaching every depth head and every organism tissue
through the paired loss. Gates in `experiments/g0-broca-interface-audit.md`. No
training happens until G0 passes — the one-second check before the forty-minute sweep.

### G1 — paired behavior through the Broca graft

Fresh-episode paired counterfactual training (a developmental population, *not*
lifetime evidence — stated up front) on the C1 wiki corpus: expose passage prose to
the organism, erase context, ask the question through the injected frozen model, train
only the paired causal-contrast margin. Pass requires correct passage to beat wrong
passage, bare floor, no-exposure, memory lesion, and internal lesion in held-out mean
correct-answer log-likelihood with a majority of strict per-question wins, plus live
per-depth lesion attribution. Gates and stop rules in
`experiments/g1-broca-paired-behavior.md`.

## Run

```bash
# CPU contract gate (toy backbone, no downloads, ~1 min) — must pass before GPU
.venv/bin/python -m fable2.test_fable2

# G0 audit on Aine (frozen Qwen3.5-4B, no training)
bash fable2/launch_g0.sh /tmp/fable2-g0

# G1 paired training on Aine (after G0 verdict is recorded)
bash fable2/launch_g1.sh /tmp/fable2-g1
```

Compute: `m@192.168.0.202` (Aine), **RTX 4090 by UUID only**
(`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`); the 2070S belongs to `jewels`.

## Layout

| File | Role |
|------|------|
| `config.py` | One dataclass; no flag graveyard |
| `backbone.py` | Frozen multi-depth injectable LM + toy CPU twin |
| `organ.py` | BrocaOrgan: sensor, output-tissue trunk, per-depth zero-init heads |
| `system.py` | Organism ⇄ backbone loop; episode arms; fp32 scoring |
| `episodes.py` | Corpus pairing; passage-prose exposure; leakage lint |
| `audit.py` | G0 interface audit CLI |
| `train.py` | G1 paired trainer CLI + telemetry + checkpoints |
| `test_fable2.py` | CPU contract gate |
| `launch_g0.sh` / `launch_g1.sh` | Aine launchers (UUID-pinned) |
| `experiments/` | Preregistrations + results |

Every code file has a companion `.md` (Purpose / Components / Decisions / Contracts).
