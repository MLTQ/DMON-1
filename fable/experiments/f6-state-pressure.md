# F6: state pressure — cued recall under load (preregistered 2026-07-30)

## Why this experiment exists (Max's steer)

Every fable task so far is state-poor: character-level Shakespeare at BPC ~2.0
is mostly local structure, and F0/F1 measured the consequence — computation
concentrates in the port cells, the bulk is functional but interchangeable,
and 4× more tissue does the same work. The trajectory data closes the
"just undertrained" escape: shuffle-internal delta is flat across all 8000
updates on all three seeds. The bulk is not undertrained; it is unemployed.

The asymmetry the tasks have been ignoring: the creature carries ~16,384
floats of persistent state (128 cells × h=128) against the matched GRU's 230.
Nothing we have run pays for that. F6 is a task whose difficulty dial is
*state demand*: it forces information out of the ports (instantaneous I/O),
past the mirror ring (32-char verbatim echo), into whatever can hold it.

Task family is chosen deliberately: **cued associative recall** is the
established small-scale discriminator between sequence architectures and the
mechanistic precursor of in-context learning (induction heads). This is the
closest tractable proxy for "the emergent capabilities that happen in larger
networks" at a scale this project can run.

## Task: the pointer stream

Natural corpus text with embedded key–value episodes, per lane:

- **Inject**: `{kv` — sentinel `{`, key `k` (a–z), value `v` (a–z), at a
  random point. The pair (k → v) is now live.
- **Query**: after delay D (log-uniform in [16, 1024] emitted chars):
  `}kv` — sentinel `}`, the key, then the value. The value position is the
  scored one: the model saw `}k` and must produce `v`. The pair then retires
  and a fresh key is injected later.
- **Load**: up to M = 8 pairs live concurrently per lane. Keys unique among
  live pairs (no rebinding in v1 — rebinding is a later dial).
- Everything else is ordinary next-char prediction on the corpus, scored
  separately so LM quality and recall never blur.

Chance on a scored position = log2(26) ≈ **4.700 bits**. A model that holds
the pair pays ~0; a model that lost it pays ~4.7. Recall BPC by delay bucket
([16,32), [32,128), [128,512), [512,1024]) is the state-capacity curve.

What each architecture *should* do, stated before running:
- **Mirror ring** covers D < ~32 verbatim — the creature's sanity band.
- **Transformer** (seq 128, stateless): recall beyond its window is
  impossible *by construction* — it should step to chance at D > 128. Its
  within-window recall should be excellent (attention is built for this).
- **GRU** (h = 230 floats): must hold up to 8 pairs plus LM state in 230
  floats through thousands of steps — should degrade with delay and load.
- **Creature**: 16k floats of distributed persistent state. If the substrate
  thesis has content, this is where it shows. If the creature also fails,
  the mean-field diagnosis deepens: capacity without addressability.

## The recruitment instrument (the actual point)

At final eval, the creature is probed with internal tissue frozen, and the
freeze cost is split by position type:

- **freeze-internal delta on recall positions** — do the pairs live in the
  bulk?
- **freeze-internal delta on natural positions** — generic LM contribution
  (F0 measured this at ~0.2–0.5).

F0's diagnosis predicts: if this task recruits the bulk, the recall-position
freeze delta should be large (bits, not centibits) and delay-dependent, while
the natural-position delta stays near F0's. That would be the first direct
evidence of task pressure moving information into the tissue — the thing F5's
growth machinery needs to exist before "growth toward demand" is testable.

## Arms

Creature, matched GRU, matched transformer (params matched to the creature at
vocab 67 = corpus 65 + two sentinels), seeds 7/13/21, identical generated
streams, 8000 updates, F0's annealed schedule (this is a capability test; the
constant-LR question stays with F2's isolation issue). B = 4 lanes × chunk 32.
Per-token raw records kept (position type, delay, loss).

Scored prequentially during training (second half only) *and* on a
frozen-weights held-out tape (16,384 chars from the holdout corpus region,
fresh pairs, same generator parameters — the training distribution contains
what is evaluated, including the sentinels).

## Pass / fail

- **Substrate signal**: creature recall beats GRU at D ≥ 128 by ≥ 0.5 bits
  on the recall curve, consistently across seeds, AND the recall-position
  freeze delta exceeds 1.0 bit (the pairs demonstrably live in the tissue).
- **Task works, creature fails**: GRU or transformer-within-window hold
  recall while the creature sits at chance at D ≥ 128 — the state advantage
  is not addressable by this architecture. Honest negative; F5/F3 get a
  concrete target (make the bulk addressable), and the mean-field diagnosis
  is confirmed from a second direction.
- **Task fails**: every model at chance for D > 32 at this scale — the task
  is too hard for 350k params / 1M chars; re-dial (shorter delays, lower
  load, more training) before concluding anything.

## Launch

```bash
bash fable/run_f6.sh      # waits for the 2070S to free, then runs all arms
python3 -m fable.recall_report --root fable/runs/f6 --out fable/runs/f6/REPORT.md
```

---

# RESULTS, round 1 (2026-07-30, 9 arms, zero skipped updates): **task fails — re-dial**

Every model, every delay bucket, at or above the 4.700-bit chance level — on
the held-out tape **and on the training stream itself** (creature 4.76, GRU
5.07, transformer 4.81 over ~9,000 scored training events). Even D ∈ [16,32),
inside the creature's mirror ring and inside every model's context window, is
at chance. The GRU sits *above* chance (5.03–5.10): it never even fully
learned that value positions are lowercase letters.

This is the third preregistered branch (task too hard at this dial), with a
sharper reading: it is not a state-capacity failure, because nobody learned
the retrieval pattern at *any* delay. Two dial errors, both self-inflicted:

1. **Weak incentive**: recall positions are ~1.7% of the loss mass. A model
   loses ~0.08 mean BPC by ignoring recall entirely — and all three did.
2. **Underbudgeted**: B=4 lanes gave every arm a third of F0's tokens at the
   same update count. Natural BPC 2.77–3.09 vs F0's 2.01–2.32 confirms the
   whole round was undertrained (F7's lesson, replicated by accident).

Also noted for round 2: round 1's "natural" positions include episode
overhead (sentinel timing, key identity — irreducibly unpredictable), which
inflates natural BPC by a small amount; round 2 excludes them.

## Round 2 dial (preregistered before running)

B=12 (F0's token rate), inject probability 0.02 → 0.06 (~3× recall density,
~5% of loss mass), 12,000 updates, episode-overhead positions excluded from
natural, everything else unchanged. Same pass/fail bars as round 1. Runs
as `fable/runs/f6_1`.
