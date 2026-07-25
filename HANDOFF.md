# DMON-1 — Handoff

You are picking up a project mid-thought. Read `PROJECT.md` first, then this, then
`ARCHITECTURE.md` when you need to know *why* something is shaped as it is.

**The architecture was reset.** If you have context suggesting the creature is a body
grown on a resource field, that context is stale — see `BOOKMARK-morphology.md`. The
creature is the network: an asynchronously stimulated NCA fed by streamed input.

---

## Where things stand

The streaming core (`dmon/stream/`) is being built now. Everything else in `dmon/` is
the bookmarked display-organ work and should not be extended.

Compute is `m@192.168.0.202` (host `Aine`), RTX 4090 + RTX 2070S.
`CUDA_VISIBLE_DEVICES=0` selects the **4090** — CUDA orders fastest-first while
`nvidia-smi` calls the 2070S index 0. Check which card is busy before believing you are
on the big one.

---

## Do this first, in this order

1. **S0: character-level prediction on a continuous stream.** Input region, mirror
   cells, output region, online backprop concurrent with the stream. The metric is
   bits-per-character. This follows petridish's own precedent — verify the machinery
   against a hard number before chasing emergence.

2. **Run the null models alongside, not afterwards.** Parameter-matched GRU and
   transformer on the identical stream. "The NCA substrate is doing something" is a
   measurement. Under the previous architecture a trained rule scored *worse than having
   no network at all*, and that surfaced only because the control existed.

3. **Only then, S1: the energy economy.** Deplete per step, replenish on prediction
   *progress*. Not before S0 works — if capability and economy arrive together, a
   failure is unattributable.

4. **S2: grow the lattice at runtime** and show added cells are recruited. This is the
   claim that justifies choosing cellular automata at all, so it needs testing rather
   than assuming.

---

## How to tell you are fooling yourself

This project is unusually easy to fake and the fakes are pretty. Every item below is
something that already happened here, not a hypothetical.

- **Check that the log measures what you are claiming.** A run reported mass above 1100
  for 4000 iterations while the evaluated quantity was zero the whole time, because the
  logged number came from a different distribution than the evaluation.
- **The training distribution must contain what is evaluated.** Three separate
  occurrences in one session. Streaming hides this better, it does not prevent it.
- **Guard every denominator.** A verdict returned "PASS, 34x noise" on four creatures
  that were each a single dead cell.
- **Beat the best baseline, not a dead one.** A null model with zero learning satisfied
  the previous milestone's entire pass condition.
- **Check feasibility before spending compute.** A 40-minute GPU sweep once ran entirely
  inside a regime where nothing could survive; the check that would have caught it took
  one second.

---

## Do not propose these

- **Mirror cells the prediction pathway can write to.** The trivial optimum is to flatten
  the target: error zero, learning zero. Write-only from the stream, stop-gradient on the
  target branch. This is BYOL/SimSiam collapse arriving through a new door, and it looks
  like success in a loss curve.
- **Paying energy for raw prediction error.** Maximised by staring at noise. Pay for
  reducible surprise — prediction progress.
- **Making attention or engagement the objective.** It has a known degenerate optimum,
  and unlike the previous architecture there is no indifferent ecology underneath to keep
  it honest. Satiety is a day-one invariant here.
- **Resuming the resource-field morphology work as the main line.** It is the display
  organ and it returns at S4. See `BOOKMARK-morphology.md`.
- **Episodic training with a reset.** If the creature has to stop to learn, it is not
  asynchronous and S0 has not been passed.
- **Assuming an invariant holds because a channel is masked.** Energy was masked out of
  the rule's update and the rule minted it anyway, through an operator it controlled.
  Check the operators.

---

## The related repos

- `~/Code/petridish` — the direct ancestor of this architecture. Metabolic economy,
  genotype inheritance with mutation, two-timescale training, and character-level
  prediction already verified at nanoGPT scale. Its central idea — energy earned from
  the *varying* component of stimulation rather than raw traffic — is what DMON-1 is
  built on.
- `~/Code/Bonsai` — Swift/Metal NCA runtime, FiLM mood manifold, limit-cycle animation,
  volumetric raymarching. Relevant at S4 for the display organ. All trainers are
  target-supervised.
- HBPE — its discovered three-tier vocabulary is the right observability instrument for
  the creature's own state trajectory. An ethogram nobody wrote.

---

## Conventions

Every code file gets a companion `.md` with Purpose / Components / Decisions /
Contracts. `PROJECT.md` is the session entry point and its Decision Log is the thing
that most needs keeping current — most of the important choices here are *negative*, and
they are invisible in the code.
