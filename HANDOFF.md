# DMON-1 — Handoff

You are picking up a project mid-thought. Read `PROJECT.md` first, then this, then
`ARCHITECTURE.md` when you need to know *why* something is shaped as it is.

---

## Where things stand

`dmon/substrate.py` and `dmon/train_m0.py` exist and run. ~420 lines, ~11K parameters.
A 300-iteration CPU smoke run on a 24×24 grid took mass from 1 → 23 with box dimension
settling near 1.54.

**That result means the plumbing works and nothing else.** The grid is too small for
box-counting to be meaningful and the run is two orders of magnitude too short. Do not
cite it as evidence of anything.

Nothing has been run at scale. No contingency test has been performed. The central
hypothesis of the project is completely untested.

---

## Do this first, in this order

1. ~~**Fix the contingency test.**~~ Done — `dmon/contingency.py`. The noise floor now
   comes from independent *training* runs (`--seeds`), cross-evaluation is used rather
   than discarded, and the verdict refuses to divide by a degenerate noise floor.

2. **Sweep diffusion length before anything else.** `field_diffusion` relative to grid
   diameter is the knob that decides whether M0 can work at all. Too fast → uniform
   field → no gradient → blob. Too slow → only cells touching a source survive. There
   is a Goldilocks band and you need to find it before interpreting any morphology.
   `python -m dmon.sweep` — ~40 min on the 4090 at a 4k-iteration screening budget.

3. **Verify the light cone.** `steps ≥ grid` (perception radius is 1, so information
   moves one cell per step). `SubstrateConfig.light_cone_ok()` checks it and `train()`
   now warns. It binds on the *eval* horizon; training deliberately samples shorter
   rollouts as a curriculum.

4. **Run M0 properly, then probe it.** 64×64, ≥64 steps, batch 32, 20k+ iterations on
   the 4090 (~33 min). Then `python -m dmon.probe` on that single checkpoint — the
   legibility probe needs one rule, not twenty, and it tests the assumption M2, M3 and
   M4 all rest on. See `ARCHITECTURE.md` §M2.

5. **Then the full contingency sweep.** 4 geometries × 5 seeds × 20k iters ≈ 11 h.
   An overnight job. Do not economise on the seed count; it is the only thing standing
   between this project and a test that cannot fail.

Compute is `m@192.168.0.202` (host `Aine`). Note that `CUDA_VISIBLE_DEVICES=0` selects
the **4090**, not the 2070S — CUDA orders devices fastest-first while `nvidia-smi` lists
the 2070S as index 0. Check `nvidia-smi` for which card is actually busy before
believing you are on the big one. At 64×64/64 steps/batch 32 the peak is 8.4 GB, so
these settings do not fit on the 2070S at all; lower `--batch` before `--grid`.

---

## How to tell you are fooling yourself

This project is unusually easy to fake and the fakes are pretty. Max's other repos
carry this discipline explicitly (petridish's README: "There are no decorative signal
particles" / "not synthetic animation") and it should be maintained here.

- **The renders are not the verdict.** Descriptors are. A run that looks alive and
  fails the contingency test has failed.
- **The dangerous success**: a rule that memorises one shape and ignores the field
  looks exactly like a rule that learned to read gradients. Only cross-evaluation
  separates them — drop the trained rule into a *different* source geometry. A
  memoriser drags its shape along unchanged. A creature grows something else.
- **If a run needs a target to converge, the run is invalid, not the objective.**
- **Descriptors degenerate at small mass.** Box-counting needs enough occupied cells
  to mean anything; below ~50 cells treat `box_dim` as noise.

---

## Do not propose these

A fresh model will pattern-match to standard NCA work and suggest all of the
following. Each one is the exact failure mode this project exists to avoid.

- **Adding a target image**, or any morphological prior (symmetry, compactness,
  connectivity regularisers). This reintroduces target supervision through the back
  door. Rejected on purpose, not by oversight.
- **Bolting an LLM onto the creature for speech.** See `ARCHITECTURE.md` §M4 — the
  point is that utterance and body share state so speech cannot lie.
- **Training on a text corpus** for the signalling channel. Bolting on English is the
  same error as bolting on an LLM, just less obvious.
- **Making attention/engagement the primary objective.** It has a known degenerate
  optimum. It is layered *over* an indifferent ecology at M3 or not at all.
- **Forking Bonsai's trainers.** They are all target-shaped. The cell rule underneath
  is worth stealing; the training loop is not.
- **Renting a cluster.** Not before M2. The bottleneck is the objective, not FLOPs.
- **A menu of forms to choose from.** "Let the creature pick chibi or shoggoth" makes
  it a selector over a library someone else authored — the spritesheet one level up.
  Morphology drifts over a continuous morphospace or it isn't emergent.
- **A display channel, before M3.** It is deliberately not load-bearing for survival,
  so with no attention field it has no path to the ledger and the gradient is exactly
  zero. This is the bug from the list below, wearing a costume.
- **Coupling appearance to mood by hand.** If distress has to be *written into* the
  display, it is a mood vector passed to a decoder and the honesty claim is void. Run
  the legibility probe (`ARCHITECTURE.md` §M2) and find out whether the coupling is
  already there.

---

## Bugs already found and fixed — do not reintroduce

- **Zero gradient from hidden channels.** The first substrate gave the rule 16
  channels with no causal path to energy. It could compute anything and never touch
  survival, so there was nothing to learn and the gradient was *exactly* zero. This
  would have looked like a learning-rate problem for a day. Fix: the rule acts through
  two metabolic levers (uptake effort ch1, transport conductance ch2). Any new channel
  you add needs a path to the ledger or it is decoration.
- **Seed dilution.** Ungated energy transport spread the seed's reserve into its halo
  and every creature starved within ~40 steps. Fix: transport is gated per-cell and
  the gate bias starts near-closed (−2), uptake near-open (+2).
- **Energy minting.** Channel 0 is masked out of the rule's update. If you ever let
  the rule write energy directly, the economy stops meaning anything and every result
  after that point is void.
- **Energy minting through non-conservative transport.** *The big one.* The transport
  term was `e += e_transport * conduct * laplace(e) * body`. The laplacian kernel sums
  to zero, so unmodulated it conserves — but `conduct` is per-cell, so cell *i* gained
  using `conduct_i` while its neighbours lost using theirs, and `Σ conduct_i·lap(e)_i`
  is not zero when conductance is high where the laplacian is positive. **The rule
  controls conductance.** Channel 0 being masked out of the rule's update did not
  prevent the rule from writing to the ledger; it only prevented it from doing so
  directly. In 4k iterations it found the exploit and grew a stable 25-cell body on a
  grid **with no food in it at all** — verified by deleting the sources and getting an
  identical body. Seed energy 2.0 became 10.4 in a closed system.
  Fixed by `Substrate._transport`: pairwise flux exchange with symmetric conductance
  `min(c_i, c_j)`, so every unit leaving one cell arrives in exactly one neighbour and
  `sum(de)` is identically zero by construction. `dmon/test_conservation.py` tests it,
  including against a conductance field deliberately shaped like the exploit.
  **Everything measured before this fix is void**, including the first diffusion sweep.
  The general lesson: an invariant that is enforced on one channel can be violated
  through any operator that channel passes through. Check the operators, not just the
  masks.
- **A verdict that divides by a collapsed noise floor.** The first smoke test of
  `contingency.py` returned `PASS — separation 34x noise` on four rules whose bodies
  were each a single dead seed cell: every rule produced identical descriptors, so the
  noise floor went to zero and the ratios exploded. The lesson generalises beyond this
  one function — **every headline number in this project is a quotient by a baseline,
  and a degenerate baseline manufactures confidence out of nothing.** Any new metric
  needs a guard that fires when its denominator collapses.

---

## Known weaknesses in the current code

- **Energy is not exactly conserved.** The transport laplacian is masked by `body`,
  which breaks conservation at the growing edge — energy leaks slightly. Tolerable for
  M0; must be replaced with a conservative flux-exchange formulation before any claim
  about metabolic efficiency.
- **No rendering.** There is no visualisation at all yet. Add one early; you will
  misdiagnose failures without it. Borrow Bonsai's approach but not its trainers.
- **No checkpointing.** Add before the first long run.
- **Descriptors are cheap and coarse.** Fine for a pass/fail verdict, inadequate for
  characterising *what kind* of shape emerged. Expect to need better ones by M1.

---

## The related repos

- `~/Code/Bonsai` — Swift/Metal NCA runtime. Real assets: the cell rule, the FiLM mood
  manifold (which becomes M2's slow variable), limit-cycle animation, volumetric
  raymarching, ongoing heteroclinic-cycling experiments. All trainers are
  target-supervised — that is what DMON-1 exists to escape.
- `~/Code/petridish` — developmental neural graph. Real assets: the metabolic economy,
  genotype inheritance with Gaussian mutation (lift this for M1's kin tags), the
  two-timescale training pattern, and one genuinely good idea worth preserving:
  **energy is earned from the batch-varying component of stimulation, not raw
  traffic**, so self-exciting loops incur load without earning food. Metabolism
  denominated in information rather than activity.
- HBPE — its discovered three-tier vocabulary is the right observability instrument
  for the creature's own state trajectory. See `ARCHITECTURE.md` §6.

---

## Conventions

This repo follows Max's `modular-docs` skill: every code file gets a companion `.md`
with Purpose / Components / Decisions / Contracts; `PROJECT.md` is the session entry
point and its Decision Log is the thing that most needs keeping current. Read the
companion doc before touching any file. Update it after.

The Decision Log matters more than usual here because most of the important choices
are *negative* — things deliberately not done — and they are invisible in the code.
