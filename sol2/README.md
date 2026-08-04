# SOL2

SOL2 is the next experimental line for DMON-1: a persistent neural organism made from
typed tissues on a sparse directed connectome. It keeps SOL's scientific discipline and
Fable's lean capability path while removing the accidental assumption that every cell
must share one universal rule.

## What is new

- Separate input, compute, relay, and output tissue rules.
- Optional bounded private target/time-constant expression for every mutable cell.
- Organ-specific learned queries that attend only to dedicated output tissue.
- Parameter-count-neutral bounded graph operators: damp-only spectral rescaling and
  unit-ball attention, bounded values, and bounded edge bias.
- Stream-written sensory memory named honestly as short-term memory, not learned credit.
- Dormant dendrite slots and append-only relay growth that never deletes a working edge.
- Freeze and within-tissue shuffle probes reported together.
- Complete process checkpoints and matched GRU/transformer controls.
- A versioned background-learning runtime that keeps foreground ticks separate from
  backward computation.
- A branch-controlled procedural-transfer benchmark that separates new interfaces from
  new algorithms and can freeze the core while sensory/output organs adapt.
- Checkpoint-compatible detachable sensor/effector bundles with explicit port
  selection, matched true-attachment branches, and removal/reattachment recovery.
- Causal-utility calibration and graded update protection that can consolidate useful
  cells and connections while preserving a plastic internal reserve.
- DMON-L0 continuous language grafts: a frozen replaceable language backbone supplies
  contextual senses and fluent decoding while the persistent organism emits low-rank
  control vectors through dedicated output tissue.

## Architecture

```text
character stream
    ├── input tissue (fast typed rule)
    └── sensory-memory tissue (stream-written FIFO)
                │
       sparse directed connectome
                │
      compute tissue ↔ relay tissue
                │
          output tissue (sink)
                │
       attentive character output organ
```

Input, compute, and relay cells own persistent state; output tissue is cleared at each
token boundary so it cannot become an autonomous language model. A rule is shared only
within its tissue. In the identity treatment, each mutable cell additionally owns
bounded private expression. Relay growth appends both state and private parameters
while preserving the rules as tissue genomes.

Output targets read only compute/relay cells. The architecture therefore permits
low-traffic reserve cells but cannot solve the task through a direct sensory-to-decoder
shortcut. Gradient participation is logged as effective, material, and reserve
fractions rather than optimized as an activity reward.

## CPU gate

```bash
.venv/bin/python -m sol2.test_sol2
.venv/bin/python -m sol2.test_procedural
.venv/bin/python -m sol2.test_living_language
```

The suite is deliberately executable without pytest so a fresh project environment
can validate the architecture before optional test-runner dependencies are installed.

This gate must pass before GPU work.

## Living language graft

DMON-L0 treats a pretrained language model as a detachable language organ, not as the
creature itself:

```text
user/generated token -> frozen contextual feature -> continuous sensory graft
    -> persistent SOL2 cells/connectome -> dedicated output tissue
    -> low-rank control vectors -> frozen language head -> next token
```

The language model remains frozen. Its final hidden state is computed without a
training graph, reused as sensory input, and modified only by an exactly zero-initialized
control residual before the frozen vocabulary head. The organism therefore owns all
new learned state and adaptation. Its state continues when visible language context is
erased and receives generated tokens as subsequent sensory input.

`test_living_language.py` includes the first architectural proof: after a tiny paired
curriculum, identical post-erasure prompts are solved perfectly in normal operation,
while zero language control, cellular reset, and internal-tissue lesion remain at or
below chance. This is a causal integration gate, not an LLM quality comparison.

The staged production protocol is in
`experiments/l0-living-language-organ.md`.

Once a decoder-only checkpoint is available locally, measure its real graft boundary
before training:

```bash
.venv/bin/python -m sol2.language_smoke \
  --model /path/to/local/model --device cuda:0 --dtype bfloat16 \
  --out /tmp/dmon-l0-smoke.json --local-files-only
```

## Procedural transfer

`experiments/s1-procedural-transfer.md` begins the organism-level evaluation line.
Freshly generated programs replace text compression; only exact procedural answers are
scored. One acquired organism is forked into unchanged, new-interface, reversed-
procedure, and SOL2 organ-only branches. Accuracy by program length and causal tissue,
identity, topology, reset, and memory interventions are retained in `metrics.json`.

```bash
.venv/bin/python -m sol2.procedural_benchmark --model creature --device cpu \
  --acquisition-updates 1000 --adaptation-updates 500 \
  --out-dir /tmp/sol2-procedural
```

This is a dense bridge test, not yet a DNC memory experiment. A procedural mode must be
useful while active before external storage and retrieval of that mode is justified.

Before attaching a new organ, compare organism sizes with resumable mastery-gated
acquisition:

```bash
.venv/bin/python -m sol2.procedural_acquisition --device cpu \
  --max-updates 10000 --mastery-accuracy 0.80 \
  --out-dir /tmp/sol2-acquisition
```

The checkpoint and JSON telemetry update every evaluation interval. They preserve the
living state for the subsequent organ-attachment fork and expose real cell activation,
identity differentiation, and connectome values to visualization tooling.

The promoted checkpoint feeds the true organ-attachment experiment described in
`experiments/s1p2-organ-attachment.md`. Each branch is independently resumable:

```bash
.venv/bin/python -m sol2.organ_attachment \
  --acquisition-checkpoint /tmp/sol2-acquisition/acquisition.pt \
  --branch full --device cpu --adaptation-updates 10 --eval-every 5 \
  --eval-batches 1 --out-dir /tmp/sol2-organ-full
```

After all four canonical branch directories complete, aggregate the frozen gates with
`python -m sol2.organ_attachment_analysis <result-root>`.

S1-P3 tests whether a doubled organism can learn through B without overwriting A by
protecting useful cell expression, edges, and the slower tissue genome. The measured
arm is compared with an equally large plastic organism and a within-tissue shuffled-
utility control:

```bash
.venv/bin/python -m sol2.consolidated_attachment \
  --acquisition-checkpoint /tmp/sol2-acquisition/acquisition.pt \
  --branch consolidated --device cpu --adaptation-updates 10 --eval-every 5 \
  --utility-batches 2 --eval-batches 1 --out-dir /tmp/sol2-consolidated
```

The frozen result protocol is in `experiments/s1p3-consolidated-reserve.md`.
Once all three branches complete, run
`python -m sol2.consolidated_attachment_analysis <result-root>` to apply its gates.

## Short local smoke

```bash
.venv/bin/python -m sol2.train --model creature --device cpu \
  --updates 20 --eval-every 20 --eval-tokens 64 \
  --out-dir /tmp/sol2-smoke
```

## GPU experiment

The preregistration is `experiments/s0r-stable-differentiation.md`. Generate its
manifest without beginning training:

```bash
.venv/bin/python -m sol2.benchmark --root sol2/runs/s0r
```

`--execute --device cuda:0` starts the recorded arms after a GPU is allocated.
The preregistered first launch is a four-creature, seed-7, 2,000-update diagnostic via
`--creature-only`; a single weak differentiation result is not treated as a conclusive
architectural rejection.

## Deliberate omissions

- No claim that chunked training is asynchronous; `runtime.py` is the separate
  concurrency prototype.
- No metabolism yet. SOL's conserved ledger will return only with an externally audited
  prediction-progress inflow.
- No content-addressed memory organ yet. Retrieval first needs a dense curriculum and a
  shared write/query key contract.
- No random structural search. Growth is append-only until the informed-proposal
  experiment has a stable, differentiated substrate.
