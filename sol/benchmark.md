# `benchmark.py`

## Purpose

Runs reproducible Tiny Shakespeare experiments for SOL, a parameter-matched GRU, and a
parameter-matched causal transformer on identical continuous train/validation streams,
with checkpointing and append-only metrics.

## Components

### `_run_sol`
- **Does**: Trains or resumes SOL, logs behavioral and organism metrics, evaluates state
  ablations, and atomically promotes the best held-out checkpoint.
- **Interacts with**: `ContinuousTrainer`, `checkpoint.py`, and `evaluate.py`.

### Scientific controls
- **Does**: `--freeze-edges` trains an otherwise identical organism without modifying
  its directed edge weights or biases; `--no-metabolism` holds energy at one.
- **Does**: `--external-energy-gain`, `--energy-transport-rate`,
  `--energy-maintenance-flow`, `--quiescence-energy`, and
  `--full-activity-energy` expose the parameter-neutral directed economy and viability
  ramp for matched experiments.
- **Does**: `--energy-maintenance-flow` adds a conserved baseline request only on
  installed non-self axons, allowing silent tissue to receive energy without geometric
  neighbor diffusion or maintenance on speculative probes.
- **Does**: `--no-reward` disables both cell and synapse reward effects while leaving
  event traces intact; `--no-fast-plasticity` isolates the older cell-level reward path.
- **Rationale**: Learned morphology and metabolic modulation must earn their claims
  against controls rather than ride on the shared cell rule; delayed eligibility
  feedback and fast synaptic memory must likewise improve prediction rather than merely
  sound plausible.
- **Rationale**: `--no-metabolism` also disables energy transport, preserving the exact
  energy-one capability control instead of silently introducing redistribution,
  including maintenance flow.
- **Does**: `--fast-plasticity-gain` records and varies the reward-to-efficacy scale for
  matched gain sweeps without changing parameter count.
- **Does**: `--reward-baseline-decay` records the time scale used to turn surprise into
  signed advantage relative to each stream lane's recent expectation.
- **Does**: `--cell-reward-gain` and `--backward-credit-gain` separate the existing
  global cell reward from an output-originating credit wave that travels against signed
  axons. `--backward-credit-decay` controls its cross-token memory.
- **Does**: `--output-error-credit-gain` enables a parameter-neutral decoder-shaped
  reverse vector and `--output-error-credit-decay` controls its cross-token memory.
  `--no-reward` disables all three reward paths for a complete matched ablation.
- **Does**: `--eligibility-routed-output-credit` redistributes the decoder-shaped vector
  within each installed dendrite fan according to signed alignment with source event
  eligibility. `--eligibility-routing-gain` scales that evidence before its bounded
  normalization; gain `1` preserves the original routed mechanism and equal evidence
  preserves historical transport scale at every gain.
- **Does**: `--reward-plastic-output-credit-routing` instead lets later prequential
  reward meet a remembered branch-specific routing event. Explicit decay, plasticity
  gain, and preference limit govern this parameter-free fast state; it is mutually
  exclusive with instantaneous eligibility routing.
- **Does**: `--exploratory-output-credit-routing` instead uses branch eligibility only
  to propose a bounded zero-sum preference change, alternates that change on/off in one
  live organism, and commits from candidate-minus-incumbent prequential reward.
  Cadence, warmup, trial length, margin, proposal step, and minimum evidence are
  explicit manifest fields. The benchmark also records `--eval-every` as the protected
  routing boundary interval, preventing a new trial from starting or crossing a plotted
  checkpoint.
- **Does**: Routing starts only on structural non-decision phases, which still update
  structural confirmations. It resolves before the next consolidation phase while
  ordinary forward traffic, probes, parameter learning, persistent state, and evidence
  accumulation remain active.
- **Rationale**: Direct reward, scalar reverse transport, decoder-shaped reverse
  transport, and their combinations must be compared explicitly rather than silently
  changing the organism's learning rule.
- **Does**: `--structural-plasticity` runs bounded between-window grow/prune phases;
  `--structural-probes-only` supplies the parameter- and intervention-matched
  fixed-topology control with identical probe accumulation and rotation.
- **Does**: Structural flags record probe gain, phase cadence, replacement budget,
  consecutive confirmation phases, credit decay/margin, maturity, and endpoint energy
  cost.
- **Does**: `--initial-active-dendrites` sets birth fan-in below fixed slot capacity.
  `--structural-variable-fan-in` enables spawn/prune decisions, with explicit minimum
  fan-in, traffic/credit pruning thresholds, and usage/vector/locality evidence gains.
- **Rationale**: Capacity-matched controls retain identical learned parameters while
  active anatomy is allowed to earn or lose connections.
- **Does**: `--structural-global-fitness` requires locally confirmed probes to also
  predict a reduction in exact sequence loss before permanent installation.
- **Does**: Structural probation flags hold a confirmed graft provisionally, accumulate
  observed prequential improvement over a checkpointed pre-graft developmental baseline
  during uninterrupted co-adaptation, and commit or restore it at the configured
  deadline. Probes-only uses a virtual wait of the same duration.
- **Does**: `--structural-probation-exploratory-traffic` instead retains incumbent
  anatomy, alternates the selected candidate probe on/off across adjacent live stream
  windows, and commits from the within-organism reward difference. All other probes and
  all ordinary learning continue in both arms.
- **Rationale**: Rewiring must beat the same causal probes without topology mutation,
  rather than receiving free exploratory traffic unavailable to control.
- **Rationale**: The fixed-topology run remains a diagnostic, but it is not the primary
  survival measure for a body that co-develops with its organ.

### Optimizer schedule
- **Does**: `--learning-rate-decay-start`, `--learning-rate-decay-end`, and
  `--minimum-learning-rate-ratio` optionally hold the base learning rate before a
  parameter-free cosine decay shared by SOL, GRU, and transformer.
- **Does**: Applies the rate for the upcoming absolute update, records it in train rows,
  summaries, and SOL checkpoint metadata, and therefore preserves schedule phase across
  resume.
- **Compatibility**: A resumed SOL organism retains the checkpoint's base learning rate;
  the ordinary `--learning-rate` argument does not silently rewrite its optimizer policy.
- **Rationale**: Repeated late-checkpoint regression should be tested as an optimization
  failure independently of metabolism, morphology, or parameter count.

### Topology report
- **Does**: Records sensory reachability, output reachability, self-edge count, and mean
  sensory-to-output distance plus structural rewrite/credit telemetry in every SOL
  summary.
- **Does**: Records probation activity, attempts, commits, rollbacks, observations, and
  mean observed advantage.
- **Does**: Records the independent routing-traffic phase, proposal, ABBA reward
  aggregates, commit/reject totals, and complete decision ledger.
- **Interacts with**: `analyze_topology` in `topology.py`.

### Stability report
- **Does**: Measures final and worst post-best BPC regression across completed
  validations and marks runs unstable beyond `--max-final-regression-bpc`.
- **Does**: Fits the final `--convergence-window` held-out evaluations and records
  terminal slope, residual noise, a 95% slope interval, and whether that interval lies
  inside `--max-terminal-slope-bpc-per-100`.
- **Does**: Aligns every non-virtual exploratory-traffic decision with held-out
  validation before and after that trial, stopping before the next intervention, and
  records trial-level body survival in the final summary.
- **Rationale**: A transient best checkpoint does not establish a viable continuous
  organism if the same uninterrupted run later collapses.
- **Rationale**: An endpoint comparison does not establish relative capability while
  the terminal curves are still moving materially.

### `_run_gru`
- **Does**: Trains a parameter-matched GRU with the same batch lanes, chunk length,
  optimizer family, token budget, held-out split, and evaluation window.
- **Interacts with**: `CharacterGRU` in `baselines.py`.

### `_run_transformer`
- **Does**: Trains a parameter-matched causal transformer with a rolling context on the
  same lanes, token budget, split, and evaluation window.
- **Interacts with**: `CausalCharacterTransformer` in `baselines.py`.

### CLI
- **Does**: Records a manifest, JSONL history, summary, and local checkpoint beneath an
  explicit output directory.
- **Does**: Records current and process-peak allocated/reserved CUDA bytes in SOL train,
  evaluation, and summary artifacts so capacity decisions include the full evaluation
  and generation path.
- **Rationale**: Large tensors stay out of Git while small evidence files remain easy to
  inspect and copy home.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dual-GPU launch | `--model sol|gru|transformer` lets each GPU run one independent job | CLI names |
| Local UI backend | `best.pt` is a complete SOL checkpoint with evaluation metadata | Checkpoint selection |
| Scientific report | Both models see the first 90% for training and final 10% for validation | Split semantics |
| `promote.py` | SOL summaries and JSONL history expose completed-run stability | Stability fields or evaluation rows |
| S6 analysis | Summary includes the full traffic ledger and decision-aligned survival | `exploratory_survival` schema |
| Schedule experiments | Disabled decay reproduces the historical constant rate | Schedule defaults or update indexing |
| Experiment horizon | Every meaningful SOL result exposes an explicit terminal-trend verdict | Convergence summary keys |

## Notes

- `--updates` is a final update number when resuming SOL, not an additional count.
- The 4090 should run SOL; the 2070S is better used for the GRU and ablations than for
  mismatched synchronous data parallelism.
