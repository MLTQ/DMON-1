# L0-C1y: answer-span coherent value transport

Status: preregistered after C1x and before any C1y optimizer update.

## Question

C1x learned where to address but not what value to retrieve: addressing KL improved
through update 100 while recall separation, recurrent gradients, and expressed control
collapsed. Can exact local credit in the organism's own sensory coordinate reopen a
passage-specific route without increasing organism size or adding an inference bypass?

## Frozen rescue treatment

- Resume the exact C1x seed-7 update-100 checkpoint, optimizer moments, RNG, schedule,
  736-cell geometry, coherent top-16 recall, Qwen 3.5 4B, prefix control, and fresh-
  episode paired-counterfactual protocol. Use only the physical RTX 4090.
- Replace the failed random-projection memory and direct-recall semantic credits
  (weights `4/4`) with exact coherent answer-value credit at weight `8` and sparse
  relay/output transport credit at weight `8`, temperature `0.05`.
- Preserve addressing KL `1`, passage-effect KL `1`, causal contrast `1`, LR `0.001`,
  recall/sensor/effector multipliers `20/4/1`, and all other zero-weight objectives.
- Locate the exact `Designated answer:` token span in each training exposure. Map it
  through the actual circular FIFO write positions and detach the bounded mean of those
  stored memory neurons as the ideal coherent recall target.
- Align the live final-token recall with that value. Independently soft-select relay
  and output cells by cosine similarity and align their pooled live state with the same
  detached value. No target, span, pool, or new readout exists in evaluation/inference.
- Continue for 25 optimizer updates, from update 100 through 125. Checkpoint every
  update, evaluate development at 125, then run the held-out causal suite once.

## Gates

- Every span is found exactly and survives memory capacity. Qwen remains frozen and
  every gradient, optimizer state, loss, and activation statistic remains finite.
- Addressing KL remains below `0.10`, selected teacher mass remains above `0.50`, and
  query/key gradients remain live.
- Coherent recall alignment becomes stably positive, recall separation grows from the
  C1x final-block `0.00352`, and live recall RMS approaches rather than evades target
  RMS. Value/output recall gradients must reopen.
- Relay and output value alignment become positive while effective-cell counts fall
  below their full 64/16-cell populations. Connectome, tissue, and transport gradients
  must reopen; unused cells are allowed.
- Control and student-effect RMS must recover above the C1x silent tail without a
  corresponding loss explosion. Late causal-advantage satisfaction must exceed C1x's
  `0.28` final block and preferably C1w's `0.542` late block.
- Correct passage must beat no exposure/floor, wrong passage, memory lesion, and
  internal lesion in mean held-out target likelihood and in a majority of strict
  per-question comparisons. Aggregate accuracy alone cannot pass.

A pass licenses longer maturation, continuous-lifetime retention, utility protection,
and demand-driven growth. Local alignment without passage-specific language control
licenses work on the relay/output-to-LLM interface. Failure with live local gradients
rejects this exact value-transport treatment; failure from optimizer inertia licenses
a fresh-weight ablation before rejecting the architecture.

## Result

C1y completed update 125 with Qwen frozen at zero trainable parameters and zero
gradient tensors. Peak allocation/reservation was 21.85/22.44 GiB. The raw artifact is
`l0c1y-answer-value-s7-u125-result.json`, SHA-256
`d527a6d05aed5832b02fdb0e9b5cb3052f9e0a6f9e7ffddd7864680f78a8b8bf`.

The local coherent route learned, but mostly as a passage-common representation.
Across updates 101-112 versus 114-125:

- addressing remained healthy: KL `0.0397 -> 0.0345`, selected teacher mass
  `0.551 -> 0.540`, and effective slots `12.64 -> 12.75`;
- recall-to-answer alignment improved `0.872 -> 0.910`, live/target RMS remained
  closely matched at about `0.176/0.179`, and value loss fell `0.0080 -> 0.0057`;
- relay alignment rose `0.447 -> 0.875` and output alignment `0.014 -> 0.281`, while
  transport loss fell `0.0391 -> 0.0207`;
- all sensor, recall, connectome, tissue, transport, and effector gradient groups stayed
  finite and nonzero. Query/key remained strong; value/output recall gradients fell
  but remained live.

The paired geometry exposes the failure. Exact answer targets differed by RMS `0.0372`
early and `0.0338` late, but live recall differed by only `0.00320` and `0.00345`:
roughly one tenth of the available counterfactual signal. The absolute target's common
component dominated its MSE. Relay selection initially concentrated to about 18-31
cells, then broadened to `38.7` late and `43.5` in the last five updates; output
pooling remained effectively all `15.9/16` cells. Control separation stayed around
`1.4e-5`, and control RMS peaked near update 110 before returning to `0.00048` in the
last five updates. Causal satisfaction improved only `0.271 -> 0.417` and ended at
`0.35`, below C1w's late `0.542`.

Development normal loss was `0.30249`, worse than the no-exposure floor `0.27222`,
wrong passage `0.29785`, and memory lesion `0.29688`; it beat internal lesion only
slightly (`0.30273`). Held out, normal was 75%/`0.56683`, better in mean loss than the
floor `0.57526`, memory lesion `0.57751`, and internal lesion `0.59058`, but worse than
wrong passage `0.56592`. Strictly it beat floor on 2/8 questions, wrong passage on 0/8
with seven exact BF16 ties, memory lesion on 3/8, and internal lesion on 4/8.

C1y therefore passes exact addressing, coherent-value, and anatomical-recruitment
gates but fails passage-specific control. This does not reject coherent value transport:
the loss allowed a predictable common-mode shortcut. The next treatment should compare
the two paired branches directly in the same stored coordinate, select relay/output
cells by their live counterfactual difference, and reward preservation of the exact
target delta. Capacity, growth, and longer training remain deferred until that paired
signal survives the route.
