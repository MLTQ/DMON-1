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
