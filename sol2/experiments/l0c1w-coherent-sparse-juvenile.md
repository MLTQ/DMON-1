# L0-C1w: coherent sparse-recall juvenile

Status: architecture preregistered before implementation on 2026-08-05.

## Evidence and question

C1v removed cross-update state saturation and retained 60% of aggregate recall gradient,
yet semantic recall alignment remained zero and larger controls harmed every held-out
question. The current recall rotates stored sensory vectors through two independently
initialized transforms and softly averages every valid slot. Should a substantially
larger juvenile instead preserve one semantic coordinate frame and address a sparse,
recent working set?

## Architecture treatment

- Add chronological FIFO reads. Before recall, physical circular slots are reordered
  oldest-to-newest from the cursor, including after wraparound.
- Add a `coherent_residual` recall mode. Values are the stored sensory vector plus a
  bounded learned residual; recall output is the attended summary plus another bounded
  residual. Residual gain is fixed and serialized, initially `0.1`. Legacy learned-only
  recall remains the default and checkpoint-compatible.
- Add a nonnegative linear recency bias to attention scores, with newest slot age 0,
  and hard top-k selection before softmax. C1w uses bias `0.08` and top-k `16`.
- Capture attention, selected-slot count, entropy, and newest-slot mass through the
  existing opt-in trace. These are observations, not inference inputs.
- Initial juvenile geometry: 16 input, 1,024 memory, 512 compute, 128 relay, 16 output,
  hidden 128, 16/12 dendrites, three microsteps, eight output queries, eight rank-16
  prefix controls. Total initial cells: 1,696.
- Reserve utility/growth is staged after the memory/throughput preflight: the existing
  append-only relay mechanism remains available, while plastic-memory allocation and
  growth demand require a separate explicit state contract rather than pretending FIFO
  slots are already differentiating neurons.

## Invariants and gates

- With residual gain 0, recalled values equal a recall-gain-bounded attentive mixture of
  stored sensory coordinates; no learned value/output rotation remains.
- At positive residual gain, value/output parameters receive gradients while the
  identity route remains present.
- Top-k never selects invalid memory. Recency ordering remains exact before and after
  circular wrap; k at or above valid count reproduces dense softmax when bias is zero.
- Sparse selection, recency, and traces cannot connect memory directly to the effector;
  recalled drive still enters input tissue before recurrent/relay/output tissue.
- CPU tests cover deterministic grafting, state/trace equivalence, cyclic ordering,
  bounds, gradients, and legacy defaults.
- A 4090 preflight must fit with margin and report per-update time before a bounded run.
  The 2070S remains excluded.

The first learning run will retain C1v fresh training lifetimes to measure the new
anatomy without saturation. It must increase attention concentration and preserve
query/key gradients, direct recall separation, and semantic alignment before held-out
language control can license persistence or growth.
