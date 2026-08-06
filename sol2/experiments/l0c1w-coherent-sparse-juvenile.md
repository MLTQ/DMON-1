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
- Add a training-only addressing KL at weight `1.0`. Its detached target ranks the
  current exposure's frozen-Qwen token features by similarity to the passage-visible
  teacher effect, using temperature `0.1`; the student surface is the raw content score
  before recency and top-k, so excluded slots can still teach query/key parameters.
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
anatomy without saturation. It preserves C1v's effect KL `1`, causal contrast `1`,
memory semantic credit `4`, direct recall credit `4`, LR `0.001`, and recall/sensor/
effector multipliers `20/4/1`; all other losses remain zero. It must reduce addressing
KL, increase selected teacher mass, and preserve query/key gradients, direct recall
separation, and semantic alignment before held-out language control can license
persistence or growth.

This addressing amendment was frozen after the trace implementation and before any
C1w optimizer initialization or GPU preflight.

## Fit amendment

The first 4090 fit gate rejected the 1,696-cell geometry before completing update 1.
It reached approximately `22.94 GiB` in use and failed on a further 2 MiB cellular
transition allocation; no metrics artifact or completed optimizer update was produced.
The frozen Qwen and prefix-training graph leave insufficient activation margin for that
geometry on a 24 GiB card.

The second preflight used 16 input, 512 memory, 256 compute, 64 relay, and 16 output
cells at hidden 128: 864 total cells. It completed update 1, including the backward and
optimizer step, but matched evaluation rose to approximately `23.77 GiB` in use. The
evaluation was stopped after its development arm because less than 1 GiB of device
margin is not acceptable for a multi-update run. Its update-1 addressing KL was
`0.4172`, selected teacher mass `0.411`, effective attention `14.95` slots, newest-four
mass `0.375`, and direct-recall alignment `0.140`; all four recall components received
gradients.

The final fit treatment is 16 input, 384 memory, 256 compute, 64 relay, and 16 output
cells at hidden 128: 736 total cells. Relative to C1v, memory, compute, and relay tissue
are each exactly 4x larger and input/output tissue 2x larger. It preserves all
addressing, objective, optimizer, and language-interface treatments above. This final
capacity amendment was frozen before its optimizer was initialized.

The 736-cell preflight completed its full development and held-out causal suite. Peak
PyTorch allocation/reservation was `21,441,373,184`/`21,581,791,232` bytes (19.97/20.10
GiB), leaving roughly 4.4 GiB of physical 4090 headroom. The organism has 1,293,586
trainable parameters; Qwen had zero trainable parameters and zero gradient tensors.

At update 1, addressing KL was `0.41715`, selected teacher mass `0.41148`, effective
attention `14.95` slots, newest-four mass `0.37499`, and direct-recall alignment
`0.13956` at separation RMS `0.01940`. Query/key/value/output gradient RMS was
`2.876e-3`/`2.659e-3`/`1.863e-5`/`2.447e-5`. Sensor, recall, and the initially opening
effector received gradients; recurrent/connectome/transport gradients remained zero on
the first update behind the exact-zero effector initialization and must become nonzero
in the bounded learning run. The preflight licenses one fresh 25-update curve at this
geometry; its single-update held-out behavior is only an initialization observation,
not a causal-memory result.
