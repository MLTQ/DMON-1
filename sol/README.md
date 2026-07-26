# SOL

SOL is an experimental line for the central creature: a persistent neural cellular
organism that is continuously stimulated, metabolizes input, remembers events long
enough for delayed reward, and continuously emits multimodal outputs.

This first slice is the **character organ**. It asks a deliberately narrow question:

> Can a homogeneous cellular rule running on a sparse directed connectome learn
> next-character behavior while its internal state, energy, and event memory persist
> across an uninterrupted stream?

It is “NanoGPT-like” only at the behavioral boundary—characters in, next-character
probabilities out. It is not a transformer imitation.

## Architecture

- Characters stimulate a small sensory population.
- Every cell owns private recurrent state but uses the same GRU rule.
- Targets read only their named dendrite slots. Synapses are directed, signed, and
  trainable; there is no convolutional neighborhood exchange.
- Designated output cells emit character logits after every input.
- Exact truncated BPTT sends task credit backward through the recurrent graph.
- Persistent eligibility traces remember which cells participated in preceding events;
  delayed scalar reward modulates those traces on the following tick, including across
  optimizer boundaries.
- Externally caused stimulation propagates along measured edge flow. Stimulation and
  energy decay when input stops.

Topology is fixed in this milestone. Adding axon growth before proving transport and
credit would make every failure ambiguous.

## Run

```bash
python -m pytest sol/test_sol.py
python -m sol.train --updates 300
python -m sol.train --file path/to/corpus.txt --device cuda --updates 10000
```

## Falsification

The prototype has failed if any of these are true:

1. A tiny repeated corpus cannot be overfit.
2. Directed synapses or early retained states receive zero gradient.
3. Resetting state has no effect on predictions.
4. Delayed reward produces the same update with and without an eligibility trace.
5. Energy does not fall during unstimulated ticks.

## Deliberate omissions

- No cell birth, death, or topology mutation yet.
- No visual or audio organs yet.
- Energy modulates cell computation but does not yet govern viability.
- Eligibility is neural event memory, not a full online parameter-gradient algorithm.
  Exact gradients remain truncated to the current optimizer window.
