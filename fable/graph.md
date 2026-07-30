# graph.py

## Purpose

Wiring and forward message-passing for the sparse directed connectome. Nothing
else — no credit transport, no probes.

## Components

- `DendriteGraph` — `sources [N,K]` buffer (wired once from `cfg.seed`, its own
  generator, so topology is independent of global RNG); `logit [N,K]`, the only
  per-edge parameters; shared bias-free `query/key/value` H×H maps;
  `aggregate(h, targets)`; `output_reachable_from_input` BFS used by the smoke
  test and after growth.

## Decisions

- **value/key/query applied to `h` once, then gathered.** grok applied them to
  the gathered `[B,N,K,H]` tensor — mathematically identical (bias-free linear
  maps commute with gather) and K× more FLOPs and retained memory (grok debt
  #1). This was the dominant reason the creature ran 7–13× slower than its GRU.
- **One per-edge logit, not two.** grok's `weights` + `bias` were exact
  duplicates, doubly weight-decayed and inconsistently consulted by the
  structural code (debt #9).
- **Output cells are sinks** (no cell samples them as a source) so readout
  state can never recirculate; each output cell's slot 0 is force-wired to an
  input cell (with dedupe — grok's forced overwrite could silently duplicate a
  source, debt #15) so sensory→output reachability holds at init.
- `aggregate` takes a `targets` index so mirror cells are never computed for
  (grok computed full messages + rule for mirrors and discarded them — 25% of
  compute, debt #2).

## Contracts

- `sources[t]` values are valid cell indices < N and never output cells.
- `aggregate` returns `[B, len(targets), H]`; grad flows to logit, q/k/v, and
  through `h`.
- Growth may only append rows to `sources`/`logit` and reassign donated slots
  (see `grow.py`); it never reorders existing rows.
