# `graph.py`

## Purpose

Moves information through a sparse directed connectome while separating installed
anatomy from dormant capacity reserved for future growth.

## Components

### `DirectedConnectome`

- `sources[N,K]` names candidate source cells for every target-owned slot.
- `active[N,K]` determines which slots are anatomical; dormant slots carry no message.
- `edge_logit[N,K]` provides private per-edge attention bias.
- Query, key, and value maps share the bounded/unbounded treatment without changing
  parameter count.
- Bounded mode projects query/key vectors into the unit ball without amplifying vectors
  already below unit norm, then uses bounded temperature, edge bias, and values.
- Reachability and output-sink checks are executable contracts.

## Contracts

- Output cells are never sources, preventing decoder state from recirculating.
- Every target begins with at least one active dendrite.
- Dormant slots receive exactly zero softmax mass.
- Growth activates dormant slots; it never needs to delete a working edge.
- Bounded mode constrains every shared graph operator and never normalizes a small
  activation upward.
