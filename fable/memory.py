"""F9a: the associative memory organ — minimal retrieval primitive.

Ring-buffer writes (the mirror ring's discipline: no learned write
addressing, nothing for write-side optimization to game), content-based
reads (learned query, cosine over stored keys, sharpened softmax), read
vector entering the field as drive on a dedicated memory-port cell group.

Output-gated to zero at init so the graft is silent — the organism starts
behaviorally identical to a memory-less creature (contract-checked) and
must *learn* to consult the organ.

Why this shape: F6 rounds 1-2 showed no architecture at this scale invents
retrieval from raw recurrence/attention embedded sparsely in text. NTM
(arXiv:1410.5401) is the existence proof that with addressable memory as a
primitive, small networks learn retrieval — they only have to learn to use
it. This is the smallest organ that supplies the primitive. Lifetime
caveats (allocation, forgetting — the DNC additions) are deliberately out
of scope for a 12k-update test and mandatory before any never-reset use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SLOTS = 48
WIDTH = 64


class MemoryOrgan(nn.Module):
    def __init__(self, hidden: int, n_slots: int = N_SLOTS, width: int = WIDTH):
        super().__init__()
        self.n_slots = n_slots
        self.width = width
        self.key_proj = nn.Linear(2 * hidden, width)
        self.val_proj = nn.Linear(2 * hidden, width)
        self.query_proj = nn.Linear(hidden, width)
        self.read_out = nn.Linear(width, hidden)
        self.sharpen = nn.Parameter(torch.tensor(5.0))
        # Round 2: live at init (0.1). The zero-gate version was a bootstrap
        # trap — random projections make reads noise, so the gradient pinned
        # the gate at zero and the projections never learned (round-1 gate
        # after 12k updates: −0.011). NTM's memory path is ungated and live
        # from step one; deviating from that strangled the organ.
        self.gate = nn.Parameter(torch.tensor(0.1))

    def empty(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(batch, self.n_slots, self.width, device=device)
        return z, z.clone()

    def write(self, keys: torch.Tensor, vals: torch.Tensor, cursor: int,
              context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ring write at cursor % n_slots. context: [B, 2H]."""
        slot = torch.tensor([cursor % self.n_slots], device=keys.device)
        k = self.key_proj(context).unsqueeze(1)
        v = self.val_proj(context).unsqueeze(1)
        return keys.index_copy(1, slot, k), vals.index_copy(1, slot, v)

    def read(self, keys: torch.Tensor, vals: torch.Tensor,
             query_state: torch.Tensor) -> torch.Tensor:
        """Content read. query_state: [B, H] → drive [B, H]."""
        q = F.normalize(self.query_proj(query_state), dim=-1)
        k = F.normalize(keys, dim=-1)
        scores = torch.einsum("bw,bsw->bs", q, k) * self.sharpen
        w = F.softmax(scores, dim=-1)
        read = torch.einsum("bs,bsw->bw", w, vals)
        # tanh bound: with a live gate this is otherwise an unbounded learned
        # scale path into the recurrence (the week's detonation pattern)
        return self.gate * torch.tanh(self.read_out(read))
