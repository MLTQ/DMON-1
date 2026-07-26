"""The streaming lattice: a persistent NCA with I/O ports and mirror cells.

The creature never resets. State carries forward forever; only the autograd graph is
truncated. There is no episode, no rollout, no seed — those belong to the bookmarked
morphology work and their absence here is the point.

Layout is a *port* model rather than a pipeline. Input and output share a small central
region, and the surrounding lattice is memory and computation that stimulus diffuses
into and back out of. The alternative — input at one edge, output at the other — makes
readout latency equal to the transport distance, so the creature could not respond to a
character until many ticks after it arrived. Here the periphery is association cortex,
not a wire.

Three cell populations:

    input cells   the current token's embedding, written by the stream
    mirror cells  a rolling buffer of recent tokens, written by the stream
    output cells  read out to logits

**Input and mirror cells are write-only from the stream.** The rule's update is masked
off at those coordinates. This is not a detail: if the prediction pathway could write
its own targets and its own history, the trivial optimum is to flatten them — error
zero, learning zero. That is the BYOL/SimSiam collapse arriving through a new door, and
it looks like success in a loss curve. See ARCHITECTURE.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

_IDENT = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_LAPLACE = [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]


def _kernels() -> torch.Tensor:
    ident = torch.tensor(_IDENT)
    sx = torch.tensor(_SOBEL_X) / 8.0
    sy = sx.T.contiguous()
    lap = torch.tensor(_LAPLACE) / 16.0
    return torch.stack([ident, sx, sy, lap])


@dataclass
class LatticeConfig:
    size: int = 32
    channels: int = 32
    hidden: int = 256
    vocab: int = 65

    micro: int = 3  # lattice steps per input tick
    mirror_len: int = 24  # tokens of recent history held in-substrate
    port: int = 4  # side of the square I/O port
    fire_rate: float = 1.0  # 1.0 = synchronous. Stochastic firing is a morphology
    #                         trick; here it only adds gradient noise.
    state_clip: float = 8.0  # a never-resetting state can drift without a bound

    # --- local objective (see `_cell` and `micro_step`) ---
    # The readout is a single small patch, so it supplies exactly one objective and it
    # lives at one point. Measured consequence: cells beyond radius ~5 are active and
    # contribute 0.004 bpc, and a 6x change in `micro` does not move that boundary — so
    # the limit is incentive, not transport. A local self-prediction objective gives
    # every cell work that does not route through the readout.
    local_pred: float = 0.0  # weight on local next-state prediction
    var_weight: float = 1.0  # anti-collapse. Predicting your own next state has the
    #                          trivial optimum "be constant" — every cell can reach it
    #                          independently, and it reports perfect error. The variance
    #                          hinge is what makes stillness unprofitable.
    var_target: float = 0.3

    def __post_init__(self):
        if self.mirror_len > self.size:
            raise ValueError("mirror_len must fit along one lattice edge")


class StreamingLattice(nn.Module):
    """Persistent gated NCA with stream-driven input, mirror and output regions."""

    def __init__(self, cfg: LatticeConfig | None = None):
        super().__init__()
        self.cfg = cfg or LatticeConfig()
        c, h, s = self.cfg.channels, self.cfg.hidden, self.cfg.size

        self.register_buffer("kern", _kernels().unsqueeze(1))
        self.embed = nn.Embedding(self.cfg.vocab, c)

        # --- gated cell. `x = x + dx` cannot hold a state under a continuous stream:
        # an additive rule must re-derive whatever it wants to remember on every tick.
        p_in = 4 * c
        self.trunk = nn.Sequential(nn.Conv2d(p_in, h, 1), nn.ReLU())
        self.to_z = nn.Conv2d(h, c, 1)  # update gate
        self.to_r = nn.Conv2d(h, c, 1)  # reset gate
        self.to_c = nn.Conv2d(h + c, c, 1, bias=False)  # candidate
        # Local objective head: from this cell's neighbourhood, predict what this cell
        # is about to become. The update depends on the 3x3 neighbourhood, so getting
        # this right requires modelling neighbours — which is work a distant cell can
        # do without any path to the readout.
        self.to_pred = nn.Conv2d(h, c, 1)
        nn.init.zeros_(self.to_c.weight)  # start inert
        nn.init.constant_(self.to_z.bias, -2.0)  # start closed: remember by default

        # --- regions. Port at centre; mirror along the row above it.
        p = self.cfg.port
        y0 = s // 2 - p // 2
        x0 = s // 2 - p
        self.in_slice = (slice(y0, y0 + p), slice(x0, x0 + p))
        # Output sits ADJACENT to input, never on top of it. Co-locating them looks
        # natural — one port, minimal latency — but input cells are stream-driven and
        # the rule cannot write them, so a readout placed there reads the current
        # token's embedding and nothing else. That configuration is a bigram model with
        # a lattice attached for decoration, and it scores exactly what an inert
        # lattice scores. Adjacency keeps latency at one micro-step without handing the
        # readout the answer.
        self.out_slice = (slice(y0, y0 + p), slice(x0 + p, x0 + 2 * p))
        my = max(0, y0 - 2)
        mx0 = max(0, s // 2 - self.cfg.mirror_len // 2)
        self.mirror_slice = (slice(my, my + 1), slice(mx0, mx0 + self.cfg.mirror_len))

        # Cells the stream drives. The rule may READ these — they are perceived like any
        # other cell — but may not WRITE them.
        driven = torch.zeros(1, 1, s, s)
        driven[:, :, self.in_slice[0], self.in_slice[1]] = 1.0
        driven[:, :, self.mirror_slice[0], self.mirror_slice[1]] = 1.0
        self.register_buffer("driven", driven)

        # Guard, not a comment: if the readout can see stream-driven cells, the model
        # can score without using the lattice at all and the inert baseline will match
        # it exactly. That happened; it cost a 60k-tick run to notice.
        overlap = driven[:, :, self.out_slice[0], self.out_slice[1]].max().item()
        if overlap > 0:
            raise ValueError(
                "output region overlaps stream-driven cells — the readout would see "
                "the input directly and the lattice would be bypassed"
            )

        self.readout = nn.Linear(c * p * p, self.cfg.vocab)

    # --- state ----------------------------------------------------------------

    def blank_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.channels, self.cfg.size, self.cfg.size, device=device)

    def _perceive(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        w = self.kern.repeat(n, 1, 1, 1)
        return F.conv2d(x, w, padding=1, groups=n)

    def _cell(self, x: torch.Tensor):
        h = self.trunk(self._perceive(x))
        z = torch.sigmoid(self.to_z(h))
        rst = torch.sigmoid(self.to_r(h))
        cand = torch.tanh(self.to_c(torch.cat([h, rst * x], dim=1)))
        pred = self.to_pred(h) if self.cfg.local_pred > 0 else None
        return z * (cand - x), pred

    def micro_step(self, x: torch.Tensor):
        dx, pred = self._cell(x)
        if self.cfg.fire_rate < 1.0:
            dx = dx * (torch.rand_like(x[:, :1]) < self.cfg.fire_rate).float()
        # the stream owns its cells; the rule cannot write them
        x_new = (x + dx * (1.0 - self.driven)).clamp(
            -self.cfg.state_clip, self.cfg.state_clip
        )
        aux = None
        if pred is not None:
            # Stop-gradient on the target. Without it the cell can satisfy its own
            # prediction by changing what it becomes rather than by predicting better,
            # which is the same asymmetry failure the mirror cells are protected from.
            free = 1.0 - self.driven  # driven cells are trivially predictable
            err = ((pred - x_new.detach()) ** 2 * free).sum() / free.sum().clamp(min=1) / x.shape[1]
            std = x_new.flatten(2).std(dim=2).mean(0)  # per-channel spread across cells
            collapse = F.relu(self.cfg.var_target - std).mean()
            aux = err + self.cfg.var_weight * collapse
        return x_new, aux

    # --- the stream interface -------------------------------------------------

    def write_input(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """Stimulus arrives. Detached: the stream is not a function of the creature."""
        e = self.embed(token)  # (B, C)
        patch = e[:, :, None, None].expand(-1, -1, self.cfg.port, self.cfg.port)
        x = x.clone()
        x[:, :, self.in_slice[0], self.in_slice[1]] = patch.detach()
        return x

    def write_mirror(self, x: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Roll recent tokens into the mirror row.

        `history` is (B, mirror_len), most recent first. Written detached and never
        updated by the rule — see the module docstring on why that is load-bearing
        rather than tidy.
        """
        e = self.embed(history)  # (B, L, C)
        x = x.clone()
        x[:, :, self.mirror_slice[0], self.mirror_slice[1]] = (
            e.permute(0, 2, 1).unsqueeze(2).detach()
        )
        return x

    def read_output(self, x: torch.Tensor) -> torch.Tensor:
        patch = x[:, :, self.out_slice[0], self.out_slice[1]]
        return self.readout(patch.reshape(patch.shape[0], -1))

    def tick(
        self, x: torch.Tensor, token: torch.Tensor, history: torch.Tensor,
        return_aux: bool = False,
    ):
        """One input tick: stimulus in, `micro` lattice steps, logits out."""
        x = self.write_input(x, token)
        x = self.write_mirror(x, history)
        aux_total = 0.0
        for _ in range(self.cfg.micro):
            x, aux = self.micro_step(x)
            if aux is not None:
                aux_total = aux_total + aux / self.cfg.micro
        if return_aux:
            return x, self.read_output(x), aux_total
        return x, self.read_output(x)

    # --- growth ---------------------------------------------------------------

    def grown(self, new_size: int) -> tuple["StreamingLattice", callable]:
        """Return a larger lattice sharing this one's weights, plus a state migrator.

        This is the operation that justifies choosing cellular automata. The rule is
        shared and local, so **the parameter count does not change** — every weight
        transfers unmodified and the larger creature is the same creature with more
        room. No retraining, no re-initialisation, no architecture change.

        New cells start at zero state. They are not noise and they are not copies; they
        are unrecruited substrate, and whether the creature actually recruits them is
        precisely what S2 tests rather than assumes.

        Regions are re-derived at the new centre, so the migrator translates old state
        to keep the port aligned. Without that the creature would wake up with its
        input arriving somewhere it has never sensed.
        """
        if new_size < self.cfg.size:
            raise ValueError("grown() does not shrink")
        from dataclasses import replace as _replace

        bigger = StreamingLattice(_replace(self.cfg, size=new_size))
        # `driven` is a size-shaped buffer describing where the stream writes; it is
        # rebuilt by the constructor at the new geometry and must not be copied across.
        # Everything else — every learnable weight — transfers verbatim, which is the
        # whole point.
        weights = {k: v for k, v in self.state_dict().items() if k != "driven"}
        missing, unexpected = bigger.load_state_dict(weights, strict=False)
        if unexpected or [m for m in missing if m != "driven"]:
            raise RuntimeError(
                f"weights did not transfer cleanly (missing={missing}, "
                f"unexpected={unexpected})"
            )

        off = (new_size - self.cfg.size) // 2

        def migrate(x: torch.Tensor) -> torch.Tensor:
            y = bigger.blank_state(x.shape[0], x.device)
            y[:, :, off : off + self.cfg.size, off : off + self.cfg.size] = x
            return y

        return bigger, migrate

    # --- introspection --------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def silence(self):
        """Inert rule, for the null model. Shutting the update gate — not zeroing the
        candidate, which would give `z*(0-x)` and actively decay the state."""
        nn.init.constant_(self.to_z.bias, -20.0)
        nn.init.zeros_(self.to_z.weight)
        nn.init.zeros_(self.to_c.weight)
        return self
