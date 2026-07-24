"""DMON-1 substrate: NCA cells embedded in a diffusing resource field.

No target image appears anywhere in this file. Shape is expected to arise because
diffusion-limited uptake makes surface area valuable and intra-body transport makes
distance from a source expensive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- fixed perception kernels -------------------------------------------------

_IDENT = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_LAPLACE = [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]


def _kernels() -> torch.Tensor:
    ident = torch.tensor(_IDENT)
    sx = torch.tensor(_SOBEL_X) / 8.0
    sy = sx.T.contiguous()
    lap = torch.tensor(_LAPLACE) / 16.0
    return torch.stack([ident, sx, sy, lap])  # (4, 3, 3)


@dataclass
class SubstrateConfig:
    """All of M0's physics. Every field here is a knob on the ecology, not the creature."""

    channels: int = 16
    hidden: int = 128
    fire_rate: float = 0.5

    # --- metabolism (cell-side) ---
    # Costs are ~20x lower than the first draft. That draft's ecology could support a
    # body of about FOUR cells (see `feasibility.py`), which is not enough for
    # morphology of any kind to exist. Lowering cost rather than raising the field's
    # magnitude keeps `r` at O(1), which matters because `r` is fed straight into the
    # rule's perception alongside inputs of that scale.
    e_death: float = 0.01  # below this a cell is not alive
    e_max: float = 0.20
    e_tau: float = 0.002  # softness of the alive gate; small = closer to a hard threshold
    maintenance: float = 2e-4  # per-step cost of merely existing
    activity_cost: float = 1e-3  # per-step cost proportional to |state update|
    uptake_rate: float = 0.35  # fraction of local resource a live cell can absorb
    e_transport: float = 0.30  # ceiling on energy sharing between adjacent body cells
    effort_cost: float = 5e-4  # absorptive machinery is not free
    seed_energy: float = 0.20  # yolk: enough reserve to reach food before starving

    # --- resource field (environment-side) ---
    # `source_rate` is total emission per step across all sources, not per source cell;
    # `make_sources` normalises geometries to equal supply so that a contingency test
    # compares arrangement rather than abundance.
    field_diffusion: float = 0.40
    field_decay: float = 0.002
    source_rate: float = 1.00
    field_cap: float = 4.00  # below ~4 the clamp throttles a point source and raising
    #                          source_rate does nothing at all

    # --- curriculum (see `train_m0.py`) ---
    spread_start: float = 0.15  # sources begin near the seed...
    spread_end: float = 0.25  # ...and walk outward to here. Calibrated by experiment,
    #                            not by taste: rules survive from seed at 8.3x
    #                            break-even (0.25) and die at 3.6x (0.30) and 1.5x
    #                            (0.35). See feasibility.py's `margin`.
    spread_frac: float = 0.6  # fraction of training spent walking them out
    spread_stages: int = 8  # quantised, so "the world changed" is a discrete event

    def light_cone_ok(self, size: int, steps: int) -> bool:
        """Perception radius is 1, so information travels one cell per step."""
        return steps >= size

    def spread_at(self, progress: float) -> float:
        """Curriculum position. A seed's yolk cannot fund a 30-cell journey through
        barren ground, so early training puts food within reach and walks it out.

        `spread_end` is capped at a value verified feasible by `feasibility.py`, not at
        1.0. Push it further only after confirming a trained rule can actually bridge
        the gap — an infeasible final stage silently turns the last 40% of training into
        noise.

        Quantised into `spread_stages` discrete steps so a stage change is an event the
        trainer can react to. It has to react: a sample pool carries already-grown
        bodies across a curriculum, so established creatures just follow the food
        outward and the *seed-to-first-meal* problem at the new distance never enters
        the training distribution. That is not hypothetical — a rule trained to spread
        0.35 grew happily from a seed with food 4 cells away and died outright at 10,
        while its training log read mass 1100+ throughout."""
        t = min(1.0, max(0.0, progress / max(self.spread_frac, 1e-9)))
        t = round(t * self.spread_stages) / self.spread_stages
        return self.spread_start + t * (self.spread_end - self.spread_start)


class Substrate(nn.Module):
    """One coupled step of field, rule, and ledger.

    Channel 0 of the state is energy and is written *only* by the metabolic ledger.
    The learned rule may read it and may not write it; otherwise the creature can
    print its own currency and the economy stops meaning anything.

    The rule acts on the world through exactly two levers, both metabolic:

      channel 1  uptake effort       how hard this cell works to absorb resource
      channel 2  transport conductance  how freely it passes energy to neighbours

    It has no direct control over shape. Morphology is a *consequence* of where the
    rule chooses to invest: cells that do not fund themselves starve out, and the
    body extends only where transport pushes energy over the death threshold. This
    is the whole bet of M0 — that sculpting by differential investment is enough.
    """

    E, EFFORT, TRANSPORT = 0, 1, 2

    def __init__(self, cfg: SubstrateConfig | None = None):
        super().__init__()
        self.cfg = cfg or SubstrateConfig()
        c = self.cfg.channels

        self.register_buffer("kern", _kernels().unsqueeze(1))  # (4,1,3,3)

        # rule may not write channel 0
        chan_mask = torch.ones(1, c, 1, 1)
        chan_mask[:, self.E] = 0.0
        self.register_buffer("chan_mask", chan_mask)

        # gates start open-ish for uptake, near-closed for transport, so a fresh
        # seed feeds itself before it starts bleeding energy into the halo
        self.gate_bias = nn.Parameter(torch.tensor([2.0, -2.0]))

        self.rule = nn.Sequential(
            nn.Conv2d(4 * c + 4, self.cfg.hidden, 1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.hidden, c, 1, bias=False),
        )
        nn.init.zeros_(self.rule[-1].weight)  # start as a no-op; grow into behaviour

    # --- helpers --------------------------------------------------------------

    def _depthwise(self, t: torch.Tensor) -> torch.Tensor:
        """Apply all four kernels to every channel of t."""
        n = t.shape[1]
        w = self.kern.repeat(n, 1, 1, 1)  # (4n,1,3,3)
        return F.conv2d(t, w, padding=1, groups=n)

    def _laplace(self, t: torch.Tensor) -> torch.Tensor:
        w = self.kern[3:4]  # (1,1,3,3)
        return F.conv2d(t, w, padding=1)

    def _transport(self, e: torch.Tensor, conduct: torch.Tensor, body: torch.Tensor):
        """Conservative pairwise flux exchange between adjacent body cells.

        Every unit of energy that leaves one cell arrives in exactly one neighbour.
        `sum(de)` is identically zero by construction, so the creature cannot mint
        currency no matter what it does with its conductance.

        The previous formulation — `e_transport * conduct * laplace(e) * body` — did
        not have this property. The laplacian kernel sums to zero, but modulating it
        by a *per-cell* conductance breaks the antisymmetry: cell i gained using
        `conduct_i` while its neighbours lost using theirs, so a rule that raised
        conductance where the laplacian was positive and lowered it where negative
        created energy from nothing. It found that in 4k iterations and grew to a
        stable body on a grid with no food in it at all.

        Pair conductance is `min(c_i, c_j)`: a channel is as narrow as its narrowest
        point, and either cell can unilaterally refuse to share — which is the
        primitive M1's kin discrimination needs.
        """
        k = self.cfg.e_transport * conduct * body / 4.0  # /4: explicit-scheme stability
        de = torch.zeros_like(e)
        for axis in (-1, -2):
            ea, eb = e.narrow(axis, 0, e.shape[axis] - 1), e.narrow(axis, 1, e.shape[axis] - 1)
            ka, kb = k.narrow(axis, 0, k.shape[axis] - 1), k.narrow(axis, 1, k.shape[axis] - 1)
            f = torch.minimum(ka, kb) * (eb - ea)  # + means b gives to a
            pad_lo = (0, 0, 1, 0) if axis == -2 else (1, 0)
            pad_hi = (0, 0, 0, 1) if axis == -2 else (0, 1)
            de = de + F.pad(f, pad_hi) - F.pad(f, pad_lo)
        return de

    def seed(self, batch: int, size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """One live cell at the centre, empty field."""
        c = self.cfg.channels
        x = torch.zeros(batch, c, size, size, device=device)
        x[:, self.E, size // 2, size // 2] = self.cfg.seed_energy
        r = torch.zeros(batch, 1, size, size, device=device)
        return x, r

    def alive(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :1] > self.cfg.e_death).float()

    def soft_alive(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x[:, :1] - self.cfg.e_death) / self.cfg.e_tau)

    def _body_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Live cells plus their halo. The halo is where growth is permitted."""
        e = x[:, :1]
        return (F.max_pool2d(e, 3, stride=1, padding=1) > self.cfg.e_death).float()

    # --- the step -------------------------------------------------------------

    def step(
        self, x: torch.Tensor, r: torch.Tensor, sources: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg

        # 1. environment moves first, indifferently
        r = r + cfg.field_diffusion * self._laplace(r) - cfg.field_decay * r
        r = r + cfg.source_rate * sources
        r = r.clamp(0.0, cfg.field_cap)

        body = self._body_mask(x)
        soft = self.soft_alive(x)

        # 2. perceive self and world
        px = self._depthwise(x)
        pr = self._depthwise(r)
        dx = self.rule(torch.cat([px, pr], dim=1))

        # 3. rule cannot mint energy, cannot act outside the body, fires stochastically
        fire = (torch.rand_like(x[:, :1]) < cfg.fire_rate).float()
        dx = dx * self.chan_mask * fire * body
        x = x + dx

        # 4. ledger
        e = x[:, self.E : self.E + 1]
        effort = torch.sigmoid(x[:, self.EFFORT : self.EFFORT + 1] + self.gate_bias[0])
        conduct = torch.sigmoid(x[:, self.TRANSPORT : self.TRANSPORT + 1] + self.gate_bias[1])

        # A cell cannot absorb more than it has room for. Without this it strips the
        # field, keeps `e_max`, and annihilates the remainder — which both destroys
        # energy and hides how scarce the world really is. It also creates the incentive
        # the whole design wants: a full cell must move energy onward before it can eat
        # again, so harvesting *requires* transport rather than merely permitting it.
        room = (cfg.e_max - e).clamp(min=0.0)
        uptake = torch.minimum(soft * cfg.uptake_rate * effort * r, room)
        r = (r - uptake).clamp(min=0.0)

        cost = (
            cfg.maintenance * soft
            + cfg.effort_cost * soft * effort
            + cfg.activity_cost * dx.abs().mean(1, keepdim=True)
        )
        e = e + uptake - cost

        # transport: energy moves through the body, so reach costs something.
        # Conservative pairwise exchange — see `_transport`. Energy is redistributed
        # here and never created; the only inflow to the ledger is `uptake`.
        e = e + self._transport(e, conduct, body)
        e = e.clamp(0.0, cfg.e_max)

        x = torch.cat([e, x[:, 1:]], dim=1) * body
        return x, r

    def rollout(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        sources: torch.Tensor,
        steps: int,
        tail: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns final state, final field, and mean soft-alive mass over the last
        `tail` steps (so survival has to be sustained rather than spiked at T)."""
        tail = tail or max(1, steps // 8)
        acc = []
        for i in range(steps):
            x, r = self.step(x, r, sources)
            if i >= steps - tail:
                acc.append(self.soft_alive(x).sum(dim=(1, 2, 3)))
        return x, r, torch.stack(acc).mean(0)


# --- source geometries --------------------------------------------------------


def make_sources(
    name: str,
    batch: int,
    size: int,
    device=None,
    normalize: bool = True,
    spread: float = 1.0,
) -> torch.Tensor:
    """Where food comes from. Moving these and retraining is the M0 falsification test.

    `normalize=True` scales every geometry to the same *total* emission, so that
    `source_rate` means "food entering the world per step" rather than "food per source
    cell". This is not cosmetic. Unnormalised, these geometries differ by a factor of
    248 in total supply — `west` has one source cell and `ring` has 248 — so a
    contingency test across them would be comparing abundance, not arrangement, and
    would report a confident PASS for entirely the wrong reason.

    Pass `normalize=False` only when abundance is deliberately the independent variable.

    `spread` moves every source along the line from the seed at the centre out to its
    full position: 0 puts food on top of the seed, 1 is the geometry as defined. This is
    the distance curriculum ARCHITECTURE.md §M0 calls for — a seed's yolk cannot fund a
    30-cell journey, so early training has to put food within reach and walk it outward.
    """
    m = size // 2
    pts = source_points(name, size)

    s = torch.zeros(batch, 1, size, size, device=device)
    for y, x in pts:
        yy = min(size - 1, max(0, int(round(m + spread * (y - m)))))
        xx = min(size - 1, max(0, int(round(m + spread * (x - m)))))
        s[:, 0, yy, xx] = 1.0

    if normalize:
        s = s / s.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    return s


def source_points(name: str, size: int) -> list[tuple[int, int]]:
    """Source cell positions at full spread. Separated out so a curriculum can move
    them toward the seed without each geometry needing its own special case."""
    m, q = size // 2, size // 4
    if name == "center":
        return [(m, m)]
    if name == "west":
        return [(m, 2)]
    if name == "poles":
        return [(2, m), (size - 3, m)]
    if name == "corners":
        return [(y, x) for y in (q, size - q) for x in (q, size - q)]
    if name == "ring":
        rad = m - 3
        n = max(8, int(2 * math.pi * rad * 1.5))
        return [
            (
                int(round(m + rad * math.sin(2 * math.pi * i / n))),
                int(round(m + rad * math.cos(2 * math.pi * i / n))),
            )
            for i in range(n)
        ]
    raise ValueError(f"unknown source geometry: {name}")


# --- morphology descriptors ---------------------------------------------------


def descriptors_per_sample(x: torch.Tensor, cfg: SubstrateConfig) -> dict[str, torch.Tensor]:
    """Same statistics as `descriptors`, kept per batch element — each value is `(B,)`.

    The probe needs within-run spread, and the contingency test needs a noise floor;
    both are destroyed by averaging over the batch before they see it.
    """
    a = (x[:, :1] > cfg.e_death).float()
    b, _, h, w = a.shape
    mass = a.sum(dim=(1, 2, 3))

    eroded = -F.max_pool2d(-a, 3, stride=1, padding=1)
    perim = (a - eroded).clamp(min=0).sum(dim=(1, 2, 3))
    compact = perim / mass.clamp(min=1).sqrt()

    yy, xx = torch.meshgrid(
        torch.arange(h, device=a.device), torch.arange(w, device=a.device), indexing="ij"
    )
    yy = yy.float()[None, None]
    xx = xx.float()[None, None]
    tot = mass.clamp(min=1)[:, None, None, None]
    cy = (a * yy).sum(dim=(1, 2, 3), keepdim=True) / tot
    cx = (a * xx).sum(dim=(1, 2, 3), keepdim=True) / tot
    gyr = ((a * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum(dim=(1, 2, 3)) / mass.clamp(min=1)).sqrt()

    counts = []
    for box in (2, 4, 8):
        occ = (F.max_pool2d(a, box, stride=box) > 0).float().sum(dim=(1, 2, 3))
        counts.append((box, occ))
    logs_n = torch.stack([torch.log(c.clamp(min=1)) for _, c in counts])
    logs_s = torch.tensor([-torch.log(torch.tensor(float(b_))) for b_, _ in counts])
    logs_s = logs_s[:, None].to(a.device)
    sm = logs_s.mean()
    nm = logs_n.mean(0, keepdim=True)
    dim = ((logs_s - sm) * (logs_n - nm)).sum(0) / ((logs_s - sm) ** 2).sum(0).clamp(min=1e-8)

    return {
        "mass": mass,
        "compactness": compact,
        "gyration": gyr,
        "box_dim": dim,
    }


def descriptors(x: torch.Tensor, cfg: SubstrateConfig) -> dict[str, float]:
    """Shape statistics used to decide whether morphology is contingent on ecology.

    Deliberately cheap and interpretable. If these do not separate across source
    geometries, M0 has failed regardless of how the renders look.
    """
    return {k: v.mean().item() for k, v in descriptors_per_sample(x, cfg).items()}
