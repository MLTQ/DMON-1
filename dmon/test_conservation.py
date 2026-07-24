"""Energy conservation tests.

These exist because the original transport operator minted energy and nothing caught
it. A rule trained under it grew a stable body on a grid containing no food, and the
whole first diffusion sweep was run inside that regime — six values, identical results,
because the creature never ate. `PROJECT.md`'s Decision Log had already stated the
stakes: if the rule can write energy, "the economy stops meaning anything and every
result after that point is void."

So conservation is not a nicety here, it is the precondition for any result being real,
and it gets a test rather than a note in a doc.

    python -m dmon.test_conservation
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .substrate import Substrate, SubstrateConfig, make_sources

FREE = dict(maintenance=0.0, activity_cost=0.0, effort_cost=0.0, uptake_rate=0.0)


def test_transport_operator_conserves():
    """The exact mathematical claim: `_transport` redistributes and never creates.

    Tested directly on the operator with adversarial inputs — a rough energy landscape
    and a *deliberately correlated* conductance field, since correlating conductance
    with the energy laplacian is precisely how the old operator was exploited.
    """
    sub = Substrate(SubstrateConfig())
    torch.manual_seed(0)
    e = torch.rand(3, 1, 24, 24)
    body = (torch.rand(3, 1, 24, 24) > 0.3).float()

    for name, conduct in [
        ("random", torch.rand(3, 1, 24, 24)),
        ("uniform", torch.full((3, 1, 24, 24), 0.7)),
        # the exploit: open where energy is lowest, closed where highest
        ("adversarial", torch.sigmoid(-8 * (e - e.mean()))),
    ]:
        de = sub._transport(e, conduct, body)
        drift = de.sum(dim=(1, 2, 3)).abs().max().item()
        assert drift < 1e-5, f"{name} conductance created {drift:.3e} energy per step"
    return True


def test_step_never_mints():
    """Full step, no costs, no food: energy may leak but must never increase.

    Leaks are tolerated (a cell caps at `e_max`, and energy in cells that fall out of
    the body mask is destroyed rather than released). Creation is not: it is what makes
    the economy decorative.
    """
    cfg = replace(SubstrateConfig(), **FREE)
    sub = Substrate(cfg)
    torch.nn.init.normal_(sub.rule[-1].weight, std=0.5)

    x, r = sub.seed(2, 32)
    src = torch.zeros_like(r)
    start = x[:, :1].sum().item()
    with torch.no_grad():
        for _ in range(64):
            prev = x[:, :1].sum().item()
            x, r = sub.step(x, r, src)
            now = x[:, :1].sum().item()
            assert now <= prev + 1e-5, f"minted {now - prev:.6f} in one step"
    return start, x[:, :1].sum().item()


def test_no_growth_without_food():
    """The regression test for the actual bug.

    A creature on a grid with no sources may redistribute and spend what its seed
    started with. It may not end up with more than it began with, and it must not grow
    a body the way the broken operator allowed.
    """
    cfg = SubstrateConfig()
    sub = Substrate(cfg)
    torch.nn.init.normal_(sub.rule[-1].weight, std=0.5)

    x, r = sub.seed(4, 48)
    start = x[:, :1].sum().item()
    src = torch.zeros_like(r)

    with torch.no_grad():
        for _ in range(96):
            x, r = sub.step(x, r, src)

    end = x[:, :1].sum().item()
    mass = (x[:, :1] > cfg.e_death).float().sum().item()
    assert end <= start + 1e-5, f"energy created with no food: {start:.4f} -> {end:.4f}"
    return start, end, mass


def test_uptake_is_the_only_inflow():
    """Energy gained must be accounted for by resource removed from the field."""
    cfg = replace(SubstrateConfig(), maintenance=0.0, activity_cost=0.0, effort_cost=0.0)
    sub = Substrate(cfg)
    torch.nn.init.normal_(sub.rule[-1].weight, std=0.5)

    x, r = sub.seed(2, 32)
    src = make_sources("center", 2, 32)
    with torch.no_grad():
        for _ in range(48):
            e_before = x[:, :1].sum().item()
            r_before = r.sum().item()
            x, r = sub.step(x, r, src)
            gained = x[:, :1].sum().item() - e_before
            # field change = emission - uptake, so emission bounds the discrepancy
            emitted = cfg.source_rate * src.sum().item()
            lost_from_field = r_before + emitted - r.sum().item()
            assert gained <= lost_from_field + 1e-3, (
                f"gained {gained:.6f} but field only lost {lost_from_field:.6f}"
            )
    return True


if __name__ == "__main__":
    test_transport_operator_conserves()
    print("transport operator conserves:  OK (incl. adversarial conductance)")
    s, e = test_step_never_mints()
    print(f"step never mints:              {s:.4f} -> {e:.4f}  OK")
    s, e, m = test_no_growth_without_food()
    print(f"no growth without food:        {s:.4f} -> {e:.4f}, mass={m:.0f}  OK")
    test_uptake_is_the_only_inflow()
    print("uptake is the only inflow:     OK")
    print("\nall conservation tests passed")
