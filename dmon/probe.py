"""The legibility probe: does internal metabolic state have a visible signature?

Everything above M2 rests on this one assumption and none of it has tested it. M2 wants
mood readable from outside. M3's handicap wants a display that degrades when the body
is failing. M4 wants speech that goes urgent because cells are actually starving. Those
are one claim in three costumes.

The experiment: take a trained rule, grow it to steady state, then *ramp* the resource
supply to zero and watch whether shape moves before death. Ramp rather than cut,
because a cut conflates shock with starvation.

The reported metric is **warning time** — how many steps before death a descriptor
leaves the band it occupied while fed.

    warning ~ 0     no signature. Any display showing distress would have to be
                    coupled by hand, which is the mood-vector-to-decoder architecture
                    ARCHITECTURE.md §M4 rejects. The honesty claim has no substrate.
    warning >> 0    the coupling is already in the physics, and a display grown on the
                    same substrate inherits it for free.

**Mass and gyration departures are near-tautological**: starvation removes cells, so a
shrinking body is not evidence that *form* tracks condition. The load-bearing metric is
`warning_shape`, which uses only compactness and box dimension. It is reported
separately for exactly this reason, and it is the number that decides the verdict.

    python -m dmon.probe --ckpt runs/west.pt --geom west --gif runs/starve.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .substrate import Substrate, descriptors_per_sample, make_sources

KEYS = ("mass", "compactness", "gyration", "box_dim")
SHAPE_KEYS = ("compactness", "box_dim")  # not trivially driven by cell count


@torch.no_grad()
def run_probe(
    sub: Substrate,
    geom: str,
    grid: int,
    reps: int = 8,
    grow: int = 128,
    baseline: int = 64,
    ramp: int = 64,
    hold: int = 128,
    device: str = "cpu",
    frames_out: Path | None = None,
    scale: int = 6,
) -> dict:
    """Grow → hold fed → ramp source to zero → hold starved. Records every step."""
    dev = torch.device(device)
    x, r = sub.seed(reps, grid, dev)
    src = make_sources(geom, reps, grid, dev)

    trace: dict[str, list] = {k: [] for k in KEYS}
    trace["energy"] = []
    trace["supply"] = []
    frames = []

    total = grow + baseline + ramp + hold
    for t in range(total):
        if t < grow + baseline:
            supply = 1.0
        elif t < grow + baseline + ramp:
            supply = 1.0 - (t - grow - baseline + 1) / ramp
        else:
            supply = 0.0

        x, r = sub.step(x, r, src * supply)

        d = descriptors_per_sample(x, sub.cfg)
        for k in KEYS:
            trace[k].append(d[k].cpu())
        alive = (x[:, :1] > sub.cfg.e_death).float()
        e_mean = (x[:, :1] * alive).sum(dim=(1, 2, 3)) / alive.sum(dim=(1, 2, 3)).clamp(min=1)
        trace["energy"].append(e_mean.cpu())
        trace["supply"].append(supply)

        if frames_out is not None and t % 2 == 0:
            from .render import render_state

            frames.append(render_state(x, r, sub, scale))

    if frames_out is not None and frames:
        from .render import to_gif

        to_gif(frames, frames_out)

    stacked = {k: torch.stack(trace[k]) for k in (*KEYS, "energy")}  # (T, reps)
    report = _analyse(stacked, grow, baseline, ramp, hold, reps)
    report["supply"] = trace["supply"]
    report["trace"] = {k: v.tolist() for k, v in stacked.items()}
    report["phases"] = {"grow": grow, "baseline": baseline, "ramp": ramp, "hold": hold}
    report["geom"] = geom
    report["grid"] = grid
    return report


def _analyse(
    stacked: dict[str, torch.Tensor],
    grow: int,
    baseline: int,
    ramp: int,
    hold: int,
    reps: int,
    band_k: float = 3.0,
    persist: int = 3,
    death_frac: float = 0.05,
) -> dict:
    """Per-sample warning time. Each sample is an independent run of the experiment."""
    b0, b1 = grow, grow + baseline  # baseline window
    per_sample = []

    for s in range(reps):
        mu, sig = {}, {}
        for k in KEYS:
            w = stacked[k][b0:b1, s]
            mu[k] = w.mean().item()
            # Floor the band at 1% of the mean. A descriptor that is perfectly flat
            # while fed would otherwise make any subsequent wobble "significant".
            # This widens the band, i.e. makes the test harder to pass — the correct
            # direction for a test whose whole purpose is to be able to fail.
            sig[k] = max(w.std().item(), 0.01 * abs(mu[k]), 1e-9)

        base_mass = mu["mass"]
        death_at = None
        for t in range(b1, stacked["mass"].shape[0]):
            if stacked["mass"][t, s].item() < max(1.0, death_frac * base_mass):
                death_at = t
                break

        depart = {}
        for k in KEYS:
            run = 0
            depart[k] = None
            for t in range(b1, stacked[k].shape[0]):
                out = abs(stacked[k][t, s].item() - mu[k]) > band_k * sig[k]
                run = run + 1 if out else 0
                if run >= persist:
                    depart[k] = t - persist + 1
                    break

        def _warn(keys):
            cand = [depart[k] for k in keys if depart[k] is not None]
            if not cand or death_at is None:
                return None
            return max(0, death_at - min(cand))

        per_sample.append(
            {
                "death_step": death_at,
                "depart": depart,
                "warning_any": _warn(KEYS),
                "warning_shape": _warn(SHAPE_KEYS),
                "baseline_mass": base_mass,
            }
        )

    def _summary(field):
        vals = sorted(v[field] for v in per_sample if v[field] is not None)
        if not vals:
            return {"n": 0, "median": None, "min": None, "max": None}
        n = len(vals)
        return {
            "n": n,
            "median": vals[n // 2],
            "min": vals[0],
            "max": vals[-1],
        }

    survived = sum(1 for v in per_sample if v["death_step"] is None)
    return {
        "per_sample": per_sample,
        "warning_any": _summary("warning_any"),
        "warning_shape": _summary("warning_shape"),
        "survived_ramp": survived,
        "reps": reps,
        "params": {"band_k": band_k, "persist": persist, "death_frac": death_frac},
    }


def verdict(report: dict) -> str:
    """Plain-language read. The probe is allowed to say 'inconclusive'."""
    ws, wa = report["warning_shape"], report["warning_any"]
    if report["survived_ramp"] == report["reps"]:
        return (
            "INCONCLUSIVE — nothing died. The ramp never actually starved it; "
            "lengthen `hold`, or the rule is living off transport reserves."
        )
    if ws["n"] == 0:
        return (
            "FAIL — shape descriptors never left their fed-state band before death. "
            "No visible signature of metabolic state. A display would have to be "
            "coupled by hand; see ARCHITECTURE.md §M4 on why that voids the honesty "
            "claim."
        )
    if ws["median"] is not None and ws["median"] < 5:
        return (
            f"WEAK — shape moves only {ws['median']} steps before death. Flat, then a "
            "cliff. Not enough signature to build a display on."
        )
    return (
        f"PASS — shape leaves its fed-state band {ws['median']} steps before death "
        f"(mass/gyration included: {wa['median']}). The coupling is in the physics; a "
        "display on this substrate inherits it."
    )


def main():
    from .checkpoint import load

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--geom", default=None, help="defaults to the trained geometry")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--grow", type=int, default=128)
    p.add_argument("--baseline", type=int, default=64)
    p.add_argument("--ramp", type=int, default=64)
    p.add_argument("--hold", type=int, default=128)
    p.add_argument("--gif", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    sub, meta = load(a.ckpt, a.device)
    geom = a.geom or meta.get("geom", "west")
    grid = a.grid or meta.get("grid", 64)

    rep = run_probe(
        sub, geom, grid, a.reps, a.grow, a.baseline, a.ramp, a.hold, a.device, a.gif
    )
    rep["trained_geom"] = meta.get("geom")

    print(f"\n=== legibility probe: geom={geom} grid={grid} reps={a.reps} ===")
    print(f"trained under: {meta.get('geom')}  iters={meta.get('iters')}")
    print(f"died: {a.reps - rep['survived_ramp']}/{a.reps}")
    print(f"warning_shape (compactness, box_dim): {rep['warning_shape']}")
    print(f"warning_any   (all descriptors)     : {rep['warning_any']}")
    print(f"\n{verdict(rep)}\n")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
