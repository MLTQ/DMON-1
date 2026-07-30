"""F2 report: savings, interference, and compression across arms and seeds.

Reads every arm's analysis.json under a runs root and answers the three
preregistered questions in fable/experiments/f2-adaptability.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path) -> dict:
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "analysis.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def _visits(a: dict) -> list:
    rows = []
    for reg, items in a.get("visits", {}).items():
        for it in items:
            rows.append({**it, "regime": int(reg)})
    return rows


def savings_slope(a: dict) -> tuple[float | None, float | None, float | None]:
    """(first-visit cost, last-visit cost, per-visit slope) pooled over
    regimes and lanes via least squares on cost ~ visit."""
    rows = _visits(a)
    if not rows:
        return None, None, None
    xs = [r["visit"] for r in rows]
    ys = [r["cost"] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
             if denom else None)
    lo = min(xs)
    hi = max(xs)
    first = sum(r["cost"] for r in rows if r["visit"] == lo) / max(
        sum(1 for r in rows if r["visit"] == lo), 1)
    last = sum(r["cost"] for r in rows if r["visit"] == hi) / max(
        sum(1 for r in rows if r["visit"] == hi), 1)
    return first, last, slope


def steady(a: dict) -> float | None:
    rows = _visits(a)
    if not rows:
        return None
    return sum(r["steady_bpc"] for r in rows) / len(rows)


def steady_regime0(a: dict) -> float | None:
    rows = [r for r in _visits(a) if r["regime"] == 0]
    if not rows:
        return None
    return sum(r["steady_bpc"] for r in rows) / len(rows)


def fmt(v, spec="+.4f"):
    return "—" if v is None else format(v, spec)


def report(root: Path) -> str:
    data = load(root)
    seeds = sorted({name.rsplit("_s", 1)[1] for name in data})
    lines = ["# F2 — adaptability under regime cycling", ""]

    lines += ["## Savings (cost of re-adapting to a returning regime)", "",
              "Cost = mean excess BPC over the first 2048 chars of a block "
              "relative to that same block's final quarter. Falling cost "
              "across visits = the learner is retaining something.", "",
              "**A-only arms are the ongoing-learning control**: they walk the "
              "same position blocks with no regime change, so any decline "
              "there is general improvement, not adaptation. Savings are real "
              "only to the extent the cycled slope exceeds the A-only slope.",
              "",
              "| Arm | Seed | Stream | Visit 0 | Visit last | Slope /visit | Visits |",
              "|---|---|---|---:|---:|---:|---:|"]
    agg: dict[tuple[str, bool], list] = {}
    for name in sorted(data):
        a = data[name]
        kind = a["kind"]
        cycled = bool(a.get("cycled"))
        seed = name.rsplit("_s", 1)[1]
        first, last, slope = savings_slope(a)
        if slope is None:
            continue
        agg.setdefault((kind, cycled), []).append((first, last, slope))
        lines.append(f"| {name} | {seed} | {'cycled' if cycled else 'A-only'} "
                     f"| {fmt(first)} | {fmt(last)} "
                     f"| {fmt(slope)} | {len(_visits(a))} |")
    lines.append("")
    means: dict[tuple[str, bool], float] = {}
    for (kind, cycled), rows in sorted(agg.items()):
        n = len(rows)
        mf = sum(r[0] for r in rows) / n
        ml = sum(r[1] for r in rows) / n
        ms = sum(r[2] for r in rows) / n
        means[(kind, cycled)] = ms
        lines.append(f"- **{kind} / {'cycled' if cycled else 'A-only'}** "
                     f"({n} seeds): visit0 {mf:+.4f} → last {ml:+.4f}, "
                     f"mean slope **{ms:+.4f}/visit**")
    lines.append("")
    lines.append("**Regime-attributable savings** (cycled slope − A-only slope; "
                 "more negative = more genuine adaptation):")
    lines.append("")
    for kind in ("creature", "gru"):
        if (kind, True) in means and (kind, False) in means:
            excess = means[(kind, True)] - means[(kind, False)]
            lines.append(f"- **{kind}**: {means[(kind, True)]:+.4f} − "
                         f"{means[(kind, False)]:+.4f} = **{excess:+.4f}/visit**")
    lines.append("")

    lines += ["## Interference (regime-A steady state: cycled vs A-only)", "",
              "Cycled arms see half the A-tokens, so a gap here bundles "
              "interference with reduced A-exposure; the creature-vs-GRU "
              "*difference* in that gap is the comparable quantity.", "",
              "| Kind | Seed | Cycled A steady | A-only steady | Gap |",
              "|---|---|---:|---:|---:|"]
    gaps: dict[str, list] = {}
    for kind in ("creature", "gru"):
        for seed in seeds:
            c = data.get(f"cycled_{kind}_s{seed}")
            a = data.get(f"aonly_{kind}_s{seed}")
            if not c or not a:
                continue
            cs, as_ = steady_regime0(c), steady(a)
            if cs is None or as_ is None:
                continue
            gaps.setdefault(kind, []).append(cs - as_)
            lines.append(f"| {kind} | {seed} | {cs:.4f} | {as_:.4f} "
                         f"| {cs - as_:+.4f} |")
    lines.append("")
    for kind, g in sorted(gaps.items()):
        lines.append(f"- **{kind}** mean interference gap: "
                     f"**{sum(g) / len(g):+.4f} BPC**")
    lines.append("")

    lines += ["## Compression sanity (A-only steady state)", "",
              "| Kind | Seed | A-only steady BPC |", "|---|---|---:|"]
    comp: dict[str, list] = {}
    for kind in ("creature", "gru"):
        for seed in seeds:
            a = data.get(f"aonly_{kind}_s{seed}")
            if not a:
                continue
            s = steady(a)
            if s is None:
                continue
            comp.setdefault(kind, []).append(s)
            lines.append(f"| {kind} | {seed} | {s:.4f} |")
    lines.append("")
    for kind, c in sorted(comp.items()):
        lines.append(f"- **{kind}** mean: **{sum(c) / len(c):.4f} BPC**")
    if "creature" in comp and "gru" in comp:
        d = (sum(comp["creature"]) / len(comp["creature"])
             - sum(comp["gru"]) / len(comp["gru"]))
        lines.append(f"- creature − gru: **{d:+.4f} BPC**")
    lines.append("")

    skipped = {n: a.get("skipped", 0) for n, a in data.items() if a.get("skipped")}
    lines += ["## Health", "",
              f"Arms with skipped (non-finite gradient) updates: "
              f"{skipped if skipped else 'none'}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    text = report(root)
    (Path(args.out) if args.out else root / "REPORT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
