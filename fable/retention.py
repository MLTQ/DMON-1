"""Zero-shot retention across regime revisits: the A→B→A test.

F2's savings metric measures how fast a learner *re-adapts* when a regime
returns. That is the transient. It does not measure how much capability
survived the absence — which is the "without losing capability" half of
adaptability, and the standard continual-learning question.

This module measures it from the per-token records F2 already saved: the loss
in the first `head` characters after a regime boundary, per visit index. At
16 characters the learner has taken at most one optimizer step inside the new
block, so this is retention, not recovery.

Two readings:
  level — retention loss at a given visit. Lower = more survived.
  slope — does retention improve across visits? A learner that consolidates
          should forget less each time it revisits a regime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate import LN2


def retention(raw_path: Path, head: int = 16) -> dict:
    raw = torch.load(raw_path, weights_only=True)
    pos, parity, loss = raw["pos"], raw["parity"], raw["loss"]
    block = raw["block"]
    half = pos.shape[1] // 2
    pos, parity, bpc = pos[:, half:], parity[:, half:], loss[:, half:] / LN2

    rows = []
    for lane in range(pos.shape[0]):
        p, par, b = pos[lane], parity[lane], bpc[lane]
        starts = [0] + (torch.where(p[1:] < p[:-1])[0] + 1).tolist() + [len(p)]
        seen = {0: 0, 1: 0}
        for s, e in zip(starts[:-1], starts[1:]):
            seg_pos, seg, reg = p[s:e], b[s:e], int(par[s])
            if int(seg_pos[0]) > 0 or int(seg_pos[-1]) < block - 1:
                continue  # partial block at the scoring boundary
            m = seg_pos < head
            if int(m.sum()):
                rows.append({"lane": lane, "regime": reg,
                             "visit": seen[reg],
                             "retention_bpc": float(seg[m].mean())})
            seen[reg] += 1

    out = {"kind": raw["kind"], "cycled": raw["cycled"], "head": head,
           "n_samples": len(rows)}
    if not rows:
        return out
    xs = [r["visit"] for r in rows]
    ys = [r["retention_bpc"] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    out["slope_per_visit"] = (
        sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None)
    lo, hi = min(xs), max(xs)
    for label, v in (("first", lo), ("last", hi)):
        sel = [r["retention_bpc"] for r in rows if r["visit"] == v]
        out[f"{label}_visit_retention_bpc"] = sum(sel) / len(sel)
    out["mean_retention_bpc"] = my
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot retention across revisits")
    ap.add_argument("--root", required=True)
    ap.add_argument("--head", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    results = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        raw = d / "raw.pt"
        if raw.exists():
            results[d.name] = retention(raw, args.head)

    lines = [f"# Zero-shot retention (first {args.head} chars after a regime boundary)",
             "",
             "At this window the learner has taken at most one optimizer step "
             "inside the new block, so this measures what survived the other "
             "regime — not how fast it recovers (that is F2's savings).", "",
             "| Arm | Stream | Mean retention BPC | First visit | Last visit | Slope/visit |",
             "|---|---|---:|---:|---:|---:|"]

    def f(v, s=".4f"):
        return "—" if v is None else format(v, s)

    agg: dict[tuple[str, bool], list] = {}
    for name, r in results.items():
        if not r.get("n_samples"):
            continue
        agg.setdefault((r["kind"], r["cycled"]), []).append(r)
        lines.append(
            f"| {name} | {'cycled' if r['cycled'] else 'A-only'} "
            f"| {f(r['mean_retention_bpc'])} "
            f"| {f(r['first_visit_retention_bpc'])} "
            f"| {f(r['last_visit_retention_bpc'])} "
            f"| {f(r.get('slope_per_visit'), '+.4f')} |")
    lines.append("")
    means = {}
    for (kind, cycled), rs in sorted(agg.items()):
        n = len(rs)
        mm = sum(r["mean_retention_bpc"] for r in rs) / n
        ms = sum(r["slope_per_visit"] for r in rs) / n
        mf = sum(r["first_visit_retention_bpc"] for r in rs) / n
        ml = sum(r["last_visit_retention_bpc"] for r in rs) / n
        means[(kind, cycled)] = (mm, ms, mf, ml)
        lines.append(f"- **{kind} / {'cycled' if cycled else 'A-only'}** "
                     f"({n} seeds): mean {mm:.4f}, first {mf:.4f} → "
                     f"last {ml:.4f}, slope **{ms:+.4f}/visit**")
    lines.append("")
    lines.append("**Retention penalty** (cycled − A-only mean: the cost of the "
                 "other regime having happened):")
    lines.append("")
    for kind in ("creature", "gru"):
        if (kind, True) in means and (kind, False) in means:
            pen = means[(kind, True)][0] - means[(kind, False)][0]
            lines.append(f"- **{kind}**: {means[(kind, True)][0]:.4f} − "
                         f"{means[(kind, False)][0]:.4f} = **{pen:+.4f} BPC**")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    if args.out:
        Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
