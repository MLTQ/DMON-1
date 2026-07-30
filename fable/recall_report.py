"""F6 report: recall curves by delay, per arm, plus the recruitment probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .recall import CHANCE_BITS, DELAY_BUCKETS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    arms = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "analysis.json"
        if f.exists():
            arms[d.name] = json.loads(f.read_text())

    cols = [f"d{lo}_{hi - 1}" for lo, hi in
            zip(DELAY_BUCKETS[:-1], DELAY_BUCKETS[1:])]
    lines = ["# F6 — cued recall under state pressure", "",
             f"Chance on a recall position = {CHANCE_BITS:.3f} bits. "
             "Frozen-tape (held-out) numbers; ~= chance means the pair did "
             "not survive, ~0 means it did.", "",
             "| Arm | Natural | Recall(all) | " + " | ".join(cols) + " | n |",
             "|---|" + "---:|" * (len(cols) + 3)]

    def f(v):
        return "—" if v is None else f"{v:.3f}"

    for name, a in arms.items():
        ev = a.get("eval", {}).get("normal", {})
        row = [f(ev.get("natural_bpc")), f(ev.get("recall_bpc_all"))]
        row += [f(ev.get(f"recall_bpc_{c}")) for c in cols]
        row.append(str(ev.get("n_recall", 0)))
        lines.append(f"| {name} | " + " | ".join(row) + " |")

    lines += ["", "## Recruitment probe (creature arms: freeze internal tissue)",
              "",
              "| Arm | Freeze Δ on recall | Freeze Δ on natural |",
              "|---|---:|---:|"]
    for name, a in arms.items():
        ev = a.get("eval", {})
        if "freeze_recall_delta" in ev:
            lines.append(f"| {name} | {f(ev['freeze_recall_delta'])} "
                         f"| {f(ev['freeze_natural_delta'])} |")
    lines += ["", "Skipped updates: " + json.dumps(
        {n: a.get("skipped", 0) for n, a in arms.items()}), ""]

    text = "\n".join(lines)
    (Path(args.out) if args.out else root / "REPORT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
