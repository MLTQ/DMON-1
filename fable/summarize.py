"""Merge arm results under a runs root into a LADDER.md table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _final_bpc(result: dict) -> float:
    ev = result.get("final_eval")
    return ev["normal"]["bits_per_character"] if ev else float("nan")


def summarize(root: Path) -> str:
    rows = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        row = {"arm": arm_dir.name}
        for kind in ("creature", "gru", "bypass"):
            f = arm_dir / f"{kind}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            row[kind] = _final_bpc(r)
            row[f"{kind}_params"] = r["params"]
            row[f"{kind}_skipped"] = r.get("skipped_updates_total", 0)
            if kind != "gru" and r.get("final_eval"):
                ev = r["final_eval"]
                row["reset_d"] = ev.get("reset_delta_bpc")
                row["shuffle_d"] = ev.get("shuffle_delta_bpc")
                row["mirror_d"] = ev.get("mirror_delta_bpc")
            row[f"{kind}_seed"] = r["config"]["seed"]
        g = arm_dir / "growth.json"
        if g.exists():
            gr = json.loads(g.read_text())
            if "probe_final" in gr:
                row["recruit_final"] = gr["probe_final"]["recruitment_delta_bpc"]
            if "probe_post_graft" in gr:
                row["recruit_post_graft"] = gr["probe_post_graft"]["recruitment_delta_bpc"]
        rows.append(row)

    def fmt(v, spec=".4f"):
        if v is None:
            return "—"
        if isinstance(v, float):
            return format(v, spec)
        return str(v)

    lines = ["# fable ladder", "",
             "| Arm | Creature BPC | GRU BPC | Gap | Bypass BPC | ResetΔ | ShufΔ | MirrorΔ | RecruitΔ(end/graft) | Skipped |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    gaps = []
    for row in rows:
        c, g = row.get("creature"), row.get("gru")
        gap = (c - g) if (c is not None and g is not None) else None
        if gap is not None and gap == gap:
            gaps.append(gap)
        recruit = "—"
        if "recruit_final" in row:
            recruit = (f"{fmt(row['recruit_final'], '+.4f')}"
                       f" / {fmt(row.get('recruit_post_graft'), '+.4f')}")
        lines.append(
            f"| {row['arm']} | {fmt(c)} | {fmt(g)} | {fmt(gap, '+.4f')} "
            f"| {fmt(row.get('bypass'))} | {fmt(row.get('reset_d'), '+.3f')} "
            f"| {fmt(row.get('shuffle_d'), '+.3f')} "
            f"| {fmt(row.get('mirror_d'), '+.3f')} | {recruit} "
            f"| {row.get('creature_skipped', 0)} |")
    if gaps:
        lines += ["", f"Mean gap over {len(gaps)} paired arms: "
                      f"**{sum(gaps) / len(gaps):+.4f} BPC**"]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    text = summarize(root)
    out = Path(args.out) if args.out else root / "LADDER.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
