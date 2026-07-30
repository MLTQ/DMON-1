"""Summarize s0-gap ladder comparison.json files into one markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="grok/runs/s0_gap")
    p.add_argument("--out", type=str, default="grok/runs/s0_gap/LADDER.md")
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("**/comparison.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        arm = path.parent.name
        cfg = data.get("config", {})
        rows.append(
            {
                "arm": arm,
                "seed": cfg.get("seed"),
                "cells": cfg.get("n_cells"),
                "hidden": cfg.get("hidden"),
                "credit": cfg.get("output_error_credit_gain"),
                "spt": cfg.get("steps_per_token"),
                "updates": cfg.get("updates"),
                "readout": cfg.get("readout_mode", "mean"),
                "creature": data.get("creature_final_bpc"),
                "gru": data.get("gru_final_bpc"),
                "gap": data.get("gap_final_bpc"),
                "reset": data.get("creature_reset_delta"),
                "shuffle": data.get("creature_shuffle_delta"),
                "params": data.get("creature", {}).get("params"),
            }
        )
    lines = [
        "# S0 gap ladder results",
        "",
        "| Arm | Seed | Cells | H | SPT | U | Readout | Credit | Params | Creature BPC | GRU BPC | Gap | ResetΔ | ShuffleΔ |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['seed']} | {r['cells']} | {r['hidden']} | {r['spt']} | "
            f"{r['updates']} | {r['readout']} | {r['credit']} | {r['params']} | "
            f"{r['creature']:.4f} | {r['gru']:.4f} | {r['gap']:+.4f} | "
            f"{r['reset']:+.3f} | {r['shuffle']:+.3f} |"
        )
    if not rows:
        lines.append("| _(no comparison.json found)_ | | | | | | | | | | | |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
