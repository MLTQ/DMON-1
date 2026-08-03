"""Manifest and optional executor for the SOL2 S0-R factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Sol2Config
from .train import run


def factorial(seed: int, updates: int) -> list[tuple[str, str, Sol2Config]]:
    rows: list[tuple[str, str, Sol2Config]] = []
    for identity in (False, True):
        identity_name = "identity" if identity else "anonymous"
        for bounded in (False, True):
            bound_name = "bounded" if bounded else "unbounded"
            name = f"creature-{identity_name}-{bound_name}-s{seed}"
            rows.append(
                (
                    name,
                    "creature",
                    Sol2Config(
                        seed=seed,
                        updates=updates,
                        cell_identity=identity,
                        bounded_operators=bounded,
                    ),
                )
            )
        matched_cfg = Sol2Config(seed=seed, updates=updates, cell_identity=identity)
        rows.append((f"gru-{identity_name}-s{seed}", "gru", matched_cfg))
        rows.append(
            (f"transformer-{identity_name}-s{seed}", "transformer", matched_cfg)
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run the SOL2 S0-R factorial")
    parser.add_argument("--root", type=Path, default=Path("sol2/runs/s0r"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=24_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--creature-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for seed in args.seeds:
        for name, kind, cfg in factorial(seed, args.updates):
            if args.creature_only and kind != "creature":
                continue
            out_dir = args.root / name
            manifest.append(
                {
                    "name": name,
                    "kind": kind,
                    "out_dir": str(out_dir),
                    "config": cfg.to_dict(),
                }
            )
            if args.execute:
                checkpoint = out_dir / "latest.pt"
                run(
                    kind,
                    cfg,
                    out_dir,
                    args.device,
                    resume=checkpoint if checkpoint.exists() else None,
                )
    (args.root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} arms to {args.root / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
