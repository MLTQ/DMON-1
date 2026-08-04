"""Select the S1-P3b genome plasticity rate from uniform pilot branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select(candidates: dict[float, Path]) -> dict:
    """Select best min(A,B), preferring higher B within a two-point tie band."""

    if not candidates:
        raise ValueError("at least one genome-rate candidate is required")
    rows = {}
    for rate, path in sorted(candidates.items()):
        payload = json.loads(path.read_text())
        final = payload["summary"]["final"]
        max_steps = str(payload["protocol"]["max_steps"])
        a = float(final["a_fixed"]["answer_accuracy"])
        b = float(final["b_by_length"][max_steps]["answer_accuracy"])
        rows[rate] = {
            "a_accuracy": a,
            "b_accuracy": b,
            "min_ab_accuracy": min(a, b),
            "path": str(path),
        }
    best_min = max(row["min_ab_accuracy"] for row in rows.values())
    eligible = {
        rate: row
        for rate, row in rows.items()
        if row["min_ab_accuracy"] >= best_min - 0.02
    }
    selected_rate, selected = max(
        eligible.items(),
        key=lambda item: (
            item[1]["b_accuracy"],
            item[1]["min_ab_accuracy"],
            -item[0],
        ),
    )
    return {
        "candidates": {str(rate): row for rate, row in rows.items()},
        "tie_band": 0.02,
        "selected_rate": selected_rate,
        "selected": selected,
    }


def _candidate(value: str) -> tuple[float, Path]:
    rate, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be RATE=METRICS_PATH")
    return float(rate), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select S1-P3b uniform genome rate")
    parser.add_argument(
        "--candidate",
        type=_candidate,
        action="append",
        required=True,
        metavar="RATE=METRICS_PATH",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise ValueError("genome rates must be unique")
    result = select(candidates)
    rendered = json.dumps(result, indent=2)
    if args.out is not None:
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
