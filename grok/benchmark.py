"""S0 benchmark: creature vs matched GRU on one stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainConfig
from .stream import CharacterVocabulary, ensure_shakespeare
from .train import train_creature, train_gru_control


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(
        updates=args.updates,
        device=args.device,
        batch_size=args.batch_size,
        chunk_length=args.chunk_length,
        n_cells=args.n_cells,
        hidden=args.hidden,
        n_dendrites=args.n_dendrites,
        n_input=args.n_input,
        n_output=args.n_output,
        n_mirror=args.n_mirror,
        steps_per_token=args.steps_per_token,
        seed=args.seed,
        data_dir=args.data_dir,
        lr=args.lr,
        readout_mode=args.readout_mode,
        structural_probe_gain=0.05 if args.structural else 0.0,
        log_every=args.log_every if args.log_every else max(10, args.updates // 40),
        eval_every=args.eval_every if args.eval_every else max(50, args.updates // 8),
    )
    if args.no_attention:
        cfg.use_attention = False
    if args.no_reverse_credit:
        cfg.output_error_credit_gain = 0.0
        cfg.reward_gain = 0.0
    if args.no_fast_plasticity:
        cfg.fast_plasticity_gain = 0.0
    if args.no_credit:
        cfg.output_error_credit_gain = 0.0
        cfg.reward_gain = 0.0
        cfg.fast_plasticity_gain = 0.0
    return cfg


def run_benchmark(cfg: TrainConfig, out_dir: Path) -> dict:
    text = ensure_shakespeare(cfg.data_dir).read_text(encoding="utf-8")
    cut = int(len(text) * 0.9)
    train_text, holdout = text[:cut], text[cut:]
    vocab = CharacterVocabulary.from_text(text)
    out_dir.mkdir(parents=True, exist_ok=True)

    creature = train_creature(
        cfg, train_text, vocab, holdout=holdout, out_dir=out_dir / "creature"
    )
    gru = train_gru_control(
        cfg,
        train_text,
        vocab,
        target_params=creature["params"],
        holdout=holdout,
    )
    report = {
        "config": cfg.to_dict(),
        "creature": {
            "params": creature["params"],
            "history": creature["history"],
            "evals": creature["evals"],
        },
        "gru": gru,
        "gap_final_bpc": (
            creature["evals"][-1]["normal"]["bits_per_character"]
            - gru["eval"]["bits_per_character"]
        ),
        "creature_reset_delta": creature["evals"][-1]["reset_delta_bpc"],
        "creature_shuffle_delta": creature["evals"][-1]["shuffle_delta_bpc"],
        "creature_final_bpc": creature["evals"][-1]["normal"]["bits_per_character"],
        "gru_final_bpc": gru["eval"]["bits_per_character"],
    }
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# grok S0 comparison",
        "",
        f"config: cells={cfg.n_cells} h={cfg.hidden} K={cfg.n_dendrites} "
        f"spt={cfg.steps_per_token} credit={cfg.output_error_credit_gain} "
        f"fast={cfg.fast_plasticity_gain} readout={cfg.readout_mode} "
        f"updates={cfg.updates} seed={cfg.seed}",
        "",
        "| Arm | Params | Final held-out BPC | Reset Δ | Shuffle Δ |",
        "|---|---:|---:|---:|---:|",
        (
            f"| creature | {creature['params']:,} | "
            f"{report['creature_final_bpc']:.4f} | "
            f"{report['creature_reset_delta']:+.3f} | "
            f"{report['creature_shuffle_delta']:+.3f} |"
        ),
        (
            f"| gru | {gru['params']:,} | "
            f"{report['gru_final_bpc']:.4f} | — | — |"
        ),
        "",
        f"Gap (creature − gru): **{report['gap_final_bpc']:+.4f} BPC**",
    ]
    (out_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="grok S0 matched-budget benchmark")
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk-length", type=int, default=32)
    p.add_argument("--n-cells", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-dendrites", type=int, default=8)
    p.add_argument("--n-input", type=int, default=8)
    p.add_argument("--n-output", type=int, default=8)
    p.add_argument("--n-mirror", type=int, default=16)
    p.add_argument("--steps-per-token", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--readout-mode", type=str, default="mean", choices=["mean", "concat", "attn"])
    p.add_argument("--out-dir", type=str, default="grok/runs/s0-bench")
    p.add_argument("--structural", action="store_true")
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--no-reverse-credit", action="store_true")
    p.add_argument("--no-fast-plasticity", action="store_true")
    p.add_argument(
        "--no-credit",
        action="store_true",
        help="Disable reward, reverse credit, and fast plasticity",
    )
    p.add_argument("--data-dir", type=str, default="data/tinyshakespeare")
    args = p.parse_args()
    cfg = build_config(args)
    run_benchmark(cfg, Path(args.out_dir))


if __name__ == "__main__":
    main()
