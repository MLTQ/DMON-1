"""M1a: retention curve of an acquired mode under neutral traffic.

Measurement only. The organism acquires a mode from demonstrations, then N
tokens of fixed neutral filler stream through it (memory writes on, exactly as
lived experience would), and the mode differential is measured as a function of
N. Preregistration and reading rules: `experiments/m1a-retention-curve.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .modes import MODES, ModeBank, load_mode_corpus, run_mode_arm, sha_seed

FILLER_LADDER = (0, 32, 64, 128, 192, 256, 384, 512)


def build_filler_stream(bank: ModeBank, tokenizer, backbone, *, max_tokens: int):
    """One fixed English filler stream: meta_train prompts, nested prefixes."""

    prompts = [item.prompt for item in bank.split_items("meta_train")]
    text = " ".join(f"Q: {prompt}" for prompt in prompts)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(bank.device)
    if ids.shape[1] < max_tokens:
        repeats = -(-max_tokens // ids.shape[1])
        ids = ids.repeat(1, repeats)
    ids = ids[:, :max_tokens]
    return backbone.encode(ids)


@torch.no_grad()
def scored_with_filler(
    system,
    bank: ModeBank,
    item,
    exposed_mode: str,
    arm: str,
    filler_features: torch.Tensor,
    n_filler: int,
    *,
    demo_k: int,
    sample_seed: int,
) -> dict:
    """A mode arm with N filler tokens streamed between exposure and scoring."""

    other = MODES[1 - MODES.index(exposed_mode)]
    state = system.initial_state(1, bank.device)
    if arm == "normal":
        exposure_mode = exposed_mode
    elif arm == "wrong_mode":
        exposure_mode = other
    elif arm == "no_exposure":
        exposure_mode = None
    else:
        raise ValueError(f"retention probes use normal/wrong_mode/no_exposure, not {arm!r}")
    if exposure_mode is not None:
        stream = bank.demo_stream(
            item.split, exposure_mode, k=demo_k, sample_seed=sample_seed
        )
        state = system.observe_feature_sequence(stream, state)
    if n_filler > 0:
        state = system.observe_feature_sequence(filler_features[:, :n_filler], state)
    log_likelihoods = {}
    for mode in MODES:
        llh, _ = system.continuation_log_likelihood(
            bank.item_ids[(item.id, mode)],
            bank.item_starts[item.id],
            bank.item_prompt_features[item.id],
            state,
        )
        log_likelihoods[mode] = llh
    return log_likelihoods


def run_retention(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    from .backbone import MultiDepthBackbone
    from .config import Fable2Config
    from .system import build_broca_system
    from .train import dtype_from_name

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != 1:
        raise ValueError("unsupported fable2 checkpoint schema")
    cfg = Fable2Config.from_dict(payload["config"])
    backbone = MultiDepthBackbone.from_pretrained(
        args.model,
        device=device,
        dtype=dtype_from_name(args.dtype),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    items, demos = load_mode_corpus(Path(args.corpus))
    bank = ModeBank(items, demos, tokenizer, backbone, device)
    system = build_broca_system(backbone, cfg, device=device)
    system.organism.load_state_dict(payload["organism"])
    if tuple(payload["depths"]) != system.depths:
        raise ValueError("checkpoint depths do not match the rebuilt system")
    system.organism.eval()
    filler = build_filler_stream(
        bank, tokenizer, backbone, max_tokens=max(FILLER_LADDER)
    )

    report: dict = {
        "schema": 1,
        "experiment": "m1a-retention-curve",
        "checkpoint_update": int(payload["update"]),
        "fifo_slots": cfg.n_memory,
        "ladder": [],
    }
    heldout = bank.split_items("heldout")
    for n_filler in FILLER_LADDER:
        differentials, mode_margins, floor_anchored = [], [], []
        wins = 0
        for item in heldout:
            for mode in MODES:
                seed = sha_seed("m1a", item.id, mode)
                normal = scored_with_filler(
                    system, bank, item, mode, "normal", filler, n_filler,
                    demo_k=args.demo_k, sample_seed=seed,
                )
                wrong = scored_with_filler(
                    system, bank, item, mode, "wrong_mode", filler, n_filler,
                    demo_k=args.demo_k, sample_seed=seed,
                )
                neutral = scored_with_filler(
                    system, bank, item, mode, "no_exposure", filler, n_filler,
                    demo_k=args.demo_k, sample_seed=seed,
                )
                other = MODES[1 - MODES.index(mode)]
                differential = float(normal[mode]) - float(wrong[mode])
                differentials.append(differential)
                wins += differential > 0
                mode_margins.append(float(normal[mode]) - float(normal[other]))
                anchor = float(neutral[mode]) - float(neutral[other])
                floor_anchored.append(mode_margins[-1] - anchor)
        row = {
            "n_filler": n_filler,
            "differential_mean": sum(differentials) / len(differentials),
            "strict_wins": wins,
            "episodes": len(differentials),
            "mode_margin_mean": sum(mode_margins) / len(mode_margins),
            "floor_anchored_margin_mean": sum(floor_anchored) / len(floor_anchored),
        }
        report["ladder"].append(row)
        print(json.dumps(row), flush=True)

    base = report["ladder"][0]["differential_mean"]
    report["n_half"] = next(
        (
            row["n_filler"]
            for row in report["ladder"]
            if row["differential_mean"] < base / 2
        ),
        None,
    )
    majority = report["ladder"][0]["episodes"] // 2 + 1
    report["n_zero"] = next(
        (
            row["n_filler"]
            for row in report["ladder"]
            if row["strict_wins"] < majority
        ),
        None,
    )
    report["residual_at_max"] = report["ladder"][-1]["differential_mean"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(
        f"n_half={report['n_half']} n_zero={report['n_zero']} "
        f"residual@{max(FILLER_LADDER)}={report['residual_at_max']:+.4f}"
    )
    return report


def render_curve(report_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(report_path.read_text())
    ladder = report["ladder"]
    ns = [row["n_filler"] for row in ladder]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        ns, [row["differential_mean"] for row in ladder],
        color="#2a78d6", linewidth=1.8, marker="o", markersize=4,
        label="differential (normal − wrong-mode)",
    )
    axes[0].plot(
        ns, [row["floor_anchored_margin_mean"] for row in ladder],
        color="#1baf7a", linewidth=1.8, marker="o", markersize=4,
        label="floor-anchored mode margin",
    )
    axes[0].axhline(0.0, color="#555555", linewidth=0.8, linestyle=":")
    axes[0].axvline(
        report["fifo_slots"], color="#e34948", linewidth=1.0, linestyle="--"
    )
    axes[0].text(
        report["fifo_slots"] + 6, axes[0].get_ylim()[1] * 0.9,
        "FIFO span", fontsize=8, color="#e34948",
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(
        ns, [row["strict_wins"] for row in ladder],
        color="#eb6834", linewidth=1.8, marker="o", markersize=4,
    )
    episodes = ladder[0]["episodes"]
    axes[1].axhline(episodes / 2 + 0.5, color="#888888", linewidth=0.9, linestyle="--")
    axes[1].axvline(
        report["fifo_slots"], color="#e34948", linewidth=1.0, linestyle="--"
    )
    axes[1].set_ylim(0, episodes + 1)
    for axis, title, ylabel in (
        (axes[0], "mode retention under neutral traffic", "nats/token"),
        (axes[1], f"strict wins (of {episodes})", "episodes"),
    ):
        axis.set_title(title, fontsize=10, loc="left")
        axis.set_xlabel("filler tokens streamed after demonstrations", fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.tick_params(labelsize=8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--corpus")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--demo-k", type=int, default=4)
    parser.add_argument("--out")
    parser.add_argument("--plot", nargs=2, metavar=("REPORT", "PNG"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot:
        render_curve(Path(args.plot[0]), Path(args.plot[1]))
        return
    if not (args.checkpoint and args.corpus and args.out):
        raise SystemExit("--checkpoint, --corpus, and --out are required to run probes")
    run_retention(args)


if __name__ == "__main__":
    main()
