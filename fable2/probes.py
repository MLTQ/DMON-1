"""G2a: relevance-gradient transport probes on a trained Broca checkpoint.

G1 established that any passage carried through the organism improves frozen-Qwen
answers (content-nonspecific transport). These probes grade exposures by how much
passage structure they retain, to locate what actually survives the FIFO->depth
route: full prose, domain vocabulary, token soup, or mere token traffic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol2.wiki_memory import load_wiki_memory_corpus

from .backbone import MultiDepthBackbone
from .config import Fable2Config
from .episodes import EpisodeBank, EpisodeItem
from .system import BrocaSystem, build_broca_system
from .train import dtype_from_name

CONDITIONS = (
    "bare",           # organism-free floor
    "none",           # organism runs the question, no exposure
    "own",            # the question's own passage
    "mate",           # the paired counterfactual passage (G1's wrong arm)
    "off_domain",     # a different-family passage from another split
    "shuffled_own",   # own passage tokens, order destroyed, unigrams kept
    "pool_random",    # same-length tokens sampled from the corpus unigram pool
)


def _stable_generator(question_id: str, seed: int) -> torch.Generator:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))


def _exposure_ids(
    condition: str,
    item: EpisodeItem,
    bank: EpisodeBank,
    exposure_token_ids: dict[str, torch.Tensor],
    token_pool: torch.Tensor,
    off_domain_map: dict[str, str],
    seed: int,
) -> torch.Tensor | None:
    if condition in ("bare", "none"):
        return None
    if condition == "own":
        return exposure_token_ids[item.document_id]
    if condition == "mate":
        return exposure_token_ids[bank.mate(item).document_id]
    if condition == "off_domain":
        return exposure_token_ids[off_domain_map[item.document_id]]
    own = exposure_token_ids[item.document_id]
    generator = _stable_generator(item.question_id, seed)
    if condition == "shuffled_own":
        order = torch.randperm(own.shape[1], generator=generator)
        return own[:, order]
    if condition == "pool_random":
        picks = torch.randint(
            len(token_pool), (own.shape[1],), generator=generator
        )
        return token_pool[picks].unsqueeze(0).to(own.device)
    raise ValueError(f"unknown probe condition {condition!r}")


def run_probes(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install transformers to run transport probes") from error

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
    corpus = load_wiki_memory_corpus(Path(args.corpus))
    bank = EpisodeBank(
        corpus,
        tokenizer,
        backbone,
        device,
        permutation_seed=cfg.permutation_seed,
        max_memory_tokens=cfg.max_memory_tokens,
        max_question_tokens=cfg.max_question_tokens,
    )
    system = build_broca_system(backbone, cfg, device=device)
    system.organism.load_state_dict(payload["organism"])
    if tuple(payload["depths"]) != system.depths:
        raise ValueError("checkpoint depths do not match the rebuilt system")
    system.organism.eval()

    from sol2.wiki_memory import encode_text

    exposure_token_ids = {
        document.id: encode_text(
            tokenizer, document.memory, device, max_tokens=cfg.max_memory_tokens
        )
        for document in corpus.documents
    }
    token_pool = torch.cat(
        [ids.flatten() for ids in exposure_token_ids.values()]
    ).cpu()

    by_family = sorted(
        (document for document in corpus.documents if document.split == "meta_train"),
        key=lambda document: document.id,
    )
    off_domain_map = {}
    for document in corpus.documents:
        partner = next(
            (
                candidate
                for candidate in by_family
                if candidate.source_family != document.source_family
            ),
            None,
        )
        if partner is None:
            raise ValueError("corpus has no off-domain partner for some document")
        off_domain_map[document.id] = partner.id

    items = bank.split_items(args.split)
    report: dict = {
        "schema": 1,
        "experiment": "g2a-transport-probes",
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": int(payload["update"]),
        "split": args.split,
        "conditions": {},
        "per_question": {},
    }
    with torch.no_grad():
        for condition in CONDITIONS:
            losses = []
            for item in items:
                if condition == "bare":
                    score = system.score_from_features(
                        item.question_features,
                        item.question_ids,
                        system.initial_state(1, bank.device),
                        bank.labels,
                        item.correct_index,
                        bare=True,
                    )
                else:
                    state = system.initial_state(1, bank.device)
                    ids = _exposure_ids(
                        condition,
                        item,
                        bank,
                        exposure_token_ids,
                        token_pool,
                        off_domain_map,
                        args.seed,
                    )
                    if ids is not None:
                        state = system.observe_feature_sequence(
                            backbone.encode(ids), state
                        )
                    score = system.score_from_features(
                        item.question_features,
                        item.question_ids,
                        state,
                        bank.labels,
                        item.correct_index,
                    )
                losses.append(float(score.loss.detach()))
                report["per_question"].setdefault(item.question_id, {})[condition] = (
                    losses[-1]
                )
            report["conditions"][condition] = {
                "mean_loss": sum(losses) / len(losses),
                "per_question_loss": losses,
            }

    own = report["conditions"]["own"]["mean_loss"]
    report["gradient"] = {
        "prose_vs_shuffled": report["conditions"]["shuffled_own"]["mean_loss"] - own,
        "shuffled_vs_pool": (
            report["conditions"]["pool_random"]["mean_loss"]
            - report["conditions"]["shuffled_own"]["mean_loss"]
        ),
        "pool_vs_none": (
            report["conditions"]["none"]["mean_loss"]
            - report["conditions"]["pool_random"]["mean_loss"]
        ),
        "own_vs_mate": report["conditions"]["mate"]["mean_loss"] - own,
        "own_vs_off_domain": report["conditions"]["off_domain"]["mean_loss"] - own,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for condition in CONDITIONS:
        print(f"{condition}: {report['conditions'][condition]['mean_loss']:.4f}")
    for name, value in report["gradient"].items():
        print(f"{name}: {value:+.4f}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--corpus", default="sol2/experiments/l0c1-wiki-memory-corpus.json"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    run_probes(parse_args())


if __name__ == "__main__":
    main()
