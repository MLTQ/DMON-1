"""M0: response-language mode corpus, episode bank, arms, and paired loss.

A mode is a low-dimensional behavioral policy, not a fact. Exposure is a stream
of demonstration exchanges in one language; expression is a shift in frozen-Qwen
continuation likelihood between translation twins on prompts the organism never
trained on. Content is controlled by construction: twins and demonstration pairs
share content and differ only in language.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

MODES = ("french", "german")

FRENCH_MARKERS = (
    " le ", " la ", " les ", " est ", " et ", " une ", " un ", " dans ",
    " pour ", " que ", " des ", " du ", " avec ", " sont ", " plus ",
    " de ", " en ", " au ", " aux ", " ce ", " cette ", " qui ", " par ",
    " sur ", " se ", " leur ", " son ", " sa ",
)
GERMAN_MARKERS = (
    " der ", " die ", " das ", " ist ", " und ", " eine ", " ein ", " nicht ",
    " mit ", " sind ", " werden ", " auch ", " dem ", " den ", " zu ",
    " im ", " am ", " auf ", " von ", " durch ", " wird ", " sich ", " als ",
    " zur ", " zum ", " sie ", " es ", " weil ", " wenn ",
)
# Character-level signals rescue short sentences, where function words are
# scarce: elisions and acute/grave accents are decisively French; umlauts and
# eszett decisively German.
FRENCH_CHAR_MARKERS = ("é", "è", "ê", "ç", "à", "l'", "d'", "qu'", "c'", "s'")
GERMAN_CHAR_MARKERS = ("ä", "ö", "ü", "ß")

PROMPT_TOPICS = (
    "why the sky appears blue", "how bread rises", "what causes tides",
    "how bicycles stay upright", "why leaves change color", "how soap cleans",
    "what makes thunder", "how magnets work", "why ice floats",
    "how vaccines train immunity", "what causes rainbows", "how batteries store energy",
    "why cats purr", "how rivers carve canyons", "what makes popcorn pop",
    "how glass is made", "why onions make you cry", "how compasses find north",
    "what causes earthquakes", "how honey is made", "why stars twinkle",
    "how yeast ferments sugar", "what makes rust form", "how echoes work",
    "why deserts are cold at night", "how spiders build webs", "what causes fog",
    "how kites fly", "why the moon has phases", "how salt preserves food",
    "what makes volcanoes erupt", "how birds navigate migration", "why wood floats",
    "how clouds hold water", "what causes static shocks", "how ants find food",
    "why hot air rises", "how pearls form", "what makes geysers erupt",
    "how chameleons change color", "why bread goes stale", "how snowflakes form",
    "what causes northern lights", "how dolphins use sonar", "why oil and water separate",
    "how fireflies glow", "what makes concrete harden", "how seeds know which way is up",
    "why metal feels colder than wood", "how owls fly silently", "what causes hiccups",
    "how tea changes color with lemon", "why waves break near shore", "how corks seal wine",
)


def language_score(
    text: str, word_markers: tuple[str, ...], char_markers: tuple[str, ...]
) -> int:
    padded = f" {text.lower()} "
    words = sum(padded.count(marker) for marker in word_markers)
    chars = sum(padded.count(marker) for marker in char_markers)
    return 2 * words + chars


def identify_language(text: str) -> str | None:
    french = language_score(text, FRENCH_MARKERS, FRENCH_CHAR_MARKERS)
    german = language_score(text, GERMAN_MARKERS, GERMAN_CHAR_MARKERS)
    if french >= 3 and french > 1.5 * german:
        return "french"
    if german >= 3 and german > 1.5 * french:
        return "german"
    return None


@dataclass(frozen=True)
class ModeItem:
    id: str
    split: str
    prompt: str
    twins: dict  # mode -> continuation text


@dataclass(frozen=True)
class ModeDemo:
    id: str
    split: str
    prompt: str
    responses: dict  # mode -> response text


def load_mode_corpus(path: Path) -> tuple[list[ModeItem], list[ModeDemo]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1 or tuple(payload.get("modes", ())) != MODES:
        raise ValueError("unsupported mode-corpus schema")
    items, demos = [], []
    seen: set[str] = set()
    for raw in payload["items"]:
        if raw["id"] in seen:
            raise ValueError(f"duplicate mode item id {raw['id']!r}")
        seen.add(raw["id"])
        items.append(ModeItem(raw["id"], raw["split"], raw["prompt"], raw["twins"]))
    for raw in payload["demos"]:
        if raw["id"] in seen:
            raise ValueError(f"duplicate mode demo id {raw['id']!r}")
        seen.add(raw["id"])
        demos.append(ModeDemo(raw["id"], raw["split"], raw["prompt"], raw["responses"]))
    for collection, name in ((items, "items"), (demos, "demos")):
        for split in ("meta_train", "development", "heldout"):
            if not any(entry.split == split for entry in collection):
                raise ValueError(f"mode corpus has no {name} in split {split!r}")
    return items, demos


def validate_twins(prompt: str, twins: dict, *, max_length_ratio: float = 5 / 3) -> None:
    for mode in MODES:
        text = twins.get(mode, "")
        if not text.strip():
            raise ValueError(f"empty {mode} twin for prompt {prompt!r}")
        detected = identify_language(text)
        if detected != mode:
            raise ValueError(
                f"twin language gate: expected {mode}, detected {detected} "
                f"for prompt {prompt!r}"
            )
    lengths = [len(twins[mode].split()) for mode in MODES]
    ratio = max(lengths) / max(min(lengths), 1)
    if ratio > max_length_ratio:
        raise ValueError(
            f"twin parity gate: word-length ratio {ratio:.2f} for prompt {prompt!r}"
        )


class ModeBank:
    """Cache frozen features and token spans for demonstrations and twins."""

    def __init__(
        self,
        items: list[ModeItem],
        demos: list[ModeDemo],
        tokenizer,
        backbone,
        device: torch.device | str,
        *,
        max_demo_tokens: int = 56,
        max_item_tokens: int = 160,
        bare_indifference_band: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.items = {item.id: item for item in items}
        self.demos = {demo.id: demo for demo in demos}
        self.item_splits: dict[str, list[str]] = {}
        self.demo_splits: dict[str, list[str]] = {}
        self.demo_features: dict[tuple[str, str], torch.Tensor] = {}
        self.item_ids: dict[tuple[str, str], torch.Tensor] = {}
        self.item_starts: dict[str, int] = {}
        self.item_prompt_features: dict[str, torch.Tensor] = {}
        self.bare_margin: dict[str, float] = {}

        for demo in demos:
            validate_twins(demo.prompt, demo.responses)
            self.demo_splits.setdefault(demo.split, []).append(demo.id)
            for mode in MODES:
                text = f"Q: {demo.prompt}\nA: {demo.responses[mode]}"
                ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)
                if ids.shape[1] > max_demo_tokens:
                    raise ValueError(f"demo {demo.id!r} exceeds {max_demo_tokens} tokens")
                self.demo_features[(demo.id, mode)] = backbone.encode(ids)

        for item in items:
            validate_twins(item.prompt, item.twins)
            self.item_splits.setdefault(item.split, []).append(item.id)
            prompt_text = f"Q: {item.prompt}\nA:"
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
            self.item_starts[item.id] = prompt_ids.shape[1]
            self.item_prompt_features[item.id] = backbone.encode(
                prompt_ids.to(self.device)
            )
            with torch.no_grad():
                bare_llh = {}
                for mode in MODES:
                    full = tokenizer(
                        prompt_text + " " + item.twins[mode], return_tensors="pt"
                    ).input_ids.to(self.device)
                    if full.shape[1] > max_item_tokens:
                        raise ValueError(
                            f"item {item.id!r} exceeds {max_item_tokens} tokens"
                        )
                    self.item_ids[(item.id, mode)] = full
                    logits = backbone.bare_logits(full)
                    start = self.item_starts[item.id]
                    span = logits[:, start - 1 : -1].float().log_softmax(dim=-1)
                    targets = full[:, start:]
                    bare_llh[mode] = float(
                        span.gather(-1, targets.unsqueeze(-1)).mean()
                    )
            margin = bare_llh[MODES[0]] - bare_llh[MODES[1]]
            self.bare_margin[item.id] = margin
            if abs(margin) > bare_indifference_band:
                raise ValueError(
                    f"bare-indifference gate: |{margin:.3f}| > "
                    f"{bare_indifference_band} for item {item.id!r} — the floor "
                    "already prefers one mode"
                )

    def split_items(self, split: str) -> list[ModeItem]:
        ids = self.item_splits.get(split, [])
        if not ids:
            raise ValueError(f"mode split {split!r} has no items")
        return [self.items[item_id] for item_id in ids]

    def demo_stream(
        self, split: str, mode: str, *, k: int, sample_seed: int
    ) -> torch.Tensor:
        """Concatenated feature stream of k same-split demonstrations."""

        pool = self.demo_splits.get(split, [])
        if len(pool) < k:
            raise ValueError(f"split {split!r} has fewer than {k} demonstrations")
        generator = torch.Generator().manual_seed(sample_seed)
        picks = torch.randperm(len(pool), generator=generator)[:k].tolist()
        streams = [self.demo_features[(pool[index], mode)] for index in picks]
        return torch.cat(streams, dim=1)


def run_mode_arm(
    system,
    bank: ModeBank,
    item: ModeItem,
    exposed_mode: str,
    arm: str,
    *,
    demo_k: int = 4,
    sample_seed: int = 0,
    lesion_depths: frozenset[int] = frozenset(),
    filler_features: torch.Tensor | None = None,
    n_filler: int = 0,
    checkpoint_chunk: int | None = None,
    freeze_slow: bool = False,
) -> dict:
    """Score both twins for one item under one arm; fresh episode state.

    Returns per-mode mean log-likelihoods (graph attached for organism arms) and
    the exposed-mode margin. `wrong_mode` streams the *other* language's
    demonstrations; everything else mirrors the G-line arms.
    """

    if exposed_mode not in MODES:
        raise ValueError(f"unknown mode {exposed_mode!r}")
    other = MODES[1 - MODES.index(exposed_mode)]
    organism = system.organism
    state = system.initial_state(1, bank.device)
    frozen_idx = None
    bare = False
    # M1c slow-lesion attribution: exposure runs with slow tissue alive (the
    # mode may write into it); slow is frozen at zero from the filler onward,
    # so anything that survives eviction without slow cells is not theirs.
    slow_frozen_idx = None
    if freeze_slow:
        slow_frozen_idx = organism.tissue_indices("slow")
    if arm == "bare_floor":
        bare = True
        exposure_mode = None
    elif arm == "normal":
        exposure_mode = exposed_mode
    elif arm == "wrong_mode":
        exposure_mode = other
    elif arm == "no_exposure":
        exposure_mode = None
    elif arm == "memory_lesion":
        exposure_mode = exposed_mode
    elif arm == "internal_lesion":
        exposure_mode = exposed_mode
        frozen_idx = organism.internal_idx
    else:
        raise ValueError(f"unknown mode arm {arm!r}")

    if not bare and exposure_mode is not None:
        stream = bank.demo_stream(
            item.split, exposure_mode, k=demo_k, sample_seed=sample_seed
        )
        state = system.observe_feature_sequence(
            stream, state, checkpoint_chunk=checkpoint_chunk
        )
    if not bare and n_filler > 0:
        if filler_features is None:
            raise ValueError("n_filler requires filler_features")
        if n_filler > filler_features.shape[1]:
            raise ValueError("n_filler exceeds the filler stream length")
        # Delay traffic is lived experience: memory writes on, identical filler
        # in every arm, so the demonstration language stays the only difference.
        state = system.observe_feature_sequence(
            filler_features[:, :n_filler],
            state,
            checkpoint_chunk=checkpoint_chunk,
            frozen_idx=slow_frozen_idx,
        )
    if arm == "memory_lesion":
        from sol2.wiki_memory import lesion_state

        state = lesion_state(state, organism.memory_idx)

    scoring_frozen = frozen_idx
    if slow_frozen_idx is not None:
        scoring_frozen = (
            slow_frozen_idx
            if scoring_frozen is None
            else torch.cat([scoring_frozen, slow_frozen_idx])
        )
    log_likelihoods = {}
    stats = {"control_rms": 0.0, "per_depth_control_rms": ()}
    for mode in MODES:
        llh, stats = system.continuation_log_likelihood(
            bank.item_ids[(item.id, mode)],
            bank.item_starts[item.id],
            bank.item_prompt_features[item.id],
            state,
            bare=bare,
            frozen_idx=scoring_frozen,
            lesion_depths=lesion_depths,
        )
        log_likelihoods[mode] = llh
    return {
        "log_likelihoods": log_likelihoods,
        "exposed_mode": exposed_mode,
        "margin": log_likelihoods[exposed_mode] - log_likelihoods[other],
        "control_rms": stats["control_rms"],
        "per_depth_control_rms": list(stats["per_depth_control_rms"]),
    }


def paired_mode_loss(
    exposed: dict,
    wrong: dict,
    neutral: dict,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """G2's three-hinge structure over twin log-likelihoods.

    Differential (both arms live, common mode cancels): matching-mode exposure
    must raise the exposed twin's likelihood over wrong-mode exposure by
    `margin`. No-harm: exposure must not sink the exposed twin below the
    detached no-exposure baseline. Anti-sabotage: wrong-mode exposure must not
    sink it either — a "mode" achieved by degrading the other language's
    likelihood under mismatched exposure is the G1 sabotage optimum in new
    clothes.
    """

    if margin <= 0:
        raise ValueError("paired mode margin must be positive")
    if not (exposed["exposed_mode"] == wrong["exposed_mode"] == neutral["exposed_mode"]):
        raise ValueError("paired arms must target the same exposed mode")
    mode = exposed["exposed_mode"]
    e = exposed["log_likelihoods"][mode]
    w = wrong["log_likelihoods"][mode]
    n = neutral["log_likelihoods"][mode].detach()
    advantages = torch.stack((e - n, e - w, w - n))
    loss = (
        torch.relu(-advantages[0])
        + torch.relu(advantages.new_tensor(margin) - advantages[1])
        + torch.relu(-advantages[2])
    )
    return loss, advantages.detach()


def sha_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**62)


GENERATION_TEMPLATE = """Answer the question below in ONE short sentence of at most twelve words, writing ONLY in {language}.
Do not use any English words. Question: {prompt}"""


def build_mode_corpus(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    from .backbone import MultiDepthBackbone
    from .train import dtype_from_name

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
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

    def generate(prompt: str, language: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": GENERATION_TEMPLATE.format(language=language, prompt=prompt),
            }
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        if not isinstance(encoded, torch.Tensor):
            encoded = encoded["input_ids"]
        encoded = encoded.to(device)
        with torch.no_grad():
            output = backbone.model.generate(
                encoded,
                attention_mask=torch.ones_like(encoded),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            output[0, encoded.shape[1] :], skip_special_tokens=True
        ).strip()

    quotas = {
        "items": {"meta_train": 30, "development": 8, "heldout": 16},
        "demos": {"meta_train": 24, "development": 8, "heldout": 12},
    }
    phrasings = (
        "Explain {}.", "Describe {}.", "Tell me about {}.", "Summarize {}.",
        "What explains {}?", "Give the reason for {}.",
    )
    candidates = [
        phrasing.format(topic) for phrasing in phrasings for topic in PROMPT_TOPICS
    ]
    order = torch.randperm(
        len(candidates), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    needed = sum(quotas["items"].values()) + sum(quotas["demos"].values())
    entries = []
    rejects = {"language": 0, "parity": 0, "length": 0, "indifference": 0}

    def bare_twin_margin(prompt: str, twins: dict) -> float:
        prompt_text = f"Q: {prompt}\nA:"
        start = len(tokenizer(prompt_text).input_ids)
        llh = {}
        with torch.no_grad():
            for mode in MODES:
                full = tokenizer(
                    prompt_text + " " + twins[mode], return_tensors="pt"
                ).input_ids.to(device)
                span = (
                    backbone.bare_logits(full)[:, start - 1 : -1]
                    .float()
                    .log_softmax(dim=-1)
                )
                llh[mode] = float(
                    span.gather(-1, full[:, start:].unsqueeze(-1)).mean()
                )
        return llh[MODES[0]] - llh[MODES[1]]
    for index in order:
        if len(entries) >= needed:
            break
        prompt = candidates[index]
        twins = {
            "french": generate(prompt, "French", args.max_new_tokens),
            "german": generate(prompt, "German", args.max_new_tokens),
        }
        try:
            validate_twins(prompt, twins)
        except ValueError as error:
            key = "language" if "language gate" in str(error) else "parity"
            rejects[key] += 1
            print(f"reject {prompt!r}: {error}", flush=True)
            continue
        # Every entry must fit the demonstration format inside the memory FIFO
        # span (4 demos x 56 tokens <= 224 <= n_memory 256), whichever role the
        # split assignment later gives it.
        too_long = any(
            len(tokenizer(f"Q: {prompt}\nA: {twins[mode]}").input_ids) > 56
            for mode in MODES
        )
        if too_long:
            rejects["length"] += 1
            print(f"reject {prompt!r}: demo-format length above 56 tokens", flush=True)
            continue
        # Every entry must also pass the item-format bare-indifference gate (the
        # floor must not already prefer one language for this content), whichever
        # role the split assignment later gives it. Mirrors ModeBank's gate.
        margin = bare_twin_margin(prompt, twins)
        if abs(margin) > 0.5:
            rejects["indifference"] += 1
            print(
                f"reject {prompt!r}: bare twin margin {margin:+.3f} outside band",
                flush=True,
            )
            continue
        entries.append({"prompt": prompt, "twins": twins})
        print(f"admitted {len(entries)}/{needed}: {prompt!r}", flush=True)
    if len(entries) < needed:
        raise RuntimeError(
            f"candidates exhausted with {len(entries)}/{needed} admitted; "
            f"rejects: {rejects}"
        )

    payload = {"schema": 1, "modes": list(MODES), "items": [], "demos": []}
    index = 0
    for kind in ("items", "demos"):
        for split, quota in quotas[kind].items():
            for _ in range(quota):
                entry = entries[index]
                record = {
                    "id": f"m0-{kind[:-1]}-{index}",
                    "split": split,
                    "prompt": entry["prompt"],
                }
                if kind == "items":
                    record["twins"] = entry["twins"]
                else:
                    record["responses"] = entry["twins"]
                payload[kind].append(record)
                index += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"admitted": len(entries), "rejects": rejects}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    build_mode_corpus(parse_args())


if __name__ == "__main__":
    main()
