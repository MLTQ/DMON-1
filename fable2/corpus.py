"""G2 corpus builder: behaviorally gated paired wiki questions at scale.

Each candidate note contributes one passage and two frozen-LLM-drafted
four-choice questions. A question is admitted only if the bare backbone *fails*
it closed-book and *passes* it open-book (passage in context) — the information
must be both necessary and extractable, which the C1 admission rule
("the model got it wrong") never checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from sol2.wiki_memory import format_wiki_question, label_token_ids

from .backbone import MultiDepthBackbone
from .train import dtype_from_name

CANDIDATE_SUBDIRS = ("concepts", "entities", "comparisons", "summaries")

GENERATION_INSTRUCTIONS = """You write exam questions about a passage.
Write exactly TWO multiple-choice questions that can be answered ONLY from the
passage below. Each question must have four distinct answer choices with exactly
one correct answer, and a paraphrase (same meaning, different wording).
Questions must be self-contained: never write "according to the passage".
Return ONLY a JSON object, no other text, in this exact shape:
{"questions": [{"question": "...", "choices": ["...", "...", "...", "..."],
"answer": 0, "paraphrase": "..."}, {...}]}

Passage:
"""


def strip_note(text: str) -> str:
    if text.startswith("---"):
        closing = text.find("\n---", 3)
        if closing != -1:
            text = text[closing + 4 :]
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#+ +", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`>|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_passage(
    text: str, tokenizer, *, min_tokens: int, max_tokens: int
) -> str | None:
    prose = strip_note(text)
    if not prose:
        return None
    ids = tokenizer.encode(prose, add_special_tokens=False)
    if len(ids) < min_tokens:
        return None
    if len(ids) <= max_tokens:
        return prose
    truncated = tokenizer.decode(ids[:max_tokens])
    last_stop = truncated.rfind(". ")
    if last_stop > len(truncated) // 2:
        truncated = truncated[: last_stop + 1]
    return truncated.strip()


def draft_questions(backbone, tokenizer, passage: str, *, max_new_tokens: int) -> list[dict]:
    messages = [{"role": "user", "content": GENERATION_INSTRUCTIONS + passage}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        # Qwen3-family thinking mode restates the prompt template inside its
        # reasoning trace and can exhaust the token budget before the answer.
        enable_thinking=False,
    )
    # transformers 5.x returns a BatchEncoding here; 4.x returned a bare tensor.
    if not isinstance(encoded, torch.Tensor):
        encoded = encoded["input_ids"]
    input_ids = encoded.to(next(backbone.model.parameters()).device)
    with torch.no_grad():
        output = backbone.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        output[0, input_ids.shape[1] :], skip_special_tokens=True
    )
    # Anchor on the LAST answer-shaped object: even without thinking mode the
    # model may echo the template (whose {...} placeholder is not valid JSON)
    # before emitting the real answer.
    start, end = completion.rfind('{"questions"'), completion.rfind("}")
    if start == -1 or end <= start:
        print(f"    draft-fail no-json: {completion[:200]!r}", flush=True)
        return []
    try:
        payload = json.loads(completion[start : end + 1])
    except json.JSONDecodeError as error:
        print(
            f"    draft-fail bad-json ({error}): {completion[start : start + 200]!r}",
            flush=True,
        )
        return []
    drafted = []
    for raw in payload.get("questions", []):
        choices = raw.get("choices")
        answer = raw.get("answer")
        question = raw.get("question")
        paraphrase = raw.get("paraphrase")
        if (
            not isinstance(question, str)
            or not isinstance(paraphrase, str)
            or not isinstance(choices, list)
            or len(choices) != 4
            or len({str(c) for c in choices}) != 4
            or not isinstance(answer, int)
            or answer not in range(4)
            or not question.strip()
            or not paraphrase.strip()
        ):
            continue
        drafted.append(
            {
                "question": question.strip(),
                "choices": [str(c).strip() for c in choices],
                "answer": answer,
                "paraphrase": paraphrase.strip(),
            }
        )
    return drafted[:2]


@torch.no_grad()
def bare_answers_correctly(
    backbone,
    tokenizer,
    labels: torch.Tensor,
    question: dict,
    *,
    permutation_seed: int,
    question_id: str,
    passage: str | None,
) -> bool:
    from sol2.wiki_memory import WikiQuestion

    formatted = format_wiki_question(
        WikiQuestion(
            id=question_id,
            question=question["question"],
            choices=tuple(question["choices"]),
            answer=question["answer"],
            paraphrase=question["paraphrase"],
        ),
        permutation_seed=permutation_seed,
    )
    prompt = formatted.prompt if passage is None else f"{passage}\n\n{formatted.prompt}"
    device = next(backbone.model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logits = backbone.bare_logits(input_ids)[:, -1]
    predicted = int(logits.index_select(-1, labels).float().argmax(dim=-1))
    return predicted == formatted.correct_index


def build_corpus(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    from transformers import AutoTokenizer

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
    labels = label_token_ids(tokenizer, device)
    wiki_root = Path(args.wiki_root)

    candidates = sorted(
        path
        for subdir in CANDIDATE_SUBDIRS
        for path in (wiki_root / subdir).rglob("*.md")
        if path.is_file()
    )
    order = torch.randperm(
        len(candidates), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()

    quotas = {
        "meta_train": args.meta_train_questions,
        "development": args.development_questions,
        "heldout": args.heldout_questions,
    }
    admitted: list[dict] = []
    report = {"scanned": 0, "passage_rejects": 0, "draft_rejects": 0, "gate_rejects": 0}
    needed = sum(quotas.values())

    for index in order:
        if sum(len(d["questions"]) for d in admitted) >= needed:
            break
        path = candidates[index]
        report["scanned"] += 1
        relative_name = str(path.relative_to(wiki_root))
        text = path.read_text(errors="replace")
        passage = extract_passage(
            text,
            tokenizer,
            min_tokens=args.min_passage_tokens,
            max_tokens=args.max_passage_tokens,
        )
        if passage is None or "Designated answer" in passage:
            report["passage_rejects"] += 1
            print(f"[{report['scanned']}] {relative_name}: passage-reject", flush=True)
            continue
        drafted = draft_questions(
            backbone, tokenizer, passage, max_new_tokens=args.max_new_tokens
        )
        if len(drafted) != 2:
            report["draft_rejects"] += 1
            print(f"[{report['scanned']}] {relative_name}: draft-reject", flush=True)
            continue
        relative = str(path.relative_to(wiki_root))
        stem = re.sub(r"[^a-z0-9]+", "-", relative[: -len(".md")].lower()).strip("-")
        document_id = f"g2-{stem}"
        # Amendment 2 (recorded in g2-corpus-scaling.md): questions are gated
        # individually — a sibling's failure says nothing about this question's
        # necessity or extractability, and the 2-of-2 rule was discarding half
        # the admissible supply. Quotas therefore count questions, not documents.
        kept_questions = []
        for position, question in enumerate(drafted):
            question_id = f"{document_id}-q{position}"
            closed = bare_answers_correctly(
                backbone,
                tokenizer,
                labels,
                question,
                permutation_seed=args.permutation_seed,
                question_id=question_id,
                passage=None,
            )
            opened = bare_answers_correctly(
                backbone,
                tokenizer,
                labels,
                question,
                permutation_seed=args.permutation_seed,
                question_id=question_id,
                passage=passage,
            )
            verdict = "admit" if (not closed and opened) else "reject"
            print(
                f"    q{position}: closed_book_correct={closed} "
                f"open_book_correct={opened} -> {verdict}",
                flush=True,
            )
            if verdict == "admit":
                kept_questions.append((question_id, question))
        if not kept_questions:
            report["gate_rejects"] += 1
            print(f"[{report['scanned']}] {relative_name}: gate-reject", flush=True)
            continue
        admitted.append(
            {
                "id": document_id,
                "source_family": stem,
                "path": str(path.relative_to(wiki_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "memory": passage,
                "questions": [
                    {"id": question_id, **question}
                    for question_id, question in kept_questions
                ],
            }
        )
        admitted_questions = sum(len(d["questions"]) for d in admitted)
        print(
            f"admitted {admitted_questions}/{needed} questions "
            f"({document_id}, scanned {report['scanned']})",
            flush=True,
        )

    total_questions = sum(len(d["questions"]) for d in admitted)
    if total_questions < needed:
        raise RuntimeError(
            f"only {total_questions} questions admitted of {needed} required; "
            f"rejects: {report}"
        )

    # Greedy split fill by question count (documents never straddle splits; a
    # two-question document may overshoot a quota by one — recorded, accepted).
    split_iter = iter(quotas.items())
    split, quota = next(split_iter)
    filled = 0
    for document in admitted:
        if filled >= quota:
            try:
                split, quota = next(split_iter)
            except StopIteration:
                split, quota = split, quota
            filled = 0
        document["split"] = split
        filled += len(document["questions"])

    corpus = {"schema": 1, "wiki_root": str(wiki_root), "documents": admitted}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus, indent=2))
    report["admitted"] = len(admitted)
    print(json.dumps(report, indent=2))
    return corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-root", default="/home/m/wiki")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--permutation-seed", type=int, default=101)
    parser.add_argument("--meta-train-questions", type=int, default=72)
    parser.add_argument("--development-questions", type=int, default=12)
    parser.add_argument("--heldout-questions", type=int, default=24)
    parser.add_argument("--min-passage-tokens", type=int, default=80)
    parser.add_argument("--max-passage-tokens", type=int, default=220)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    build_corpus(parse_args())


if __name__ == "__main__":
    main()
