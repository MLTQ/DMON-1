"""CPU contracts for the provenance-aware L0-C1 wiki memory episode."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from .test_living_language import build_system
from .wiki_memory import (
    WikiDocument,
    WikiMemoryCorpus,
    WikiQuestion,
    format_wiki_question,
    label_token_ids,
    load_wiki_memory_corpus,
    run_wiki_memory_episode,
    score_frozen_baseline,
    summarize_results,
    verify_wiki_sources,
)


DATA_PATH = Path(__file__).with_name("experiments") / "l0c1-wiki-memory-corpus.json"


class TinyTokenizer:
    label_ids = {" A": 7, " B": 8, " C": 9, " D": 10}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        if text in self.label_ids:
            return [self.label_ids[text]]
        words = text.split()
        ids = [1 + sum(word.encode()) % 6 for word in words]
        return ([0] if add_special_tokens else []) + ids

    def __call__(self, text: str, *, return_tensors: str):
        assert return_tensors == "pt"
        return SimpleNamespace(
            input_ids=torch.tensor([self.encode(text)], dtype=torch.long)
        )


def test_manifest_is_source_family_disjoint_and_complete() -> None:
    corpus = load_wiki_memory_corpus(DATA_PATH)
    assert len(corpus.documents) == 12
    assert len(corpus.split("meta_train")) == 6
    assert len(corpus.split("development")) == 2
    assert len(corpus.split("heldout")) == 4
    assert sum(len(document.questions) for document in corpus.documents) == 24
    families = [document.source_family for document in corpus.documents]
    assert len(families) == len(set(families))


def test_source_hash_and_question_permutation_are_exact() -> None:
    content = b"provenance fixture\n"
    digest = hashlib.sha256(content).hexdigest()
    question = WikiQuestion(
        id="fixture_question",
        question="Original wording?",
        choices=("one", "two", "three", "four"),
        answer=2,
        paraphrase="Paraphrased wording?",
    )
    document = WikiDocument(
        id="fixture",
        split="heldout",
        source_family="fixture_family",
        path="fixture.md",
        sha256=digest,
        memory="Remember that the fixture answer is three.",
        questions=(question,),
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "fixture.md").write_bytes(content)
        verify_wiki_sources(WikiMemoryCorpus(str(root), (document,)), root)
        (root / "fixture.md").write_bytes(b"changed")
        try:
            verify_wiki_sources(WikiMemoryCorpus(str(root), (document,)), root)
        except ValueError:
            pass
        else:
            raise AssertionError("changed source hash was accepted")

    first = format_wiki_question(question, permutation_seed=8)
    repeated = format_wiki_question(question, permutation_seed=8)
    paraphrased = format_wiki_question(
        question, permutation_seed=8, paraphrase=True
    )
    assert first == repeated
    assert first.order == paraphrased.order
    assert "Original wording?" in first.prompt
    assert "Paraphrased wording?" in paraphrased.prompt
    assert first.correct_label in ("A", "B", "C", "D")


def test_wiki_episode_exposes_scores_and_backpropagates() -> None:
    system, organ = build_system()
    tokenizer = TinyTokenizer()
    corpus = load_wiki_memory_corpus(DATA_PATH)
    document = corpus.split("meta_train")[0]
    question = document.questions[0]
    labels = label_token_ids(tokenizer, "cpu")
    assert labels.tolist() == [7, 8, 9, 10]

    baseline = score_frozen_baseline(
        system, tokenizer, question, permutation_seed=3
    )
    initial = system.initial_state(1, "cpu")
    result = run_wiki_memory_episode(
        system,
        tokenizer,
        document,
        question,
        initial,
        permutation_seed=3,
        collect_health=True,
    )
    assert result.state.memory_cursor > baseline.state.memory_cursor
    assert set(result.label_log_probs) == {"A", "B", "C", "D"}
    assert abs(sum(result.label_log_probs.values())) > 0.0
    result.loss.backward()
    assert organ.output.decoder.weight.grad is not None
    assert float(organ.output.decoder.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in system.backbone.parameters())

    no_exposure = run_wiki_memory_episode(
        system,
        tokenizer,
        document,
        question,
        initial,
        permutation_seed=3,
        expose=False,
    )
    reset = run_wiki_memory_episode(
        system,
        tokenizer,
        document,
        question,
        initial,
        permutation_seed=3,
        reset_after_exposure=True,
    )
    summary = summarize_results([result, no_exposure, reset])
    assert summary["count"] == 3.0
    assert 0.0 <= summary["accuracy"] <= 1.0


def main() -> None:
    tests = [
        test_manifest_is_source_family_disjoint_and_complete,
        test_source_hash_and_question_permutation_are_exact,
        test_wiki_episode_exposes_scores_and_backpropagates,
    ]
    print("L0-C1 wiki memory CPU contract gate")
    for test in tests:
        print(f"- {test.__name__}", flush=True)
        test()
    print("all L0-C1 wiki memory contracts hold")


if __name__ == "__main__":
    main()
