"""CPU contracts for the provenance-aware L0-C1 wiki memory episode."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from .test_living_language import LANGUAGE_CFG, build_system
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
from .wiki_memory_train import (
    WikiMemoryTrainConfig,
    counterfactual_training_record,
    evaluate_wiki_memory,
    load_wiki_memory_checkpoint,
    save_wiki_memory_checkpoint,
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


def test_counterfactual_schedule_and_checkpoint_resume_are_exact() -> None:
    corpus = load_wiki_memory_corpus(DATA_PATH)
    document = corpus.split("meta_train")[0]
    question = document.questions[0]
    first_document, first_question = counterfactual_training_record(
        document, question, epoch=0
    )
    second_document, second_question = counterfactual_training_record(
        document, question, epoch=1
    )
    assert first_question.answer != question.answer
    assert second_question.answer != first_question.answer
    assert "Temporary archive erratum" in first_document.memory
    assert document.memory not in ("", first_document.memory)

    system, _ = build_system()
    optimizer = torch.optim.AdamW(system.organism.parameters(), lr=1e-3)
    lane_state = system.initial_state(1, "cpu")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        save_wiki_memory_checkpoint(
            checkpoint,
            system=system,
            optimizer=optimizer,
            lane_state=lane_state,
            organism_config=LANGUAGE_CFG,
            train_config=WikiMemoryTrainConfig(updates=2),
            model_name="toy",
            dtype="float32",
            corpus_sha256="fixture-hash",
            update=1,
            history=[{"update": 1}],
            evaluations=[],
        )
        restored, _ = build_system()
        restored_optimizer = torch.optim.AdamW(
            restored.organism.parameters(), lr=1e-3
        )
        payload, restored_state = load_wiki_memory_checkpoint(
            checkpoint,
            system=restored,
            optimizer=restored_optimizer,
            device="cpu",
            expected_corpus_sha256="fixture-hash",
        )
        assert payload["update"] == 1
        assert torch.equal(restored_state.hidden, lane_state.hidden)
        assert restored_state.memory_cursor == lane_state.memory_cursor
        left = system.organism.state_dict()
        right = restored.organism.state_dict()
        assert left.keys() == right.keys()
        assert all(torch.equal(left[name], right[name]) for name in left)


def test_causal_evaluator_runs_matched_arms() -> None:
    system, _ = build_system()
    tokenizer = TinyTokenizer()
    corpus = load_wiki_memory_corpus(DATA_PATH)
    evaluation = evaluate_wiki_memory(
        system,
        tokenizer,
        WikiMemoryCorpus(corpus.wiki_root, corpus.split("development")[:1]),
        split="development",
        permutation_seed=33,
        max_memory_tokens=256,
        max_question_tokens=256,
    )
    expected = {
        "normal",
        "no_exposure",
        "zero_control",
        "reset_after_exposure",
        "wrong_passage",
        "memory_lesion",
        "internal_lesion",
        "question_paraphrase",
    }
    assert set(evaluation["summaries"]) == expected
    assert len(evaluation["rows"]) == 2


def main() -> None:
    tests = [
        test_manifest_is_source_family_disjoint_and_complete,
        test_source_hash_and_question_permutation_are_exact,
        test_wiki_episode_exposes_scores_and_backpropagates,
        test_counterfactual_schedule_and_checkpoint_resume_are_exact,
        test_causal_evaluator_runs_matched_arms,
    ]
    print("L0-C1 wiki memory CPU contract gate")
    for test in tests:
        print(f"- {test.__name__}", flush=True)
        test()
    print("all L0-C1 wiki memory contracts hold")


if __name__ == "__main__":
    main()
