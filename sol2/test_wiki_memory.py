"""CPU contracts for the provenance-aware L0-C1 wiki memory episode."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .backbone_audit import summarize_binding_rows
from .model import StepTrace
from .test_living_language import LANGUAGE_CFG, build_system
from .wiki_causal_contrast import passage_causal_contrast_loss
from .wiki_distillation import (
    control_delta_energy,
    full_vocab_effect_reverse_kl,
    full_vocab_reverse_kl,
    passage_effect_target_logits,
)
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
    counterfactual_training_pair,
    counterfactual_training_record,
    evaluate_wiki_memory,
    freeze_organism_parameter_group,
    load_wiki_memory_checkpoint,
    organism_gradient_groups,
    organism_optimizer_groups,
    paired_binding_margin_loss,
    passage_visible_teacher_outputs,
    question_only_teacher_outputs,
    require_finite_organism_gradients,
    save_wiki_memory_checkpoint,
)
from .wiki_output_credit import fixed_output_codebook, output_tissue_credit_loss
from .wiki_output_eligibility import eligibility_gated_transport_loss
from .wiki_semantic_credit import (
    fixed_semantic_projection,
    paired_tissue_semantic_credit,
    paired_vector_semantic_credit,
    semantic_target_delta,
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


class RecordingTinyTokenizer(TinyTokenizer):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str, *, return_tensors: str):
        self.texts.append(text)
        return super().__call__(text, return_tensors=return_tensors)


def test_manifest_is_source_family_disjoint_and_complete() -> None:
    corpus = load_wiki_memory_corpus(DATA_PATH)
    assert len(corpus.documents) == 12
    assert len(corpus.split("meta_train")) == 6
    assert len(corpus.split("development")) == 2
    assert len(corpus.split("heldout")) == 4
    assert sum(len(document.questions) for document in corpus.documents) == 24
    families = [document.source_family for document in corpus.documents]
    assert len(families) == len(set(families))


def test_backbone_binding_summary_retains_probability_margin_and_separation() -> None:
    rows = [
        {
            "correct": True,
            "correct_label_probability": 0.75,
            "correct_label_margin": 2.0,
        },
        {
            "correct": False,
            "correct_label_probability": 0.25,
            "correct_label_margin": -1.0,
        },
    ]
    summary = summarize_binding_rows(rows, [0.2, 0.4])
    assert summary == {
        "count": 2.0,
        "accuracy": 0.5,
        "mean_correct_label_probability": 0.5,
        "mean_correct_label_margin": 0.5,
        "mean_pair_logit_separation_rms": 0.30000000000000004,
        "minimum_pair_logit_separation_rms": 0.2,
    }


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


def test_dense_teacher_credit_is_detached_and_passage_visible() -> None:
    student = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    teacher = student.detach().clone().requires_grad_()
    matching = full_vocab_reverse_kl(student, teacher)
    assert abs(float(matching.detach())) < 1e-7
    shifted_teacher = torch.tensor([[2.0, -1.0, -1.0]], requires_grad=True)
    shifted = full_vocab_reverse_kl(student, shifted_teacher)
    assert float(shifted.detach()) > 0.0
    shifted.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0.0
    assert teacher.grad is None and shifted_teacher.grad is None

    student_baseline = torch.tensor([[0.4, -0.2, 0.1]], requires_grad=True)
    common_teacher_prior = torch.tensor([[8.0, -3.0, 1.5]], requires_grad=True)
    teacher_effect = torch.tensor([[0.5, -0.25, 0.0]])
    effect_target = passage_effect_target_logits(
        student_baseline,
        common_teacher_prior + teacher_effect,
        common_teacher_prior,
    )
    assert torch.allclose(
        effect_target,
        student_baseline.detach() + teacher_effect,
        atol=1e-6,
        rtol=0.0,
    )
    different_prior = torch.tensor([[-5.0, 7.0, 2.0]])
    assert torch.allclose(
        effect_target,
        passage_effect_target_logits(
            student_baseline,
            different_prior + teacher_effect,
            different_prior,
        ),
        atol=1e-6,
        rtol=0.0,
    )
    conditioned_student = effect_target.detach().clone().requires_grad_()
    effect_loss = full_vocab_effect_reverse_kl(
        conditioned_student,
        student_baseline,
        common_teacher_prior + teacher_effect,
        common_teacher_prior,
    )
    assert abs(float(effect_loss.detach())) < 1e-7
    mismatched_student = student_baseline.detach().clone().requires_grad_()
    mismatched_loss = full_vocab_effect_reverse_kl(
        mismatched_student,
        student_baseline,
        common_teacher_prior + teacher_effect,
        common_teacher_prior,
    )
    assert float(mismatched_loss.detach()) > 0.0
    mismatched_loss.backward()
    assert mismatched_student.grad is not None
    assert float(mismatched_student.grad.abs().sum()) > 0.0
    assert student_baseline.grad is None and common_teacher_prior.grad is None
    zero_effect_target = passage_effect_target_logits(
        student_baseline,
        common_teacher_prior,
        common_teacher_prior,
    )
    assert torch.equal(zero_effect_target, student_baseline.detach())
    zero_controls = torch.zeros(1, 2, 3)
    active_controls = torch.full((1, 2, 3), 0.5)
    assert float(control_delta_energy(zero_controls)) == 0.0
    assert abs(float(control_delta_energy(active_controls)) - 0.25) < 1e-7

    system, _ = build_system()
    tokenizer = RecordingTinyTokenizer()
    corpus = load_wiki_memory_corpus(DATA_PATH)
    document = corpus.split("meta_train")[0]
    question = document.questions[0]
    left, right = counterfactual_training_pair(
        document, question, epoch=0, compact=True
    )
    left_features, left_teacher = passage_visible_teacher_outputs(
        system,
        tokenizer,
        left[0],
        left[1],
        permutation_seed=19,
        max_memory_tokens=256,
        max_question_tokens=256,
    )
    left_teacher_input = tokenizer.texts[-1]
    right_features, right_teacher = passage_visible_teacher_outputs(
        system,
        tokenizer,
        right[0],
        right[1],
        permutation_seed=19,
        max_memory_tokens=256,
        max_question_tokens=256,
    )
    right_teacher_input = tokenizer.texts[-1]
    baseline_features, baseline_teacher = question_only_teacher_outputs(
        system,
        tokenizer,
        left[1],
        permutation_seed=19,
        max_question_tokens=256,
    )
    baseline_teacher_input = tokenizer.texts[-1]
    assert left_teacher.shape == (system.backbone.vocab_size,)
    assert baseline_teacher.shape == left_teacher.shape
    assert left_features.shape == (1, system.backbone.width)
    assert right_features.shape == left_features.shape
    assert baseline_features.shape == left_features.shape
    assert not any(
        logits.requires_grad
        for logits in (
            left_features,
            right_features,
            baseline_features,
            left_teacher,
            right_teacher,
            baseline_teacher,
        )
    )
    assert left[0].memory in left_teacher_input
    assert right[0].memory in right_teacher_input
    assert left_teacher_input != right_teacher_input
    assert left[0].memory not in baseline_teacher_input
    assert right[0].memory not in baseline_teacher_input
    formatted = format_wiki_question(left[1], permutation_seed=19)
    assert baseline_teacher_input == formatted.prompt


def test_local_semantic_credit_is_differential_and_detached() -> None:
    for field in (
        "memory_semantic_credit_weight",
        "relay_semantic_credit_weight",
        "recall_semantic_credit_weight",
    ):
        try:
            WikiMemoryTrainConfig(**{field: 1.0}).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{field} accepted an unpaired training schedule"
            )
    WikiMemoryTrainConfig(
        paired_counterfactual=True,
        memory_semantic_credit_weight=4.0,
        relay_semantic_credit_weight=4.0,
        recall_semantic_credit_weight=4.0,
    ).validate()

    projection = fixed_semantic_projection(5, 3, seed=307)
    repeated = fixed_semantic_projection(5, 3, seed=307)
    changed = fixed_semantic_projection(5, 3, seed=308)
    assert torch.equal(projection, repeated)
    assert not torch.equal(projection, changed)
    assert projection.requires_grad is False

    left_teacher = torch.tensor(
        [[1.0, -0.5, 0.25, 0.75, -1.0]], requires_grad=True
    )
    right_teacher = torch.tensor(
        [[-0.25, 0.5, -0.75, 0.0, 0.25]], requires_grad=True
    )
    target, raw_target_rms = semantic_target_delta(
        left_teacher,
        right_teacher,
        projection,
        target_rms=0.25,
    )
    assert abs(float(target.pow(2).mean().sqrt()) - 0.25) < 1e-6
    assert float(raw_target_rms) > 0.0
    assert target.requires_grad is False

    left_state = torch.zeros(1, 2, 3, requires_grad=True)
    right_state = torch.zeros(1, 2, 3, requires_grad=True)
    loss, separation, alignment, measured_raw_rms = paired_tissue_semantic_credit(
        left_state,
        right_state,
        left_teacher,
        right_teacher,
        projection,
        target_rms=0.25,
    )
    assert abs(float(loss.detach()) - 0.25**2) < 1e-6
    assert float(separation.detach()) == 0.0
    assert float(alignment.detach()) == 0.0
    assert torch.equal(raw_target_rms, measured_raw_rms)
    loss.backward()
    assert left_state.grad is not None and right_state.grad is not None
    assert float(left_state.grad.abs().sum()) > 0.0
    assert torch.allclose(left_state.grad, -right_state.grad)
    assert left_teacher.grad is None and right_teacher.grad is None

    swapped = paired_tissue_semantic_credit(
        right_state.detach(),
        left_state.detach(),
        right_teacher,
        left_teacher,
        projection,
        target_rms=0.25,
    )[0]
    assert torch.equal(loss.detach(), swapped.detach())

    aligned_left = target[:, None, :].expand(-1, 2, -1) / 2
    aligned_right = -aligned_left
    aligned_loss, _, aligned_cosine, _ = paired_tissue_semantic_credit(
        aligned_left,
        aligned_right,
        left_teacher,
        right_teacher,
        projection,
        target_rms=0.25,
    )
    assert float(aligned_loss) < 1e-12
    assert abs(float(aligned_cosine) - 1.0) < 1e-6


def test_direct_recall_capture_and_semantic_credit_are_exact() -> None:
    system, organ = build_system()
    system.eval()
    torch.manual_seed(991)
    left_exposure = torch.randn(1, 3, system.backbone.width)
    right_exposure = torch.randn(1, 3, system.backbone.width)
    query = torch.randn(1, system.backbone.width)
    left_state, _ = system.observe_feature_sequence(
        left_exposure, system.initial_state(1, "cpu")
    )
    right_state, _ = system.observe_feature_sequence(
        right_exposure, system.initial_state(1, "cpu")
    )

    plain_controls, plain_state, plain_health = system.organism.step(
        query, left_state.clone_detached(), organ_name="language", write_memory=False
    )
    trace = StepTrace()
    traced_controls, traced_state, traced_health = system.organism.step(
        query,
        left_state.clone_detached(),
        organ_name="language",
        write_memory=False,
        trace=trace,
    )
    assert trace.recalled is not None and trace.recall_memory is not None
    expected_query, _ = organ.sense(query)
    expected_recall, _ = organ.recall(
        expected_query,
        left_state.hidden[:, system.organism.memory_idx],
        valid_count=left_state.memory_cursor,
    )
    assert torch.equal(trace.recalled, expected_recall)
    assert torch.equal(plain_controls, traced_controls)
    assert torch.equal(plain_state.hidden, traced_state.hidden)
    assert plain_state.memory_cursor == traced_state.memory_cursor
    assert plain_health is traced_health is None

    left_trace = StepTrace()
    right_trace = StepTrace()
    system.organism.step(
        query,
        left_state,
        organ_name="language",
        write_memory=False,
        trace=left_trace,
    )
    system.organism.step(
        query,
        right_state,
        organ_name="language",
        write_memory=False,
        trace=right_trace,
    )
    assert left_trace.recalled is not None and right_trace.recalled is not None
    assert left_trace.recall_memory is not None and right_trace.recall_memory is not None
    left_trace.recall_memory.retain_grad()
    right_trace.recall_memory.retain_grad()
    projection = fixed_semantic_projection(
        system.backbone.width, LANGUAGE_CFG.hidden, seed=309
    )
    teacher_left = torch.randn(1, system.backbone.width, requires_grad=True)
    teacher_right = torch.randn(1, system.backbone.width, requires_grad=True)
    loss, separation, alignment, raw_rms = paired_vector_semantic_credit(
        left_trace.recalled,
        right_trace.recalled,
        teacher_left,
        teacher_right,
        projection,
        target_rms=0.25,
    )
    assert float(separation.detach()) > 0.0
    assert bool(torch.isfinite(alignment)) and float(raw_rms) > 0.0
    loss.backward()
    for component in ("recall_query", "recall_key", "recall_value", "recall_output"):
        gradient = getattr(organ, component).weight.grad
        assert gradient is not None and float(gradient.abs().sum()) > 0.0
    assert left_trace.recall_memory.grad is not None
    assert right_trace.recall_memory.grad is not None
    assert float(left_trace.recall_memory.grad.abs().sum()) > 0.0
    assert float(right_trace.recall_memory.grad.abs().sum()) > 0.0
    assert teacher_left.grad is None and teacher_right.grad is None
    assert organ.output.decoder.weight.grad is None

    swapped = paired_vector_semantic_credit(
        right_trace.recalled.detach(),
        left_trace.recalled.detach(),
        teacher_right,
        teacher_left,
        projection,
        target_rms=0.25,
    )[0]
    assert torch.allclose(loss.detach(), swapped)


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
        capture_recall=True,
    )
    assert result.state.memory_cursor > baseline.state.memory_cursor
    assert set(result.label_log_probs) == {"A", "B", "C", "D"}
    assert result.output_state is not None
    assert result.output_state.shape == (1, LANGUAGE_CFG.n_output, LANGUAGE_CFG.hidden)
    assert result.relay_state is not None
    assert result.relay_state.shape == (1, LANGUAGE_CFG.n_relay, LANGUAGE_CFG.hidden)
    assert result.exposure_memory_state is not None
    assert result.exposure_memory_state.shape == (
        1,
        LANGUAGE_CFG.n_memory,
        LANGUAGE_CFG.hidden,
    )
    assert result.recalled is not None
    assert result.recalled.shape == (1, LANGUAGE_CFG.hidden)
    assert abs(sum(result.label_log_probs.values())) > 0.0
    result.loss.backward()
    assert organ.output.decoder.weight.grad is not None
    assert float(organ.output.decoder.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in system.backbone.parameters())
    gradient_groups = organism_gradient_groups(system)
    assert {"sensor", "recall", "effector", "connectome", "tissues"}.issubset(
        gradient_groups
    )
    assert gradient_groups["effector"]["rms"] > 0.0
    plasticity = WikiMemoryTrainConfig(
        lr=2e-3,
        control_gain=64.0,
        recall_lr_multiplier=20.0,
        sensor_lr_multiplier=4.0,
        effector_lr_multiplier=0.1,
    )
    plasticity.validate()
    assert plasticity.control_gain == 64.0
    optimizer_groups = organism_optimizer_groups(system, plasticity)
    rates = {group["group_name"]: group["lr"] for group in optimizer_groups}
    assert rates["recall"] == 4e-2
    assert rates["sensor"] == 8e-3
    assert rates["effector"] == 2e-4
    grouped_parameters = [
        parameter for group in optimizer_groups for parameter in group["params"]
    ]
    assert len({id(parameter) for parameter in grouped_parameters}) == len(
        grouped_parameters
    )
    assert sum(parameter.numel() for parameter in grouped_parameters) == sum(
        parameter.numel() for parameter in system.organism.parameters()
    )

    frozen_system, frozen_organ = build_system()
    with torch.no_grad():
        nn.init.normal_(frozen_organ.output.decoder.weight, std=0.02)
    frozen_count = freeze_organism_parameter_group(frozen_system, "effector")
    assert frozen_count > 0
    frozen_result = run_wiki_memory_episode(
        frozen_system,
        tokenizer,
        document,
        question,
        frozen_system.initial_state(1, "cpu"),
        permutation_seed=3,
    )
    frozen_result.loss.backward()
    frozen_groups = organism_gradient_groups(frozen_system)
    assert frozen_groups["effector"]["tensors"] == 0.0
    assert frozen_groups["tissues"]["rms"] > 0.0

    protected_system, _ = build_system()
    protected_parameter = next(protected_system.organism.parameters())
    protected_before = protected_parameter.detach().clone()
    protected_parameter.grad = torch.full_like(protected_parameter, float("nan"))
    try:
        require_finite_organism_gradients(protected_system)
    except FloatingPointError as error:
        assert "non-finite organism gradients" in str(error)
    else:
        raise AssertionError("non-finite organism gradient was accepted")
    assert torch.equal(protected_parameter, protected_before)

    prefix_system, prefix_organ = build_system()
    prefix_result = run_wiki_memory_episode(
        prefix_system,
        tokenizer,
        document,
        question,
        prefix_system.initial_state(1, "cpu"),
        permutation_seed=3,
        control_mode="prefix",
    )
    prefix_result.loss.backward()
    assert prefix_organ.output.decoder.weight.grad is not None
    assert float(prefix_organ.output.decoder.weight.grad.abs().sum()) > 0.0
    assert all(
        parameter.grad is None for parameter in prefix_system.backbone.parameters()
    )

    no_exposure = run_wiki_memory_episode(
        system,
        tokenizer,
        document,
        question,
        initial,
        permutation_seed=3,
        expose=False,
    )
    assert no_exposure.exposure_memory_state is None
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
    assert "Temporary archive binding" in first_document.memory
    assert document.memory not in ("", first_document.memory)
    paired = counterfactual_training_pair(document, question, epoch=0)
    assert paired[0][1].question == paired[1][1].question
    assert paired[0][1].choices == paired[1][1].choices
    assert paired[0][1].answer != paired[1][1].answer
    assert paired[0][0].memory != paired[1][0].memory
    compact = counterfactual_training_pair(
        document, question, epoch=0, compact=True
    )
    assert document.memory not in compact[0][0].memory
    assert len(compact[0][0].memory) < len(paired[0][0].memory)
    assert "Designated answer:" in compact[0][0].memory

    paired_system, paired_organ = build_system()
    with torch.no_grad():
        nn.init.normal_(paired_organ.output.decoder.weight, std=0.02)
    tokenizer = TinyTokenizer()
    start = paired_system.initial_state(1, "cpu")
    branch_results = [
        run_wiki_memory_episode(
            paired_system,
            tokenizer,
            paired_document,
            paired_question,
            start,
            permutation_seed=19,
        )
        for paired_document, paired_question in paired
    ]
    assert all(result.controls is not None for result in branch_results)
    assert all(result.label_logits is not None for result in branch_results)
    separation = (
        branch_results[0].controls - branch_results[1].controls
    ).pow(2).mean().sqrt()
    assert float(separation.detach()) > 0.0

    codebook = fixed_output_codebook(
        LANGUAGE_CFG.hidden, seed=211, device="cpu"
    )
    repeated_codebook = fixed_output_codebook(
        LANGUAGE_CFG.hidden, seed=211, device="cpu"
    )
    assert torch.equal(codebook, repeated_codebook)
    assert torch.allclose(
        codebook @ codebook.T, torch.eye(4), atol=1e-6, rtol=0.0
    )
    left_state = torch.randn(1, LANGUAGE_CFG.n_output, LANGUAGE_CFG.hidden)
    left_state.requires_grad_()
    right_state = left_state.detach().clone().requires_grad_()
    auxiliary_loss, auxiliary_accuracy, output_separation = output_tissue_credit_loss(
        (
            SimpleNamespace(correct_label="A", output_state=left_state),
            SimpleNamespace(correct_label="B", output_state=right_state),
        ),
        codebook=codebook,
        scale=4.0,
    )
    assert float(auxiliary_loss.detach()) > 0.0
    assert 0.0 <= float(auxiliary_accuracy.detach()) <= 1.0
    assert float(output_separation.detach()) == 0.0
    auxiliary_loss.backward()
    assert float(left_state.grad.abs().sum()) > 0.0
    assert float(right_state.grad.abs().sum()) > 0.0
    assert not torch.allclose(left_state.grad, right_state.grad)

    aligned_left = codebook[0].view(1, 1, -1).expand(
        1, LANGUAGE_CFG.n_output, -1
    )
    aligned_right = codebook[1].view(1, 1, -1).expand(
        1, LANGUAGE_CFG.n_output, -1
    )
    aligned_loss, aligned_accuracy, aligned_separation = output_tissue_credit_loss(
        (
            SimpleNamespace(correct_label="A", output_state=aligned_left),
            SimpleNamespace(correct_label="B", output_state=aligned_right),
        ),
        codebook=codebook,
        scale=4.0,
    )
    assert float(aligned_accuracy) == 1.0
    assert float(aligned_loss) < float(auxiliary_loss.detach())
    assert float(aligned_separation) > 0.0

    paired_output = torch.zeros(
        1, LANGUAGE_CFG.n_output, LANGUAGE_CFG.hidden, requires_grad=True
    )
    mate_output = paired_output.detach().clone().requires_grad_()
    active_relay = torch.ones(
        1, LANGUAGE_CFG.n_relay, LANGUAGE_CFG.hidden, requires_grad=True
    )
    mate_relay = torch.zeros(
        1, LANGUAGE_CFG.n_relay, LANGUAGE_CFG.hidden, requires_grad=True
    )
    transport_results = (
        SimpleNamespace(
            correct_label="A",
            output_state=paired_output,
            relay_state=active_relay,
        ),
        SimpleNamespace(
            correct_label="B",
            output_state=mate_output,
            relay_state=mate_relay,
        ),
    )
    (
        transport_loss,
        target_projection,
        relay_separation,
        transport_output_separation,
        transport_ratio,
        eligibility_gate,
    ) = eligibility_gated_transport_loss(
        transport_results,
        codebook=codebook,
        relay_reference=0.005,
        margin=0.005,
        temperature=0.005,
    )
    assert float(transport_loss.detach()) > 0.0
    assert float(target_projection.detach()) == 0.0
    assert float(relay_separation.detach()) == 1.0
    assert float(transport_output_separation.detach()) == 0.0
    assert float(transport_ratio.detach()) == 0.0
    assert float(eligibility_gate.detach()) == 1.0
    swapped = eligibility_gated_transport_loss(
        tuple(reversed(transport_results)),
        codebook=codebook,
        relay_reference=0.005,
        margin=0.005,
        temperature=0.005,
    )
    assert torch.equal(swapped[0], transport_loss)
    assert torch.equal(swapped[1], target_projection)
    transport_loss.backward()
    assert torch.allclose(paired_output.grad, -mate_output.grad)
    assert float(paired_output.grad.abs().sum()) > 0.0
    assert active_relay.grad is None
    assert mate_relay.grad is None

    gated_left = torch.zeros_like(paired_output, requires_grad=True)
    gated_right = torch.zeros_like(mate_output, requires_grad=True)
    zero_relay = torch.zeros_like(active_relay, requires_grad=True)
    gated_loss = eligibility_gated_transport_loss(
        (
            SimpleNamespace(
                correct_label="A",
                output_state=gated_left,
                relay_state=zero_relay,
            ),
            SimpleNamespace(
                correct_label="B",
                output_state=gated_right,
                relay_state=zero_relay,
            ),
        ),
        codebook=codebook,
        relay_reference=0.005,
        margin=0.005,
        temperature=0.005,
    )[0]
    assert float(gated_loss.detach()) == 0.0
    gated_loss.backward()
    assert float(gated_left.grad.abs().sum()) == 0.0
    assert float(gated_right.grad.abs().sum()) == 0.0
    assert zero_relay.grad is None

    left_logits = torch.zeros(4, requires_grad=True)
    right_logits = torch.zeros(4, requires_grad=True)
    binding_loss, preferences = paired_binding_margin_loss(
        SimpleNamespace(correct_label="A", label_logits=left_logits),
        SimpleNamespace(correct_label="B", label_logits=right_logits),
        margin=1.0,
    )
    assert torch.equal(preferences, torch.zeros(2))
    assert float(binding_loss.detach()) == 1.0
    binding_loss.backward()
    assert float(left_logits.grad.abs().sum()) > 0.0
    assert float(right_logits.grad.abs().sum()) > 0.0

    satisfied, _ = paired_binding_margin_loss(
        SimpleNamespace(
            correct_label="A", label_logits=torch.tensor([2.0, 0.0, 0.0, 0.0])
        ),
        SimpleNamespace(
            correct_label="B", label_logits=torch.tensor([0.0, 2.0, 0.0, 0.0])
        ),
        margin=1.0,
    )
    assert float(satisfied) == 0.0

    causal_left = torch.zeros(4, requires_grad=True)
    causal_right = torch.zeros(4, requires_grad=True)
    causal_neutral = torch.zeros(4, requires_grad=True)
    causal_loss, advantages = passage_causal_contrast_loss(
        SimpleNamespace(correct_label="A", label_logits=causal_left),
        SimpleNamespace(correct_label="B", label_logits=causal_right),
        SimpleNamespace(correct_label="A", label_logits=causal_neutral),
        margin=0.1,
    )
    assert torch.equal(advantages, torch.zeros(4))
    assert abs(float(causal_loss.detach()) - 0.1) < 1e-6
    causal_loss.backward()
    assert float(causal_left.grad.abs().sum()) > 0.0
    assert float(causal_right.grad.abs().sum()) > 0.0
    assert causal_neutral.grad is None

    causal_satisfied, causal_advantages = passage_causal_contrast_loss(
        SimpleNamespace(
            correct_label="A", label_logits=torch.tensor([4.0, 0.0, 0.0, 0.0])
        ),
        SimpleNamespace(
            correct_label="B", label_logits=torch.tensor([0.0, 4.0, 0.0, 0.0])
        ),
        SimpleNamespace(
            correct_label="A", label_logits=torch.zeros(4)
        ),
        margin=0.1,
    )
    assert torch.all(causal_advantages > 0.1)
    assert float(causal_satisfied) == 0.0

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
            train_config=WikiMemoryTrainConfig(
                updates=2,
                paired_counterfactual=True,
                output_credit_weight=1.0,
                output_credit_scale=4.0,
                output_credit_seed=211,
                output_eligibility_weight=4.0,
                output_eligibility_margin=0.005,
                output_eligibility_temperature=0.005,
                output_eligibility_relay_reference=0.005,
                language_control_mode="prefix",
                teacher_distillation_weight=1.0,
                teacher_effect_distillation_weight=2.0,
                teacher_distillation_temperature=0.75,
                memory_semantic_credit_weight=4.0,
                relay_semantic_credit_weight=4.0,
                recall_semantic_credit_weight=4.0,
                semantic_credit_target_rms=0.25,
                semantic_credit_seed=307,
                control_energy_weight=0.1,
                reference_centered_controls=True,
            ),
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
        assert payload["train_config"]["output_credit_weight"] == 1.0
        assert payload["train_config"]["output_credit_seed"] == 211
        assert payload["train_config"]["output_eligibility_weight"] == 4.0
        assert payload["train_config"]["output_eligibility_margin"] == 0.005
        assert payload["train_config"]["language_control_mode"] == "prefix"
        assert payload["train_config"]["teacher_distillation_weight"] == 1.0
        assert payload["train_config"]["teacher_effect_distillation_weight"] == 2.0
        assert payload["train_config"]["teacher_distillation_temperature"] == 0.75
        assert payload["train_config"]["memory_semantic_credit_weight"] == 4.0
        assert payload["train_config"]["relay_semantic_credit_weight"] == 4.0
        assert payload["train_config"]["recall_semantic_credit_weight"] == 4.0
        assert payload["train_config"]["semantic_credit_target_rms"] == 0.25
        assert payload["train_config"]["semantic_credit_seed"] == 307
        assert payload["train_config"]["control_energy_weight"] == 0.1
        assert payload["train_config"]["reference_centered_controls"] is True
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
    assert "output_credit_accuracy" not in evaluation["summaries"]
    assert "output_eligibility_loss" not in evaluation["summaries"]

    prefix_evaluation = evaluate_wiki_memory(
        system,
        tokenizer,
        WikiMemoryCorpus(corpus.wiki_root, corpus.split("development")[:1]),
        split="development",
        permutation_seed=33,
        max_memory_tokens=256,
        max_question_tokens=256,
        control_mode="prefix",
        reference_centered_controls=True,
    )
    assert set(prefix_evaluation["summaries"]) == expected
    assert len(prefix_evaluation["rows"]) == 2
    assert prefix_evaluation["summaries"]["no_exposure"]["control_rms"] == 0.0
    assert prefix_evaluation["summaries"]["zero_control"]["control_rms"] == 0.0


def main() -> None:
    tests = [
        test_manifest_is_source_family_disjoint_and_complete,
        test_backbone_binding_summary_retains_probability_margin_and_separation,
        test_source_hash_and_question_permutation_are_exact,
        test_dense_teacher_credit_is_detached_and_passage_visible,
        test_local_semantic_credit_is_differential_and_detached,
        test_direct_recall_capture_and_semantic_credit_are_exact,
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
