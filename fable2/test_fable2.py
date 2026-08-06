"""CPU contract gate for the fable2 Broca graft. Runs without pytest or downloads."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from sol2.wiki_memory import LABELS, WikiDocument, WikiMemoryCorpus, WikiQuestion

from .backbone import ToyMultiDepthBackbone, inject_control_residual, resolve_depths
from .config import Fable2Config
from .episodes import EpisodeBank, paired_contrast_loss, run_arm, run_depth_lesion
from .system import build_broca_system
from .train import load_checkpoint, save_checkpoint, trainable_parameter_groups


class TinyTokenizer:
    """Deterministic whitespace tokenizer satisfying the wiki-memory surface."""

    def __init__(self, vocab_size: int = 97) -> None:
        self.vocab_size = vocab_size

    def _word_id(self, word: str) -> int:
        digest = hashlib.sha256(word.encode()).digest()
        return 2 + int.from_bytes(digest[:4], "big") % (self.vocab_size - 2)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._word_id(word) for word in text.split()]

    def __call__(self, text: str, return_tensors: str = "pt") -> SimpleNamespace:
        ids = self.encode(text)
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def small_config() -> Fable2Config:
    return Fable2Config().scaled(
        n_input=4,
        n_memory=24,
        n_compute=24,
        n_relay=8,
        n_output=8,
        hidden=32,
        n_dendrites=6,
        initial_active_dendrites=4,
        steps_per_token=2,
        cell_adapter_rank=2,
        depth_fractions=(0.5, 1.0),
        n_control_tokens=2,
        control_rank=4,
        trunk_width=32,
        max_memory_tokens=24,
        max_question_tokens=64,
        eval_every=10,
        checkpoint_every=10,
    )


def toy_corpus() -> WikiMemoryCorpus:
    specs = [
        ("meta_train", "fam-a", "The turquoise archive stores maps of the sunken canal city", "Where are maps of the sunken canal city stored", ("Turquoise archive", "Iron vault", "Sky library", "Salt mine")),
        ("meta_train", "fam-b", "The copper bell rings only when the northern ferry departs", "What event makes the copper bell ring", ("Northern ferry departing", "Sunrise over the bay", "Market opening", "Rainfall starting")),
        ("meta_train", "fam-c", "Nine glass turbines power the floating observatory at night", "What powers the floating observatory at night", ("Nine glass turbines", "A coal furnace", "Tidal chains", "Solar mirrors")),
        ("development", "fam-d", "The archivist feeds the lighthouse lamp with pressed juniper oil", "What fuels the lighthouse lamp", ("Pressed juniper oil", "Whale tallow", "Pine resin", "Refined peat")),
        ("development", "fam-e", "A striped kite marks the only safe path across the mudflats", "What marks the safe path across the mudflats", ("A striped kite", "White stones", "Rope fences", "Carved posts")),
        ("development", "fam-f", "The night train carries mail between the twin river towns", "What does the night train carry between the twin towns", ("Mail", "Timber", "Livestock", "Coal")),
        ("heldout", "fam-g", "Blue lanterns warn divers away from the reef nursery", "What warns divers away from the reef nursery", ("Blue lanterns", "Bell buoys", "Flag ropes", "Painted rocks")),
        ("heldout", "fam-h", "The miller trades flour for kelp twine every third market day", "What does the miller trade flour for", ("Kelp twine", "River clay", "Copper nails", "Dried figs")),
        ("heldout", "fam-i", "A clockwork heron guards the seed vault door", "What guards the seed vault door", ("A clockwork heron", "A stone lion", "Twin wolfhounds", "An iron golem")),
    ]
    documents = []
    for index, (split, family, memory, question, choices) in enumerate(specs):
        documents.append(
            WikiDocument(
                id=f"doc-{index}",
                split=split,
                source_family=family,
                path=f"toy/{index}.md",
                sha256="0" * 64,
                memory=memory,
                questions=(
                    WikiQuestion(
                        id=f"q-{index}",
                        question=question,
                        choices=choices,
                        answer=0,
                        paraphrase=question,
                    ),
                ),
            )
        )
    return WikiMemoryCorpus(wiki_root="/nonexistent", documents=tuple(documents))


def build_toy_bank(cfg: Fable2Config, backbone: ToyMultiDepthBackbone) -> EpisodeBank:
    tokenizer = TinyTokenizer(vocab_size=backbone.vocab_size)
    corpus = toy_corpus()
    last_error = None
    for permutation_seed in range(101, 131):
        try:
            return EpisodeBank(
                corpus,
                tokenizer,
                backbone,
                "cpu",
                permutation_seed=permutation_seed,
                max_memory_tokens=cfg.max_memory_tokens,
                max_question_tokens=cfg.max_question_tokens,
            )
        except ValueError as error:
            last_error = error
    raise AssertionError(f"no permutation seed yields usable mates: {last_error}")


def test_config_contracts() -> None:
    cfg = small_config()
    assert Fable2Config.from_json(cfg.to_json()) == cfg
    try:
        cfg.scaled(max_memory_tokens=999)
        raise AssertionError("memory-span guard did not fire")
    except ValueError:
        pass
    try:
        cfg.scaled(depth_fractions=(0.5, 0.5))
        raise AssertionError("duplicate depth guard did not fire")
    except ValueError:
        pass


def test_resolve_depths() -> None:
    assert resolve_depths((0.25, 0.5, 0.75, 1.0), 24) == (5, 11, 17, 23)
    assert resolve_depths((1.0,), 1) == (0,)
    try:
        resolve_depths((0.5, 0.51), 4)
        raise AssertionError("depth collision guard did not fire")
    except ValueError:
        pass


def test_injection_bounds_and_floor_identity() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 16)
    controls = 0.7 * torch.tanh(torch.randn(2, 3, 16))
    injected = inject_control_residual(hidden, controls)
    residual = injected - hidden
    assert float(residual.abs().max()) <= 0.7 + 1e-6, "residual exceeded control bound"

    zero_eval = torch.zeros(2, 3, 16)
    assert torch.equal(inject_control_residual(hidden, zero_eval), hidden)
    zero_live = torch.zeros(2, 3, 16, requires_grad=True)
    out = inject_control_residual(hidden, zero_live)
    assert torch.equal(out.detach(), hidden), "live zero controls must be a no-op"
    out.sum().backward()
    assert zero_live.grad is not None and bool(torch.isfinite(zero_live.grad).all()), (
        "zero controls must still receive gradient (recruitment path)"
    )


def test_zero_init_graft_is_silent_and_arms_behave() -> None:
    cfg = small_config()
    backbone = ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3)
    bank = build_toy_bank(cfg, backbone)
    system = build_broca_system(backbone, cfg, device="cpu")
    item = bank.split_items("development")[0]

    bare = backbone.bare_logits(item.question_ids)[:, -1]
    state = system.initial_state(1, bank.device)
    state = system.observe_feature_sequence(item.exposure_features, state)
    controls, _ = system._evolve(item.question_features, state, write_memory=False)
    injected = backbone.injected_logits(
        item.question_ids, system.depth_controls(controls)
    )[:, -1]
    assert torch.equal(injected.detach(), bare), "fresh graft must be exactly silent"

    floor = run_arm(system, bank, item, "bare_floor")
    zero_control = system.score_from_features(
        item.question_features,
        item.question_ids,
        system.initial_state(1, bank.device),
        bank.labels,
        item.correct_index,
        inject=False,
    )
    assert torch.equal(zero_control.label_logits, floor.label_logits), (
        "zero-control must be bitwise scaffold-free, identical to the bare floor"
    )
    assert floor.label_logits.dtype == torch.float32
    assert zero_control.label_logits.dtype == torch.float32


def test_output_tissue_read_boundary() -> None:
    cfg = small_config()
    backbone = ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3)
    build_toy_bank(cfg, backbone)
    system = build_broca_system(backbone, cfg, device="cpu")
    organism = system.organism
    torch.manual_seed(11)
    hidden_a = torch.randn(1, organism.n_cells, organism.cfg.hidden)
    hidden_b = torch.randn(1, organism.n_cells, organism.cfg.hidden)
    hidden_b[:, organism.output_idx] = hidden_a[:, organism.output_idx]
    read_a, _ = organism._read_output(hidden_a, system.organ_name, False)
    read_b, _ = organism._read_output(hidden_b, system.organ_name, False)
    assert torch.equal(read_a, read_b), "effector must read only output tissue"


def randomize_heads(system) -> None:
    organ = system.organism.attached_organs[system.organ_name]
    with torch.no_grad():
        for head in organ.depth_heads:
            head.weight.normal_(std=0.2)
            head.bias.normal_(std=0.05)


def test_arm_distinctness_after_recruitment() -> None:
    cfg = small_config()
    backbone = ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3)
    bank = build_toy_bank(cfg, backbone)
    system = build_broca_system(backbone, cfg, device="cpu")
    torch.manual_seed(23)
    randomize_heads(system)
    item = bank.split_items("development")[0]
    with torch.no_grad():
        scores = {
            arm: run_arm(system, bank, item, arm)
            for arm in ("normal", "wrong_passage", "no_exposure", "memory_lesion", "internal_lesion")
        }
        lesioned = run_depth_lesion(system, bank, item, system.depths[0])
    names = list(scores)
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            a, b = scores[names[first]], scores[names[second]]
            assert not torch.equal(a.label_logits, b.label_logits), (
                f"arms {names[first]} and {names[second]} are the same computation; "
                "an arm that cannot differ is not a control"
            )
    assert not torch.equal(lesioned.label_logits, scores["normal"].label_logits), (
        "depth lesion must alter behavior once heads are live"
    )


def test_gradient_recruitment_and_frozen_backbone() -> None:
    cfg = small_config()
    backbone = ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3)
    bank = build_toy_bank(cfg, backbone)
    system = build_broca_system(backbone, cfg, device="cpu")
    optimizer = torch.optim.AdamW(trainable_parameter_groups(system, cfg))
    item = bank.split_items("meta_train")[0]
    organ = system.organism.attached_organs[system.organ_name]

    def one_backward() -> None:
        with torch.no_grad():
            neutral = run_arm(system, bank, item, "no_exposure")
        wrong = run_arm(system, bank, item, "wrong_passage")
        exposed = run_arm(system, bank, item, "normal")
        loss, _ = paired_contrast_loss(
            exposed, wrong, neutral, margin=cfg.contrast_margin
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

    one_backward()
    for index, head in enumerate(organ.depth_heads):
        assert head.weight.grad is not None and float(head.weight.grad.abs().sum()) > 0, (
            f"depth head {index} received no gradient at init"
        )
    assert all(p.grad is None for p in backbone.parameters()), "backbone must stay frozen"
    optimizer.step()

    one_backward()
    live = {
        "trunk": sum(float(p.grad.abs().sum()) for p in organ.trunk.parameters() if p.grad is not None),
        "sensor": float(organ.sensor.weight.grad.abs().sum()),
        "tissues": sum(
            float(p.grad.abs().sum())
            for n, p in system.organism.named_parameters()
            if n.startswith("tissues.") and p.grad is not None
        ),
        "graph": sum(
            float(p.grad.abs().sum())
            for n, p in system.organism.named_parameters()
            if n.startswith("graph.") and p.grad is not None
        ),
    }
    for name, value in live.items():
        assert value > 0, f"{name} not recruited after one optimizer step"
    assert all(p.grad is None for p in backbone.parameters()), "backbone must stay frozen"


def test_paired_loss_arithmetic_and_detachment() -> None:
    def fake(logits: list[float], requires_grad: bool):
        tensor = torch.tensor(logits, dtype=torch.float32, requires_grad=requires_grad)
        return SimpleNamespace(correct_label=LABELS[0], label_logits=tensor)

    exposed = fake([2.0, 0.0, 0.0, 0.0], True)
    wrong = fake([0.0, 1.0, 0.0, 0.0], False)
    neutral = fake([0.0, 0.0, 1.0, 0.0], False)
    loss, advantages = paired_contrast_loss(exposed, wrong, neutral, margin=0.1)
    assert advantages.shape == (2,)
    assert float(advantages.min()) > 0
    assert float(loss.detach()) == 0.0, "satisfied margins must produce zero loss"
    exposed_live = fake([0.0, 2.0, 0.0, 0.0], True)
    loss, _ = paired_contrast_loss(exposed_live, wrong, neutral, margin=0.1)
    loss.backward()
    assert exposed_live.label_logits.grad is not None
    try:
        paired_contrast_loss(
            exposed, fake([0.0] * 4, False), SimpleNamespace(correct_label=LABELS[1], label_logits=torch.zeros(4)), margin=0.1
        )
        raise AssertionError("same-question guard did not fire")
    except ValueError:
        pass


def test_leakage_lint_and_guards() -> None:
    cfg = small_config()
    backbone = ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3)
    tokenizer = TinyTokenizer(vocab_size=backbone.vocab_size)
    document = toy_corpus().documents[0]
    poisoned = WikiDocument(
        id=document.id,
        split=document.split,
        source_family=document.source_family,
        path=document.path,
        sha256=document.sha256,
        memory=document.memory + " Designated answer: Turquoise archive",
        questions=document.questions,
    )
    corpus = WikiMemoryCorpus(wiki_root="/nonexistent", documents=(poisoned,))
    try:
        EpisodeBank(
            corpus,
            tokenizer,
            backbone,
            "cpu",
            permutation_seed=101,
            max_memory_tokens=cfg.max_memory_tokens,
            max_question_tokens=cfg.max_question_tokens,
        )
        raise AssertionError("answer-card leakage lint did not fire")
    except ValueError as error:
        assert "Designated answer" in str(error)

    bank = build_toy_bank(cfg, backbone)
    try:
        bank.split_items("nonexistent")
        raise AssertionError("empty-split guard did not fire")
    except ValueError:
        pass


def test_mate_constraints() -> None:
    cfg = small_config()
    backbone = ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3)
    bank = build_toy_bank(cfg, backbone)
    for split in ("meta_train", "development", "heldout"):
        for item in bank.split_items(split):
            mate = bank.mate(item)
            assert mate.split == item.split
            assert mate.source_family != item.source_family
            assert mate.correct_index != item.correct_index


def test_checkpoint_roundtrip() -> None:
    cfg = small_config()
    backbone = ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3)
    bank = build_toy_bank(cfg, backbone)
    system = build_broca_system(backbone, cfg, device="cpu")
    optimizer = torch.optim.AdamW(trainable_parameter_groups(system, cfg))
    randomize_heads(system)
    # Eval mode throughout: training-mode forwards advance EffectiveLinear's
    # power-iteration buffer, which would make scores depend on call order.
    system.organism.eval()
    item = bank.split_items("meta_train")[0]
    with torch.no_grad():
        reference = run_arm(system, bank, item, "normal")
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "broca.pt"
        save_checkpoint(path, system, optimizer, cfg, update=5)
        clone_system = build_broca_system(
            ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3),
            cfg,
            device="cpu",
        )
        clone_system.organism.eval()
        clone_optimizer = torch.optim.AdamW(
            trainable_parameter_groups(clone_system, cfg)
        )
        update = load_checkpoint(path, clone_system, clone_optimizer)
    assert update == 5, f"resumed update counter was {update}"
    for (name_a, a), (name_b, b) in zip(
        system.organism.state_dict().items(), clone_system.organism.state_dict().items()
    ):
        assert name_a == name_b and torch.equal(a, b), f"state mismatch at {name_a}"
    with torch.no_grad():
        resumed = run_arm(clone_system, bank, item, "normal")
    assert torch.equal(reference.label_logits, resumed.label_logits)


def test_learnability_smoke() -> None:
    cfg = small_config().scaled(lr=3e-3)
    backbone = ToyMultiDepthBackbone(vocab_size=97, width=32, n_layers=4, seed=3)
    bank = build_toy_bank(cfg, backbone)
    system = build_broca_system(backbone, cfg, device="cpu")
    optimizer = torch.optim.AdamW(trainable_parameter_groups(system, cfg))
    item = bank.split_items("meta_train")[0]
    first_loss = None
    last_loss = None
    last_advantages = None
    for _ in range(60):
        with torch.no_grad():
            neutral = run_arm(system, bank, item, "no_exposure")
        wrong = run_arm(system, bank, item, "wrong_passage")
        exposed = run_arm(system, bank, item, "normal")
        loss, advantages = paired_contrast_loss(
            exposed, wrong, neutral, margin=cfg.contrast_margin
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            cfg.grad_clip,
        )
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
        last_advantages = advantages
    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss, (
        f"paired loss did not descend on the toy curriculum ({first_loss} -> {last_loss})"
    )
    assert last_advantages is not None and float(last_advantages[1]) > 0, (
        "correct passage must beat wrong passage on the trained toy question"
    )


TESTS = [
    test_config_contracts,
    test_resolve_depths,
    test_injection_bounds_and_floor_identity,
    test_zero_init_graft_is_silent_and_arms_behave,
    test_output_tissue_read_boundary,
    test_arm_distinctness_after_recruitment,
    test_gradient_recruitment_and_frozen_backbone,
    test_paired_loss_arithmetic_and_detachment,
    test_leakage_lint_and_guards,
    test_mate_constraints,
    test_checkpoint_roundtrip,
    test_learnability_smoke,
]


def main() -> None:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as error:  # noqa: BLE001 - gate reports every failure
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
    if failures:
        raise SystemExit(f"{failures} contract(s) failed")
    print(f"all {len(TESTS)} fable2 contracts passed")


if __name__ == "__main__":
    main()
