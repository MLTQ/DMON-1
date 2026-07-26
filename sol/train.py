"""Train and sample the SOL character organ on a continuous text stream."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .model import FieldState, SolConfig, SparseAxonField
from .stream import CharacterVocabulary, ContinuousCharStream


@dataclass(frozen=True)
class TrainMetrics:
    loss: float
    perplexity: float
    mean_energy: float
    mean_novelty: float
    cell_credit: float
    edge_credit: float
    mean_fast_weight: float
    mean_edge_eligibility: float
    fast_weight_saturation: float


class ContinuousTrainer:
    """Owns one model state, stream cursor, optimizer, and continuous credit loop."""

    def __init__(
        self,
        model: SparseAxonField,
        text: str,
        vocabulary: CharacterVocabulary,
        batch_size: int = 8,
        chunk_length: int = 16,
        learning_rate: float = 3e-3,
        grad_clip: float = 1.0,
        frozen_parameters: tuple[str, ...] = (),
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.vocabulary = vocabulary
        self.stream = ContinuousCharStream(
            text, vocabulary, batch_size, self.device
        )
        self.chunk_length = chunk_length
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        parameters_by_name = dict(self.model.named_parameters())
        unknown_frozen = sorted(set(frozen_parameters) - set(parameters_by_name))
        if unknown_frozen:
            raise ValueError(f"unknown frozen parameters: {unknown_frozen}")
        self.frozen_parameters = tuple(sorted(set(frozen_parameters)))
        for name in self.frozen_parameters:
            parameters_by_name[name].requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.state = self.model.initial_state(batch_size, self.device)
        self.updates = 0

    def state_dict(self) -> dict[str, Any]:
        """Serialize the uninterrupted training process, not only model weights."""

        return {
            "model_config": asdict(self.model.cfg),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "field_state": {
                field.name: getattr(self.state, field.name).detach().cpu()
                for field in fields(FieldState)
            },
            "stream": self.stream.state_dict(),
            "vocabulary": list(self.vocabulary.characters),
            "batch_size": self.stream.batch_size,
            "chunk_length": self.chunk_length,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "frozen_parameters": list(self.frozen_parameters),
            "updates": self.updates,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: dict[str, Any],
        text: str,
        device: torch.device | str = "cpu",
    ) -> "ContinuousTrainer":
        """Rebuild a trainer at the exact stream and organism state in a checkpoint."""

        vocabulary = CharacterVocabulary(tuple(payload["vocabulary"]))
        model = SparseAxonField(SolConfig(**payload["model_config"]))
        trainer = cls(
            model,
            text,
            vocabulary,
            batch_size=int(payload["batch_size"]),
            chunk_length=int(payload["chunk_length"]),
            learning_rate=float(payload["learning_rate"]),
            grad_clip=float(payload["grad_clip"]),
            frozen_parameters=tuple(payload.get("frozen_parameters", ())),
            device=device,
        )
        trainer.model.load_state_dict(payload["model"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.state = trainer.model.state_from_snapshot(
            payload["field_state"],
            int(payload["batch_size"]),
            trainer.device,
        )
        trainer.stream.load_state_dict(payload["stream"])
        trainer.updates = int(payload["updates"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_rng_state = payload.get("cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        return trainer

    def step(self) -> TrainMetrics:
        """Advance the stream once, backpropagate, and keep the detached organism."""

        inputs, targets = self.stream.next(self.chunk_length)
        self.optimizer.zero_grad(set_to_none=True)
        logits, next_state, trace = self.model.forward_sequence(
            inputs,
            self.state.detached(),
            targets=targets,
            retain_credit=True,
        )
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()

        cell_credit = trace.cell_credit().mean().item()
        edge_grad = self.model.edge_weight.grad
        edge_credit = (
            (edge_grad * self.model.edge_weight).abs().mean().item()
            if edge_grad is not None
            else 0.0
        )
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.state = next_state.detached()
        self.updates += 1
        mean_energy = torch.stack(trace.mean_energy).mean().item()
        mean_novelty = torch.stack(trace.novelty).mean().item()
        mean_fast_weight = torch.stack(trace.mean_fast_weight).mean().item()
        mean_edge_eligibility = torch.stack(
            trace.mean_edge_eligibility
        ).mean().item()
        fast_weight_saturation = torch.stack(
            trace.fast_weight_saturation
        ).mean().item()
        value = loss.item()
        return TrainMetrics(
            loss=value,
            perplexity=math.exp(min(20.0, value)),
            mean_energy=mean_energy,
            mean_novelty=mean_novelty,
            cell_credit=cell_credit,
            edge_credit=edge_credit,
            mean_fast_weight=mean_fast_weight,
            mean_edge_eligibility=mean_edge_eligibility,
            fast_weight_saturation=fast_weight_saturation,
        )


@torch.no_grad()
def generate(
    model: SparseAxonField,
    vocabulary: CharacterVocabulary,
    prompt: str,
    characters: int,
    state: FieldState | None = None,
    temperature: float = 0.8,
    device: torch.device | str = "cpu",
) -> tuple[str, FieldState]:
    """Continue a prompt one character at a time through the normal stream path."""

    if not prompt:
        raise ValueError("prompt must not be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = torch.device(device)
    model.eval()
    encoded = vocabulary.encode(prompt, device).unsqueeze(0)
    if state is None or state.hidden.shape[0] != 1:
        state = model.initial_state(1, device)

    logits = None
    for index in range(encoded.shape[1]):
        logits, state, _ = model.tick(state, encoded[:, index])

    output: list[int] = []
    assert logits is not None
    for _ in range(characters):
        probabilities = torch.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probabilities, 1).squeeze(1)
        output.append(int(token.item()))
        logits, state, _ = model.tick(state, token)
    model.train()
    return prompt + vocabulary.decode(output), state


def _default_text() -> str:
    return (
        "to be, or not to be: that is the question.\n"
        "the stream continues; the field remembers.\n"
    ) * 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--dendrites", type=int, default=4)
    parser.add_argument("--message-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--generate", type=int, default=160)
    parser.add_argument("--prompt", default="to ")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    text = args.file.read_text() if args.file else _default_text()
    vocabulary = CharacterVocabulary.from_text(text)
    cfg = SolConfig(
        vocab_size=len(vocabulary),
        cells=args.cells,
        channels=args.channels,
        dendrites=args.dendrites,
        message_steps=args.message_steps,
        topology_seed=args.seed,
    )
    model = SparseAxonField(cfg)
    trainer = ContinuousTrainer(
        model,
        text,
        vocabulary,
        batch_size=args.batch,
        chunk_length=args.chunk,
        learning_rate=args.lr,
        device=args.device,
    )

    for update in range(1, args.updates + 1):
        metrics = trainer.step()
        if update == 1 or update % args.log_every == 0:
            print(
                f"[{update:5d}] loss={metrics.loss:.4f} "
                f"ppl={metrics.perplexity:7.2f} energy={metrics.mean_energy:.3f} "
                f"novelty={metrics.mean_novelty:.3f} "
                f"cell_credit={metrics.cell_credit:.2e} "
                f"edge_credit={metrics.edge_credit:.2e}"
            )

    sample, _ = generate(
        model,
        vocabulary,
        args.prompt,
        args.generate,
        device=args.device,
    )
    print(f"\n--- sample ---\n{sample}")


if __name__ == "__main__":
    main()
