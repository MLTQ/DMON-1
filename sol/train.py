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
from .routing import (
    RoutingTrafficConfig,
    RoutingTrafficTrial,
    routing_traffic_due,
    routing_traffic_update,
)
from .structure import (
    StructuralConfig,
    StructuralProbation,
    StructuralUpdate,
    accumulate_structural_credit,
    apply_structural_phase,
    next_probe_sources,
    structural_decision_due,
)
from .stream import CharacterVocabulary, ContinuousCharStream


@dataclass(frozen=True)
class TrainMetrics:
    loss: float
    perplexity: float
    mean_energy: float
    mean_viability: float
    quiescent_fraction: float
    energy_input: float
    energy_spent: float
    energy_transport_drift: float
    mean_backward_credit: float
    mean_output_error_credit: float
    mean_credit_routing_eligibility: float
    mean_credit_routing_preference: float
    credit_routing_preference_saturation: float
    routing_trial_active: bool
    routing_trial_started: bool
    routing_trial_resolved: bool
    routing_trial_committed: bool
    routing_candidate_exposed: bool
    routing_trials_started: int
    routing_trials_committed: int
    routing_trials_rejected: int
    routing_trial_advantage: float
    routing_trial_randomization_p_value: float
    routing_trial_target: int
    routing_trial_slot: int
    routing_candidate_observations: int
    routing_incumbent_observations: int
    routing_candidate_reward: float
    routing_incumbent_reward: float
    mean_novelty: float
    cell_credit: float
    edge_credit: float
    mean_fast_weight: float
    mean_edge_eligibility: float
    fast_weight_saturation: float
    mean_probe_eligibility: float
    mean_probe_flow: float
    mean_probe_fitness: float
    rewired_edges: int
    spawned_edges: int
    pruned_edges: int
    total_rewires: int
    total_spawns: int
    total_prunes: int
    candidate_advantage: float
    rejected_rewires: int
    probation_active: bool
    probation_started: bool
    probation_committed: int
    probation_rolled_back: int
    probation_rejected: int
    probation_advantage: float
    probation_randomization_p_value: float
    probation_candidate_exposed: bool
    probation_candidate_observations: int
    probation_incumbent_observations: int
    probation_candidate_reward: float
    probation_incumbent_reward: float
    prequential_advantage: float
    prequential_baseline: float


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
        structural_config: StructuralConfig | None = None,
        routing_traffic_config: RoutingTrafficConfig | None = None,
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
        self.structural_config = structural_config or StructuralConfig()
        self.routing_traffic_config = (
            routing_traffic_config or RoutingTrafficConfig()
        )
        parameters_by_name = dict(self.model.named_parameters())
        unknown_frozen = sorted(set(frozen_parameters) - set(parameters_by_name))
        if unknown_frozen:
            raise ValueError(f"unknown frozen parameters: {unknown_frozen}")
        self.frozen_parameters = tuple(sorted(set(frozen_parameters)))
        if (
            self.structural_config.enabled
            and self.model.cfg.structural_probe_gain <= 0
        ):
            raise ValueError(
                "structural plasticity requires a positive probe gain"
            )
        if (
            self.structural_config.min_active_dendrites
            > self.model.cfg.dendrites
        ):
            raise ValueError(
                "minimum active dendrites cannot exceed slot capacity"
            )
        if self.routing_traffic_config.enabled:
            if not self.model.cfg.exploratory_output_credit_routing:
                raise ValueError(
                    "routing traffic requires exploratory output-credit routing"
                )
            if self.model.cfg.output_error_credit_gain <= 0:
                raise ValueError(
                    "routing traffic requires output-error credit"
                )
            if (
                self.routing_traffic_config.proposal_step
                > self.model.cfg.credit_routing_preference_limit
            ):
                raise ValueError(
                    "routing traffic proposal step cannot exceed "
                    "the preference limit"
                )
        frozen_edges = bool(
            {"edge_weight", "edge_bias"}.intersection(
                self.frozen_parameters
            )
        )
        if (
            self.structural_config.enabled
            and self.structural_config.allow_rewiring
            and frozen_edges
        ):
            raise ValueError(
                "structural plasticity is incompatible with frozen edges"
            )
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
        self.structural_probation = StructuralProbation()
        self.routing_traffic_trial = RoutingTrafficTrial()
        self.prequential_advantage_ema = 0.0
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
            "structural_config": asdict(self.structural_config),
            "structural_probation": self.structural_probation.state_dict(),
            "routing_traffic_config": asdict(
                self.routing_traffic_config
            ),
            "routing_traffic_trial": (
                self.routing_traffic_trial.state_dict()
            ),
            "prequential_advantage_ema": self.prequential_advantage_ema,
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
            structural_config=StructuralConfig(
                **payload.get("structural_config", {})
            ),
            routing_traffic_config=RoutingTrafficConfig(
                **payload.get("routing_traffic_config", {})
            ),
            device=device,
        )
        trainer.model.load_compatible_state_dict(payload["model"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.state = trainer.model.state_from_snapshot(
            payload["field_state"],
            int(payload["batch_size"]),
            trainer.device,
        )
        trainer.structural_probation = StructuralProbation.from_state_dict(
            payload.get("structural_probation"),
            trainer.device,
        )
        trainer.routing_traffic_trial = (
            RoutingTrafficTrial.from_state_dict(
                payload.get("routing_traffic_trial"),
                trainer.device,
            )
        )
        trainer.prequential_advantage_ema = float(
            payload.get("prequential_advantage_ema", 0.0)
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
        probe_mask, candidate_exposed = (
            self.structural_probation.exploratory_probe_mask(
                self.model
            )
        )
        routing_candidate_exposed = (
            self.routing_traffic_trial.candidate_exposed
        )
        routing_delta = self.routing_traffic_trial.active_delta()
        self.optimizer.zero_grad(set_to_none=True)
        logits, next_state, trace = self.model.forward_sequence(
            inputs,
            self.state.detached(),
            targets=targets,
            retain_credit=True,
            structural_probe_mask=probe_mask,
            credit_routing_preference_delta=routing_delta,
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
        next_update = self.updates + 1
        edge_evidence = torch.stack(
            trace.structural_edge_evidence
        ).mean(dim=0)
        probe_evidence = torch.stack(
            trace.structural_probe_evidence
        ).mean(dim=0)
        probe_fitness = trace.probe_fitness().mean(dim=0)
        edge_usage = torch.stack(trace.edge_flow).mean(dim=0)
        edge_vector_evidence = torch.stack(
            trace.structural_edge_vector_evidence
        ).mean(dim=0)
        probe_vector_evidence = torch.stack(
            trace.structural_probe_vector_evidence
        ).mean(dim=0)
        prequential_advantage = float(
            torch.stack(trace.prequential_reward).mean().item()
        )
        accumulate_structural_credit(
            self.model,
            edge_evidence,
            probe_evidence,
            probe_fitness,
            self.structural_config,
            edge_usage=edge_usage,
            edge_vector_evidence=edge_vector_evidence,
            probe_vector_evidence=probe_vector_evidence,
        )
        baseline_before = self.prequential_advantage_ema
        self.routing_traffic_trial.observe(
            prequential_advantage,
            routing_candidate_exposed,
        )
        routing_resolved = False
        routing_committed = False
        if (
            self.routing_traffic_trial.active
            and self.routing_traffic_trial.observations
            >= self.routing_traffic_config.trial_updates
        ):
            routing_committed = self.routing_traffic_trial.resolve(
                self.model,
                self.state,
                self.routing_traffic_config,
                next_update,
            )
            routing_resolved = True
        exploratory_trial = (
            self.structural_probation.active
            and self.structural_probation.exploratory_traffic
        )
        self.structural_probation.observe(
            prequential_advantage,
            (
                candidate_exposed
                if exploratory_trial
                else None
            ),
        )
        resolved = False
        if (
            self.structural_probation.active
            and (
                (
                    self.structural_probation.exploratory_traffic
                    and self.structural_probation.observations
                    >= self.structural_config.probation_updates
                )
                or (
                    not self.structural_probation.exploratory_traffic
                    and next_update
                    - self.structural_probation.started_update
                    >= self.structural_config.probation_updates
                )
            )
        ):
            self.structural_probation.resolve(
                self.model,
                self.state,
                self.optimizer,
                self.structural_config.probation_margin,
                self.structural_config.growth_cost,
                self.structural_config.min_endpoint_energy,
                randomization_alpha=(
                    self.structural_config
                    .probation_randomization_alpha
                ),
                resolved_update=next_update,
            )
            self.model.probe_sources.copy_(
                next_probe_sources(
                    self.model.sources,
                    self.model.probe_sources,
                    self.model.active_edges,
                    prefer_local=(
                        self.structural_config.locality_gain > 0
                    ),
                )
            )
            self.model.structural_probe_credit.zero_()
            self.model.structural_probe_fitness.zero_()
            self.model.structural_probe_vector_credit.zero_()
            self.model.structural_probe_confirmations.zero_()
            self.state.probe_eligibility.zero_()
            resolved = True
        routing_started = False
        if (
            not self.routing_traffic_trial.active
            and not routing_resolved
            and not self.structural_probation.active
            and routing_traffic_due(
                self.routing_traffic_config,
                next_update,
                structural_decision_due=structural_decision_due(
                    self.structural_config,
                    next_update,
                ),
            )
        ):
            routing_started = self.routing_traffic_trial.begin(
                self.model,
                self.state,
                self.routing_traffic_config,
                next_update,
            )
        structural_update = (
            StructuralUpdate(
                0,
                int(self.model.total_rewires.item()),
                0.0,
                0,
                probation_active=self.structural_probation.active,
                probation_committed=(
                    self.structural_probation.total_committed
                ),
                probation_rolled_back=(
                    self.structural_probation.total_rolled_back
                ),
                probation_rejected=(
                    self.structural_probation.total_rejected
                ),
                probation_advantage=(
                    self.structural_probation.mean_advantage
                ),
                probation_candidate_exposed=candidate_exposed,
            )
            if resolved
            else (
                StructuralUpdate(
                    0,
                    int(self.model.total_rewires.item()),
                    0.0,
                    0,
                    probation_active=(
                        self.structural_probation.active
                    ),
                    probation_committed=(
                        self.structural_probation.total_committed
                    ),
                    probation_rolled_back=(
                        self.structural_probation.total_rolled_back
                    ),
                    probation_rejected=(
                        self.structural_probation.total_rejected
                    ),
                    probation_advantage=(
                        self.structural_probation.mean_advantage
                    ),
                    probation_candidate_exposed=(
                        candidate_exposed
                    ),
                )
                if (
                    self.routing_traffic_trial.active
                    and not routing_started
                )
                else apply_structural_phase(
                    self.model,
                    self.state,
                    self.optimizer,
                    self.structural_config,
                    next_update,
                    self.structural_probation,
                )
            )
        )
        routing_update = routing_traffic_update(
            self.routing_traffic_trial,
            started=routing_started,
            resolved=routing_resolved,
            committed=routing_committed,
            candidate_exposed=routing_candidate_exposed,
        )
        if structural_update.probation_started:
            self.structural_probation.baseline_advantage = baseline_before
        decay = self.structural_config.probation_baseline_decay
        self.prequential_advantage_ema = (
            decay * self.prequential_advantage_ema
            + (1 - decay) * prequential_advantage
        )
        self.updates = next_update
        mean_energy = torch.stack(trace.mean_energy).mean().item()
        mean_viability = torch.stack(trace.mean_viability).mean().item()
        quiescent_fraction = torch.stack(
            trace.quiescent_fraction
        ).mean().item()
        energy_input = torch.stack(trace.energy_input).mean().item()
        energy_spent = torch.stack(trace.energy_spent).mean().item()
        energy_transport_drift = torch.stack(
            trace.energy_transport_drift
        ).mean().item()
        mean_backward_credit = torch.stack(
            trace.mean_backward_credit
        ).mean().item()
        mean_output_error_credit = torch.stack(
            trace.mean_output_error_credit
        ).mean().item()
        mean_credit_routing_eligibility = torch.stack(
            trace.mean_credit_routing_eligibility
        ).mean().item()
        mean_credit_routing_preference = torch.stack(
            trace.mean_credit_routing_preference
        ).mean().item()
        credit_routing_preference_saturation = torch.stack(
            trace.credit_routing_preference_saturation
        ).mean().item()
        mean_novelty = torch.stack(trace.novelty).mean().item()
        mean_fast_weight = torch.stack(trace.mean_fast_weight).mean().item()
        mean_edge_eligibility = torch.stack(
            trace.mean_edge_eligibility
        ).mean().item()
        fast_weight_saturation = torch.stack(
            trace.fast_weight_saturation
        ).mean().item()
        mean_probe_eligibility = torch.stack(
            trace.mean_probe_eligibility
        ).mean().item()
        mean_probe_flow = torch.stack(trace.mean_probe_flow).mean().item()
        value = loss.item()
        return TrainMetrics(
            loss=value,
            perplexity=math.exp(min(20.0, value)),
            mean_energy=mean_energy,
            mean_viability=mean_viability,
            quiescent_fraction=quiescent_fraction,
            energy_input=energy_input,
            energy_spent=energy_spent,
            energy_transport_drift=energy_transport_drift,
            mean_backward_credit=mean_backward_credit,
            mean_output_error_credit=mean_output_error_credit,
            mean_credit_routing_eligibility=(
                mean_credit_routing_eligibility
            ),
            mean_credit_routing_preference=(
                mean_credit_routing_preference
            ),
            credit_routing_preference_saturation=(
                credit_routing_preference_saturation
            ),
            routing_trial_active=routing_update.active,
            routing_trial_started=routing_update.started,
            routing_trial_resolved=routing_update.resolved,
            routing_trial_committed=routing_update.committed,
            routing_candidate_exposed=(
                routing_update.candidate_exposed
            ),
            routing_trials_started=routing_update.total_started,
            routing_trials_committed=(
                routing_update.total_committed
            ),
            routing_trials_rejected=routing_update.total_rejected,
            routing_trial_advantage=routing_update.advantage,
            routing_trial_randomization_p_value=(
                routing_update.randomization_p_value
            ),
            routing_trial_target=routing_update.target,
            routing_trial_slot=routing_update.slot,
            routing_candidate_observations=(
                self.routing_traffic_trial.candidate_observations
            ),
            routing_incumbent_observations=(
                self.routing_traffic_trial.incumbent_observations
            ),
            routing_candidate_reward=(
                self.routing_traffic_trial.candidate_reward_sum
                / max(
                    1,
                    self.routing_traffic_trial.candidate_observations,
                )
            ),
            routing_incumbent_reward=(
                self.routing_traffic_trial.incumbent_reward_sum
                / max(
                    1,
                    self.routing_traffic_trial.incumbent_observations,
                )
            ),
            mean_novelty=mean_novelty,
            cell_credit=cell_credit,
            edge_credit=edge_credit,
            mean_fast_weight=mean_fast_weight,
            mean_edge_eligibility=mean_edge_eligibility,
            fast_weight_saturation=fast_weight_saturation,
            mean_probe_eligibility=mean_probe_eligibility,
            mean_probe_flow=mean_probe_flow,
            mean_probe_fitness=float(
                self.model.structural_probe_fitness.mean().item()
            ),
            rewired_edges=structural_update.rewired_edges,
            spawned_edges=structural_update.spawned_edges,
            pruned_edges=structural_update.pruned_edges,
            total_rewires=structural_update.total_rewires,
            total_spawns=int(self.model.total_spawns.item()),
            total_prunes=int(self.model.total_prunes.item()),
            candidate_advantage=structural_update.candidate_advantage,
            rejected_rewires=(
                structural_update.rejected_for_reachability
            ),
            probation_active=structural_update.probation_active,
            probation_started=structural_update.probation_started,
            probation_committed=(
                self.structural_probation.total_committed
            ),
            probation_rolled_back=(
                self.structural_probation.total_rolled_back
            ),
            probation_rejected=(
                self.structural_probation.total_rejected
            ),
            probation_advantage=(
                self.structural_probation.mean_advantage
            ),
            probation_randomization_p_value=(
                self.structural_probation.randomization_p_value
                if self.structural_probation.active
                else (
                    self.structural_probation
                    .last_randomization_p_value
                )
            ),
            probation_candidate_exposed=candidate_exposed,
            probation_candidate_observations=(
                self.structural_probation.candidate_observations
            ),
            probation_incumbent_observations=(
                self.structural_probation.incumbent_observations
            ),
            probation_candidate_reward=(
                self.structural_probation.candidate_reward_sum
                / max(
                    1,
                    self.structural_probation.candidate_observations,
                )
            ),
            probation_incumbent_reward=(
                self.structural_probation.incumbent_reward_sum
                / max(
                    1,
                    self.structural_probation.incumbent_observations,
                )
            ),
            prequential_advantage=prequential_advantage,
            prequential_baseline=self.prequential_advantage_ema,
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
    parser.add_argument("--initial-active-dendrites", type=int, default=None)
    parser.add_argument("--message-steps", type=int, default=3)
    parser.add_argument("--cell-reward-gain", type=float, default=0.25)
    parser.add_argument(
        "--backward-credit-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--backward-credit-decay",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--output-error-credit-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output-error-credit-decay",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--reward-plastic-output-credit-routing",
        action="store_true",
    )
    parser.add_argument(
        "--exploratory-output-credit-routing",
        action="store_true",
    )
    parser.add_argument(
        "--credit-routing-preference-decay",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--credit-routing-plasticity-gain",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--credit-routing-preference-limit",
        type=float,
        default=0.25,
    )
    parser.add_argument("--routing-traffic-interval", type=int, default=25)
    parser.add_argument("--routing-traffic-warmup", type=int, default=75)
    parser.add_argument("--routing-traffic-updates", type=int, default=20)
    parser.add_argument(
        "--routing-traffic-randomization-alpha",
        type=float,
        default=0.10,
    )
    parser.add_argument("--routing-traffic-margin", type=float, default=0.0)
    parser.add_argument("--routing-traffic-step", type=float, default=0.05)
    parser.add_argument(
        "--routing-traffic-minimum-eligibility",
        type=float,
        default=1e-6,
    )
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
        initial_active_dendrites=args.initial_active_dendrites,
        message_steps=args.message_steps,
        topology_seed=args.seed,
        reward_gain=args.cell_reward_gain,
        backward_credit_gain=args.backward_credit_gain,
        backward_credit_decay=args.backward_credit_decay,
        output_error_credit_gain=args.output_error_credit_gain,
        output_error_credit_decay=args.output_error_credit_decay,
        reward_plastic_output_credit_routing=(
            args.reward_plastic_output_credit_routing
        ),
        exploratory_output_credit_routing=(
            args.exploratory_output_credit_routing
        ),
        credit_routing_preference_decay=(
            args.credit_routing_preference_decay
        ),
        credit_routing_plasticity_gain=(
            args.credit_routing_plasticity_gain
        ),
        credit_routing_preference_limit=(
            args.credit_routing_preference_limit
        ),
    )
    model = SparseAxonField(cfg)
    trainer = ContinuousTrainer(
        model,
        text,
        vocabulary,
        batch_size=args.batch,
        chunk_length=args.chunk,
        learning_rate=args.lr,
        routing_traffic_config=RoutingTrafficConfig(
            enabled=args.exploratory_output_credit_routing,
            interval=args.routing_traffic_interval,
            warmup_updates=args.routing_traffic_warmup,
            trial_updates=args.routing_traffic_updates,
            randomization_alpha=(
                args.routing_traffic_randomization_alpha
            ),
            margin=args.routing_traffic_margin,
            proposal_step=args.routing_traffic_step,
            minimum_eligibility=(
                args.routing_traffic_minimum_eligibility
            ),
        ),
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
