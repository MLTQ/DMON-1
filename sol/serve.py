"""Local-only HTTP bridge from a live SOL checkpoint to the character console."""

from __future__ import annotations

import argparse
import json
import math
import threading
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .checkpoint import LoadedOrganism, load_organism


class LiveOrganism:
    """Serializes interaction with one persistent checkpoint state."""

    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.loaded: LoadedOrganism = load_organism(self.checkpoint, device)
        self.device = torch.device(device)
        self.lock = threading.Lock()
        model = self.loaded.model
        self.clock_ticks = 0
        self.last_output = ""
        self.last_input: str | None = None
        self.last_cell_credit = 0.0
        self.last_edge_credit = 0.0
        self.last_perplexity = self._fallback_perplexity()
        self.last_novelty = 0.0
        self.last_edge_flow = torch.zeros(
            model.cfg.cells,
            model.cfg.dendrites,
            dtype=self.loaded.state.energy.dtype,
        )
        self.last_probe_flow = torch.zeros(
            model.cfg.cells,
            dtype=self.loaded.state.energy.dtype,
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.seed()

    def _fallback_perplexity(self) -> float:
        evaluation = self.loaded.metadata.get("evaluation", {})
        persistent = (
            evaluation.get("persistent", {})
            if isinstance(evaluation, dict)
            else {}
        )
        return float(persistent.get("perplexity", math.nan))

    def _remember_diagnostics(
        self,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        self.last_novelty = float(diagnostics["novelty"].mean().item())
        self.last_edge_flow = diagnostics["edge_flow"].detach().cpu()
        self.last_probe_flow = diagnostics["probe_flow"].detach().cpu()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        state = self.loaded.state
        model = self.loaded.model
        viability = model._viability(state.energy)
        metabolism_enabled = (
            model.cfg.basal_cost > 0
            or model.cfg.activity_cost > 0
            or model.cfg.stimulation_gain > 0
        )
        return {
            "mode": "live-checkpoint",
            "checkpoint": {
                "name": self.checkpoint.name,
                "updates": self.loaded.updates,
                "cells": model.cfg.cells,
                "dendrites": model.cfg.dendrites,
                "metabolismEnabled": metabolism_enabled,
            },
            "clock": {
                "ticks": self.clock_ticks,
                "lastInput": self.last_input,
                "lastOutput": self.last_output,
            },
            "metrics": {
                "energy": float(state.energy.mean().item()),
                "viability": float(viability.mean().item()),
                "quiescentFraction": float(
                    (
                        state.energy
                        <= model.cfg.quiescence_energy
                    )
                    .to(state.energy.dtype)
                    .mean()
                    .item()
                ),
                "stimulation": float(state.stimulation.mean().item()),
                "eligibility": float(state.eligibility.abs().mean().item()),
                "backwardCredit": float(
                    state.backward_credit.abs().mean().item()
                ),
                "outputErrorCredit": float(
                    state.output_error_credit.abs().mean().item()
                ),
                "edgeEligibility": float(
                    state.edge_eligibility.abs().mean().item()
                ),
                "fastWeight": float(state.fast_weight.abs().mean().item()),
                "fastSaturation": float(
                    (
                        state.fast_weight.abs()
                        >= 0.95 * model.cfg.fast_weight_limit
                    )
                    .to(state.fast_weight.dtype)
                    .mean()
                    .item()
                ),
                "rewardBaseline": float(
                    state.reward_baseline.mean().item()
                ),
                "probeEligibility": float(
                    state.probe_eligibility.abs().mean().item()
                ),
                "structuralRewires": int(model.total_rewires.item()),
                "novelty": self.last_novelty,
                "cellCredit": self.last_cell_credit,
                "edgeCredit": self.last_edge_credit,
                "perplexity": self.last_perplexity,
                "edgeFlow": float(self.last_edge_flow.mean().item()),
            },
            "topology": {
                "sources": model.sources.cpu().tolist(),
                "activeEdges": model.active_edges.cpu().tolist(),
                "probeSources": model.probe_sources.cpu().tolist(),
                "structuralEdgeCredit": (
                    model.structural_edge_credit.cpu().tolist()
                ),
                "structuralProbeCredit": (
                    model.structural_probe_credit.cpu().tolist()
                ),
                "weights": torch.tanh(
                    model.edge_weight.detach()
                )
                .cpu()
                .tolist(),
                "fastWeights": state.fast_weight[0].cpu().tolist(),
                "edgeFlow": self.last_edge_flow.tolist(),
                "probeFlow": self.last_probe_flow.tolist(),
                "cellActivity": state.stimulation[0].cpu().tolist(),
                "cellEnergy": state.energy[0].cpu().tolist(),
                "cellViability": viability[0].cpu().tolist(),
                "sensoryCells": model.sensory_indices.cpu().tolist(),
                "outputCells": model.output_indices.cpu().tolist(),
            },
        }

    def generate(
        self,
        prompt: str,
        characters: int,
        *,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Stimulate with a prompt, measure real backward credit, and emit characters."""

        if not prompt:
            raise ValueError("prompt must not be empty")
        if characters < 1:
            raise ValueError("characters must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        with self.lock:
            model = self.loaded.model
            vocabulary = self.loaded.vocabulary
            encoded = vocabulary.encode(prompt, self.device).unsqueeze(0)
            state = self.loaded.state.detached()
            elapsed_ticks = 0
            model.zero_grad(set_to_none=True)
            cell_credit = 0.0
            edge_credit = 0.0
            perplexity = self._fallback_perplexity()

            if encoded.shape[1] > 1:
                logits, state, trace = model.forward_sequence(
                    encoded[:, :-1],
                    state,
                    targets=encoded[:, 1:],
                    retain_credit=True,
                )
                prompt_loss = F.cross_entropy(
                    logits.flatten(0, 1), encoded[:, 1:].flatten()
                )
                prompt_loss.backward()
                cell_credit = float(trace.cell_credit().mean().item())
                edge_gradient = model.edge_weight.grad
                if edge_gradient is not None:
                    edge_credit = float(
                        (edge_gradient * model.edge_weight).abs().mean().item()
                    )
                perplexity = math.exp(min(20.0, float(prompt_loss.item())))
                state = state.detached()
                model.zero_grad(set_to_none=True)
                elapsed_ticks += encoded.shape[1] - 1

            if seed is None:
                generator = self.generator
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
            output: list[int] = []
            with torch.no_grad():
                logits, state, diagnostics = model.tick(
                    state, encoded[:, -1]
                )
                elapsed_ticks += 1
                novelty_values = [
                    float(diagnostics["novelty"].mean().item())
                ]
                viability_values = [
                    float(diagnostics["mean_viability"].mean().item())
                ]
                quiescent_values = [
                    float(
                        diagnostics["quiescent_fraction"].mean().item()
                    )
                ]
                energy_input_values = [
                    float(diagnostics["energy_input"].mean().item())
                ]
                energy_spent_values = [
                    float(diagnostics["energy_spent"].mean().item())
                ]
                energy_drift_values = [
                    float(
                        diagnostics["energy_transport_drift"].mean().item()
                    )
                ]
                flow_values = [
                    float(diagnostics["edge_flow"].mean().item())
                ]
                fast_weight_values = [
                    float(diagnostics["mean_fast_weight"].mean().item())
                ]
                fast_saturation_values = [
                    float(
                        diagnostics["fast_weight_saturation"].mean().item()
                    )
                ]
                probe_flow_values = [
                    float(diagnostics["probe_flow"].mean().item())
                ]

                for _ in range(characters):
                    probabilities = torch.softmax(
                        logits / temperature, dim=-1
                    )
                    token = torch.multinomial(
                        probabilities, 1, generator=generator
                    ).squeeze(1)
                    output.append(int(token.item()))
                    logits, state, diagnostics = model.tick(state, token)
                    elapsed_ticks += 1
                    novelty_values.append(
                        float(diagnostics["novelty"].mean().item())
                    )
                    viability_values.append(
                        float(
                            diagnostics["mean_viability"].mean().item()
                        )
                    )
                    quiescent_values.append(
                        float(
                            diagnostics["quiescent_fraction"].mean().item()
                        )
                    )
                    energy_input_values.append(
                        float(diagnostics["energy_input"].mean().item())
                    )
                    energy_spent_values.append(
                        float(diagnostics["energy_spent"].mean().item())
                    )
                    energy_drift_values.append(
                        float(
                            diagnostics["energy_transport_drift"]
                            .mean()
                            .item()
                        )
                    )
                    flow_values.append(
                        float(diagnostics["edge_flow"].mean().item())
                    )
                    fast_weight_values.append(
                        float(
                            diagnostics["mean_fast_weight"].mean().item()
                        )
                    )
                    fast_saturation_values.append(
                        float(
                            diagnostics["fast_weight_saturation"]
                            .mean()
                            .item()
                        )
                    )
                    probe_flow_values.append(
                        float(diagnostics["probe_flow"].mean().item())
                    )

            decoded = vocabulary.decode(output)
            self.loaded.state = state.detached()
            self.clock_ticks += elapsed_ticks
            self.last_output = decoded[-1:] if decoded else ""
            self.last_input = self.last_output or prompt[-1:]
            self.last_cell_credit = cell_credit
            self.last_edge_credit = edge_credit
            self.last_perplexity = perplexity
            self._remember_diagnostics(diagnostics)
            payload = self._snapshot_unlocked()
            payload["output"] = decoded
            payload["metrics"].update(
                {
                    "energy": float(state.energy.mean().item()),
                    "viability": (
                        sum(viability_values) / len(viability_values)
                    ),
                    "quiescentFraction": (
                        sum(quiescent_values) / len(quiescent_values)
                    ),
                    "energyInput": (
                        sum(energy_input_values) / len(energy_input_values)
                    ),
                    "energySpent": (
                        sum(energy_spent_values) / len(energy_spent_values)
                    ),
                    "energyTransportDrift": (
                        sum(energy_drift_values) / len(energy_drift_values)
                    ),
                    "novelty": sum(novelty_values) / len(novelty_values),
                    "cellCredit": cell_credit,
                    "edgeCredit": edge_credit,
                    "perplexity": perplexity,
                    "edgeFlow": sum(flow_values) / len(flow_values),
                    "fastWeight": (
                        sum(fast_weight_values) / len(fast_weight_values)
                    ),
                    "fastSaturation": (
                        sum(fast_saturation_values)
                        / len(fast_saturation_values)
                    ),
                    "probeFlow": (
                        sum(probe_flow_values) / len(probe_flow_values)
                    ),
                    "probeEligibility": float(
                        state.probe_eligibility.abs().mean().item()
                    ),
                    "backwardCredit": float(
                        state.backward_credit.abs().mean().item()
                    ),
                    "outputErrorCredit": float(
                        state.output_error_credit.abs().mean().item()
                    ),
                    "structuralRewires": int(
                        model.total_rewires.item()
                    ),
                }
            )
            return payload

    def advance_silence(
        self,
        steps: int = 1,
        *,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Advance genuine no-input ticks and sample the output organ."""

        if not 1 <= steps <= 64:
            raise ValueError("steps must be in [1, 64]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        with self.lock:
            model = self.loaded.model
            vocabulary = self.loaded.vocabulary
            state = self.loaded.state.detached()
            if seed is None:
                generator = self.generator
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
            output: list[int] = []
            novelty_values: list[float] = []
            viability_values: list[float] = []
            energy_input_values: list[float] = []
            energy_spent_values: list[float] = []

            with torch.no_grad():
                for _ in range(steps):
                    logits, state, diagnostics = model.tick(state, None)
                    probabilities = torch.softmax(
                        logits / temperature, dim=-1
                    )
                    token = torch.multinomial(
                        probabilities,
                        1,
                        generator=generator,
                    ).squeeze(1)
                    output.append(int(token.item()))
                    novelty_values.append(
                        float(diagnostics["novelty"].mean().item())
                    )
                    viability_values.append(
                        float(
                            diagnostics["mean_viability"].mean().item()
                        )
                    )
                    energy_input_values.append(
                        float(diagnostics["energy_input"].mean().item())
                    )
                    energy_spent_values.append(
                        float(diagnostics["energy_spent"].mean().item())
                    )

            decoded = vocabulary.decode(output)
            self.loaded.state = state.detached()
            self.clock_ticks += steps
            self.last_input = None
            self.last_output = decoded[-1:] if decoded else ""
            self._remember_diagnostics(diagnostics)
            payload = self._snapshot_unlocked()
            payload["output"] = decoded
            payload["metrics"].update(
                {
                    "novelty": sum(novelty_values) / len(novelty_values),
                    "viability": (
                        sum(viability_values) / len(viability_values)
                    ),
                    "energyInput": (
                        sum(energy_input_values)
                        / len(energy_input_values)
                    ),
                    "energySpent": (
                        sum(energy_spent_values)
                        / len(energy_spent_values)
                    ),
                }
            )
            return payload

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot_unlocked()


def make_handler(organism: LiveOrganism) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "mode": "live-checkpoint",
                        "checkpoint": organism.checkpoint.name,
                    },
                )
            elif self.path == "/snapshot":
                self._json(HTTPStatus.OK, organism.snapshot())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/generate", "/advance"}:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("content-length", "0"))
                if content_length > 64_000:
                    raise ValueError("request is too large")
                payload = json.loads(self.rfile.read(content_length) or b"{}")
                temperature = float(payload.get("temperature", 0.8))
                seed = payload.get("seed")
                if seed is not None:
                    seed = int(seed)
                if self.path == "/generate":
                    prompt = payload.get("prompt")
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError("prompt is required")
                    length = int(payload.get("length", 160))
                    length = max(1, min(512, length))
                    response = organism.generate(
                        prompt.strip()[:2000],
                        length,
                        temperature=temperature,
                        seed=seed,
                    )
                else:
                    response = organism.advance_silence(
                        int(payload.get("steps", 1)),
                        temperature=temperature,
                        seed=seed,
                    )
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._json(
                    HTTPStatus.BAD_REQUEST, {"error": str(error)}
                )
                return
            self._json(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[sol-bridge] {self.address_string()} {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("sol/runs/live.pt")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    organism = LiveOrganism(args.checkpoint, args.device)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(organism)
    )
    print(
        f"SOL checkpoint bridge listening on "
        f"http://{args.host}:{args.port} ({args.checkpoint})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSOL checkpoint bridge stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
