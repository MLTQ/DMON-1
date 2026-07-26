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

    def _fallback_perplexity(self) -> float:
        evaluation = self.loaded.metadata.get("evaluation", {})
        persistent = (
            evaluation.get("persistent", {})
            if isinstance(evaluation, dict)
            else {}
        )
        return float(persistent.get("perplexity", math.nan))

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

            generator = torch.Generator(device=self.device)
            if seed is None:
                generator.seed()
            else:
                generator.manual_seed(seed)
            output: list[int] = []
            with torch.no_grad():
                logits, state, diagnostics = model.tick(
                    state, encoded[:, -1]
                )
                novelty_values = [
                    float(diagnostics["novelty"].mean().item())
                ]
                flow_values = [
                    float(diagnostics["edge_flow"].mean().item())
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
                    novelty_values.append(
                        float(diagnostics["novelty"].mean().item())
                    )
                    flow_values.append(
                        float(diagnostics["edge_flow"].mean().item())
                    )

            self.loaded.state = state.detached()
            return {
                "output": vocabulary.decode(output),
                "mode": "live-checkpoint",
                "metrics": {
                    "energy": float(state.energy.mean().item()),
                    "novelty": sum(novelty_values) / len(novelty_values),
                    "cellCredit": cell_credit,
                    "edgeCredit": edge_credit,
                    "perplexity": perplexity,
                    "edgeFlow": sum(flow_values) / len(flow_values),
                },
                "checkpoint": {
                    "name": self.checkpoint.name,
                    "updates": self.loaded.updates,
                    "cells": model.cfg.cells,
                    "dendrites": model.cfg.dendrites,
                },
            }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            state = self.loaded.state
            return {
                "mode": "live-checkpoint",
                "checkpoint": {
                    "name": self.checkpoint.name,
                    "updates": self.loaded.updates,
                },
                "metrics": {
                    "energy": float(state.energy.mean().item()),
                    "stimulation": float(state.stimulation.mean().item()),
                    "eligibility": float(state.eligibility.abs().mean().item()),
                },
                "topology": {
                    "sources": self.loaded.model.sources.cpu().tolist(),
                    "weights": torch.tanh(
                        self.loaded.model.edge_weight.detach()
                    )
                    .cpu()
                    .tolist(),
                },
            }


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
            if self.path != "/generate":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("content-length", "0"))
                if content_length > 64_000:
                    raise ValueError("request is too large")
                payload = json.loads(self.rfile.read(content_length) or b"{}")
                prompt = payload.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt is required")
                length = int(payload.get("length", 160))
                length = max(1, min(512, length))
                temperature = float(payload.get("temperature", 0.8))
                seed = payload.get("seed")
                if seed is not None:
                    seed = int(seed)
                response = organism.generate(
                    prompt.strip()[:2000],
                    length,
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
    parser.add_argument("--checkpoint", type=Path, required=True)
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
