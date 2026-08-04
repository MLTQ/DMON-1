"""Complete resumable checkpoints for SOL2 training processes."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Sol2Config
from .state import OrganismState


def same_tensor_values(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare checkpoint invariants independently of map-location devices."""

    return torch.equal(left.detach().cpu(), right.detach().cpu())


def restore_rng_state(payload: dict) -> None:
    """Restore CPU and CUDA RNG tensors after device-aware checkpoint loading."""

    torch.set_rng_state(payload["torch_rng_state"].detach().cpu())
    cuda_state = payload.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_state])


def pack_state(state) -> dict:
    if isinstance(state, OrganismState):
        return {
            "kind": "organism",
            "hidden": state.hidden.detach().cpu(),
            "memory_cursor": state.memory_cursor,
            "weight_version": state.weight_version,
        }
    return {"kind": "tensor", "value": state.detach().cpu()}


def unpack_state(payload: dict, device: str):
    if payload["kind"] == "organism":
        return OrganismState(
            hidden=payload["hidden"].to(device),
            memory_cursor=int(payload["memory_cursor"]),
            weight_version=int(payload.get("weight_version", 0)),
        )
    return payload["value"].to(device)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state,
    stream,
    cfg: Sol2Config,
    kind: str,
    update: int,
    accepted_updates: int,
    rejected_updates: int,
    history: list[dict],
    evaluations: list[dict],
) -> None:
    payload = {
        "schema": 1,
        "kind": kind,
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": pack_state(state),
        "stream": stream.state_dict(),
        "update": update,
        "accepted_updates": accepted_updates,
        "rejected_updates": rejected_updates,
        "history": history,
        "evaluations": evaluations,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: str) -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != 1:
        raise ValueError("unsupported SOL2 checkpoint schema")
    return payload
