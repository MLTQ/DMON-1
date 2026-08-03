"""Versioned tick runtime with a background truncated-learning worker."""

from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass

import torch

from .config import Sol2Config
from .model import Sol2
from .optim import build_optimizer, guarded_step, set_learning_rate
from .state import OrganismState


@dataclass
class LearningWindow:
    initial_state: OrganismState
    inputs: torch.Tensor
    targets: torch.Tensor
    base_version: int


@dataclass
class RuntimeStats:
    ticks: int = 0
    queued_windows: int = 0
    learned_windows: int = 0
    rejected_windows: int = 0
    stale_windows: int = 0
    weight_version: int = 0


class AsyncOrganismRuntime:
    """Keep ticking one active organism while a shadow copy learns.

    This is an engineering proof of the concurrency boundary, not the S0-R training
    harness. The active state is never rolled back when a new parameter version lands.
    """

    def __init__(
        self,
        model: Sol2,
        cfg: Sol2Config,
        device: str,
        *,
        learning_window: int | None = None,
        max_staleness: int = 2,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.learning_window = learning_window or cfg.chunk_length
        self.max_staleness = max_staleness
        self.state = model.initial_state(1, device)
        self.stats = RuntimeStats()
        self._lock = threading.RLock()
        self._jobs: queue.Queue[LearningWindow | None] = queue.Queue(maxsize=4)
        self._learner = copy.deepcopy(model).to(device)
        self._optimizer = build_optimizer(self._learner, cfg)
        self._worker = threading.Thread(target=self._learn_loop, daemon=True)
        self._closed = False
        self._previous_token: torch.Tensor | None = None
        self._previous_start: OrganismState | None = None
        self._window_start: OrganismState | None = None
        self._window_inputs: list[torch.Tensor] = []
        self._window_targets: list[torch.Tensor] = []
        self._worker.start()

    @torch.no_grad()
    def observe(self, token: int | torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("runtime is closed")
        token_tensor = torch.as_tensor(token, dtype=torch.long, device=self.device).reshape(1)
        with self._lock:
            start = self.state.clone_detached()
            if self._previous_token is not None:
                if not self._window_inputs:
                    self._window_start = self._previous_start
                self._window_inputs.append(self._previous_token.detach().clone())
                self._window_targets.append(token_tensor.detach().clone())
                if len(self._window_inputs) == self.learning_window:
                    self._enqueue_window()
            logits, self.state, _ = self.model.step(token_tensor, self.state)
            self.state.weight_version = self.stats.weight_version
            self._previous_token = token_tensor
            self._previous_start = start
            self.stats.ticks += 1
            return logits.detach()

    def _enqueue_window(self) -> None:
        assert self._window_start is not None
        job = LearningWindow(
            initial_state=self._window_start,
            inputs=torch.stack(self._window_inputs, dim=1),
            targets=torch.stack(self._window_targets, dim=1),
            base_version=self._window_start.weight_version,
        )
        try:
            self._jobs.put_nowait(job)
            self.stats.queued_windows += 1
        except queue.Full:
            self.stats.stale_windows += 1
        self._window_start = None
        self._window_inputs = []
        self._window_targets = []

    def _learn_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return
            try:
                if self.stats.weight_version - job.base_version > self.max_staleness:
                    self.stats.stale_windows += 1
                    continue
                self._learner.train()
                set_learning_rate(
                    self._optimizer, self.cfg, max(self.stats.learned_windows + 1, 1)
                )
                state = job.initial_state
                loss, _, _ = self._learner.forward_chunk(job.inputs, job.targets, state)
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                health = guarded_step(self._learner, self._optimizer, self.cfg)
                if not health.accepted:
                    self.stats.rejected_windows += 1
                    with self._lock:
                        self._learner.load_state_dict(self.model.state_dict())
                    continue
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(self.device))
                with self._lock:
                    self.model.load_state_dict(self._learner.state_dict())
                    self.stats.weight_version += 1
                    self.stats.learned_windows += 1
            finally:
                self._jobs.task_done()

    def wait_for_learning(self) -> None:
        self._jobs.join()

    def close(self) -> None:
        if self._closed:
            return
        self._jobs.put(None)
        self._worker.join()
        self._closed = True

    def __enter__(self) -> "AsyncOrganismRuntime":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
