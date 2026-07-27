"""Present different organisms under one interface so experiments compare fairly.

An experiment should not know which substrate it is driving. Without this, each new arm
tempts a slightly different training loop, and then a difference in results is a
difference in harnesses rather than in architectures — which is the same class of error
as an unmatched parameter budget, and just as invisible.

The contract every arm satisfies:

    blank_state(batch, device) -> state
    tick(state, token, history) -> (state, logits)
    detach(state)              -> state with the graph cut but values kept
    parameters(), param_count(), to(device)

`history` is accepted by every arm and used only by those that want it. The lattice
writes it into mirror cells; SOL ignores it because its memory is internal, which is a
real architectural difference and not a handicap to equalise.
"""

from __future__ import annotations

import torch


class SolArm:
    """Adapter for `dmon.organism.model.SparseAxonField`.

    SOL's tick returns `(logits, state, aux)` and carries a structured `FieldState`
    rather than a tensor, so both the return order and the detach step differ from a
    plain recurrent model. Getting `detached()` wrong here would silently either leak
    the graph across truncation boundaries (unbounded memory) or drop persistent
    energy/eligibility state at every window (destroying the mechanisms under test).
    """

    def __init__(self, cfg):
        from ..organism.model import SparseAxonField

        self.field = SparseAxonField(cfg)
        self.cfg = cfg
        self.last_aux: dict | None = None

    def blank_state(self, batch: int, device=None):
        return self.field.initial_state(batch, device)

    def tick(self, state, token, history=None):
        logits, state, aux = self.field.tick(state, token)
        self.last_aux = aux
        return state, logits

    @staticmethod
    def detach(state):
        return state.detached()

    def parameters(self):
        return self.field.parameters()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.field.parameters())

    def to(self, device):
        self.field = self.field.to(device)
        return self

    def train(self, mode: bool = True):
        self.field.train(mode)
        return self

    def eval(self):
        return self.train(False)


class TensorStateArm:
    """Adapter for models whose state is a plain tensor (the lattice, the GRU)."""

    def __init__(self, model):
        self.model = model

    def blank_state(self, batch: int, device=None):
        return self.model.blank_state(batch, device)

    def tick(self, state, token, history=None):
        return self.model.tick(state, token, history)

    @staticmethod
    def detach(state):
        return state.detach() if state is not None else None

    def parameters(self):
        return self.model.parameters()

    def param_count(self) -> int:
        return self.model.param_count()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)


def sol_matched_to(target_params: int, vocab: int, **overrides):
    """Largest SOL organism whose parameter count does not exceed the budget.

    Searched rather than solved, for the same reason the GRU and transformer baselines
    are: the closed form is easy to get subtly wrong, and an under-budgeted arm makes a
    comparison meaningless in the direction that flatters whoever wrote it.
    """
    from ..organism.model import SolConfig

    best = None
    for cells in range(16, 257, 8):
        cfg = SolConfig(vocab_size=vocab, cells=cells,
                        channels=overrides.get("channels", 64),
                        dendrites=overrides.get("dendrites", 8),
                        sensory_cells=min(8, cells // 4),
                        output_cells=min(8, cells // 4),
                        **{k: v for k, v in overrides.items()
                           if k not in ("channels", "dendrites")})
        try:
            n = SolArm(cfg).param_count()
        except Exception:
            break
        if n <= target_params:
            best = cfg
        else:
            break
    if best is None:
        raise RuntimeError(f"no SOL config fits {target_params} parameters")
    return SolArm(best)
