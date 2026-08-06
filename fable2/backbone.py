"""Frozen causal LMs that accept bounded control residuals at several depths."""

from __future__ import annotations

import torch
from torch import nn

from sol2.language_backbone import HuggingFaceFrozenBackbone


def resolve_depths(fractions: tuple[float, ...], n_layers: int) -> tuple[int, ...]:
    """Map depth fractions in (0, 1] to distinct post-block layer indices."""

    if n_layers < 1:
        raise ValueError("backbone must expose at least one decoder layer")
    indices = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("depth fractions must lie in (0, 1]")
        index = min(n_layers - 1, max(0, round(fraction * n_layers) - 1))
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"depth fractions {fractions} collide on {n_layers} layers; "
            "choose more separated fractions"
        )
    return tuple(indices)


def inject_control_residual(
    hidden: torch.Tensor, controls: torch.Tensor
) -> torch.Tensor:
    """Add an attention-weighted control residual to one depth's hidden stream.

    The residual is a convex combination of the control vectors, so its
    componentwise magnitude never exceeds the control bound, and exactly zero
    controls produce a bitwise no-op. Softmax runs in fp32 for stability under
    reduced-precision backbones.
    """

    if controls.ndim != 3 or controls.shape[0] != hidden.shape[0]:
        raise ValueError("controls must have shape [batch, tokens, width]")
    if controls.shape[2] != hidden.shape[2]:
        raise ValueError("control width must match the backbone hidden width")
    controls = controls.to(device=hidden.device, dtype=hidden.dtype)
    if not controls.requires_grad and not bool(controls.detach().any()):
        # Exact-zero evaluation controls short-circuit so the floor claim is a
        # no-op even in bf16. Never short-circuit a live graph: the zero-initialized
        # graft recruits through exactly this gradient path.
        return hidden
    scale = hidden.shape[2] ** -0.5
    scores = torch.einsum("bsw,btw->bst", hidden, controls) * scale
    attention = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
    return hidden + torch.einsum("bst,btw->bsw", attention, controls)


class MultiDepthBackbone(HuggingFaceFrozenBackbone):
    """A frozen Hugging Face decoder with per-depth control injection.

    Zero controls (or an empty control dict) reproduce the bare model exactly, so
    the no-control floor is the bare-LM floor: there is no scaffold, unlike the
    C1n continuous prefix, whose "floor" still carried prepended embeddings.
    """

    def decoder_layers(self) -> nn.ModuleList:
        for holder in (getattr(self.model, "model", None), self.model):
            layers = getattr(holder, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers):
                return layers
        raise TypeError("backbone does not expose a decoder-layer ModuleList")

    @property
    def n_layers(self) -> int:
        return len(self.decoder_layers())

    def bare_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The true floor: no hooks, no controls, no gradient graph."""

        self._validate_input_ids(input_ids)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            )
        return output.logits.detach()

    def injected_logits(
        self,
        input_ids: torch.Tensor,
        depth_controls: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Decode under per-depth control residuals, differentiable in the controls."""

        self._validate_input_ids(input_ids)
        layers = self.decoder_layers()
        for depth in depth_controls:
            if not 0 <= depth < len(layers):
                raise ValueError(f"injection depth {depth} outside 0..{len(layers) - 1}")

        handles = []

        def make_hook(controls: torch.Tensor):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    return (inject_control_residual(output[0], controls),) + output[1:]
                return inject_control_residual(output, controls)

            return hook

        try:
            for depth, controls in depth_controls.items():
                handles.append(layers[depth].register_forward_hook(make_hook(controls)))
            output = self.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            )
        finally:
            for handle in handles:
                handle.remove()
        return output.logits


class ToyMultiDepthBackbone(nn.Module):
    """Deterministic frozen CPU twin with the same multi-depth injection surface.

    Contract tests exercise injection, gradient flow, and floor identity without
    model downloads. This is not a performance baseline.
    """

    def __init__(
        self, vocab_size: int, width: int, n_layers: int = 4, *, seed: int = 0
    ) -> None:
        super().__init__()
        if vocab_size < 2 or width < 2 or n_layers < 1:
            raise ValueError("vocab_size, width, and n_layers must be nontrivial")
        self.vocab_size = vocab_size
        self.width = width
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, width)
            self.layers = nn.ModuleList(
                nn.GRU(width, width, batch_first=True) for _ in range(n_layers)
            )
            self.decoder = nn.Linear(width, vocab_size, bias=False)
            nn.init.normal_(self.embedding.weight, std=width**-0.5)
            nn.init.orthogonal_(self.decoder.weight)
        self.requires_grad_(False)
        self.eval()

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def _hidden_states(
        self,
        input_ids: torch.Tensor,
        depth_controls: dict[int, torch.Tensor] | None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a [batch, sequence] long tensor")
        hidden = self.embedding(input_ids)
        for depth, layer in enumerate(self.layers):
            residual, _ = layer(hidden)
            hidden = hidden + residual
            if depth_controls and depth in depth_controls:
                hidden = inject_control_residual(hidden, depth_controls[depth])
        return hidden

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._hidden_states(input_ids, None).detach()

    def bare_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(self._hidden_states(input_ids, None)).detach()

    def injected_logits(
        self,
        input_ids: torch.Tensor,
        depth_controls: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        for depth in depth_controls:
            if not 0 <= depth < self.n_layers:
                raise ValueError(f"injection depth {depth} outside 0..{self.n_layers - 1}")
        return self.decoder(self._hidden_states(input_ids, depth_controls))
