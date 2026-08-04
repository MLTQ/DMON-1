"""Frozen language-backbone contracts and a deterministic CPU reference backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class FrozenLanguageBackbone(Protocol):
    """Minimal replaceable language-organ surface consumed by DMON-L0."""

    width: int
    vocab_size: int

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return frozen contextual features shaped `[batch, sequence, width]`."""

    def controlled_logits(
        self,
        input_ids: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return token logits, with `None` denoting the exact baseline path."""

    def controlled_logits_from_features(
        self,
        features: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode already-computed frozen features under optional controls."""

    def controlled_logits_sequence_from_features(
        self,
        features: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode features under one separate control bank per sequence position."""


def _apply_control_residual(
    features: torch.Tensor,
    controls: torch.Tensor | None,
    width: int,
) -> torch.Tensor:
    """Attend from frozen features to global or position-specific control banks."""

    if controls is None:
        return features
    controls = controls.to(device=features.device, dtype=features.dtype)
    scale = width**-0.5
    if controls.ndim == 3:
        if controls.shape[0] != features.shape[0] or controls.shape[2] != width:
            raise ValueError("controls must have shape [batch, tokens, width]")
        scores = torch.einsum("bsw,btw->bst", features, controls) * scale
        attention = torch.softmax(scores.float(), dim=-1).to(features.dtype)
        residual = torch.einsum("bst,btw->bsw", attention, controls)
    elif controls.ndim == 4:
        expected = (features.shape[0], features.shape[1], width)
        if (
            controls.shape[0] != expected[0]
            or controls.shape[1] != expected[1]
            or controls.shape[2] < 1
            or controls.shape[3] != expected[2]
        ):
            raise ValueError(
                "sequence controls must have shape [batch, sequence, tokens, width]"
            )
        scores = torch.einsum("bsw,bstw->bst", features, controls) * scale
        attention = torch.softmax(scores.float(), dim=-1).to(features.dtype)
        residual = torch.einsum("bst,bstw->bsw", attention, controls)
    else:
        raise ValueError("controls must be a global or position-specific token bank")
    return features + residual


class ToyFrozenLanguageBackbone(nn.Module):
    """Small deterministic backbone for contract tests and CPU development.

    This is deliberately not a performance baseline. It proves that frozen language
    computation can expose contextual features and remain differentiable with respect
    to continuous SOL2 controls while every backbone parameter stays fixed.
    """

    def __init__(self, vocab_size: int, width: int, *, seed: int = 0) -> None:
        super().__init__()
        if vocab_size < 2 or width < 2:
            raise ValueError("vocab_size and width must both be at least two")
        self.vocab_size = vocab_size
        self.width = width

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, width)
            self.recurrent = nn.GRU(width, width, batch_first=True)
            self.decoder = nn.Linear(width, vocab_size, bias=False)
            nn.init.normal_(self.embedding.weight, std=width**-0.5)
            nn.init.orthogonal_(self.decoder.weight)
        self.requires_grad_(False)
        self.eval()

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        with torch.no_grad():
            features, _ = self.recurrent(self.embedding(input_ids))
        return features.detach()

    def controlled_logits(
        self,
        input_ids: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.controlled_logits_from_features(self.encode(input_ids), controls)

    def controlled_logits_from_features(
        self,
        features: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.width:
            raise ValueError("features must have shape [batch, sequence, width]")
        return self.decoder(_apply_control_residual(features, controls, self.width))

    def controlled_logits_sequence_from_features(
        self,
        features: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.controlled_logits_from_features(features, controls)

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")


class HuggingFaceFrozenBackbone(nn.Module):
    """Adapter for decoder-only Hugging Face models with an exposed output head."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        config = getattr(model, "config", None)
        width = getattr(config, "hidden_size", getattr(config, "n_embd", None))
        vocab_size = getattr(config, "vocab_size", None)
        output_head = getattr(model, "get_output_embeddings", lambda: None)()
        if width is None or vocab_size is None or output_head is None:
            raise TypeError(
                "model must expose hidden width, vocabulary size, and output embeddings"
            )
        self.model = model
        self.width = int(width)
        self.vocab_size = int(vocab_size)
        self.model.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | str = "auto",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> "HuggingFaceFrozenBackbone":
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError(
                "install transformers to load a pretrained language backbone"
            ) from error
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model.to(device)
        return cls(model)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(output, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("causal model did not return hidden states")
        return hidden_states[-1].detach()

    def controlled_logits(
        self,
        input_ids: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.controlled_logits_from_features(self.encode(input_ids), controls)

    def controlled_logits_from_features(
        self,
        features: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.width:
            raise ValueError("features must have shape [batch, sequence, width]")
        controlled = _apply_control_residual(features, controls, self.width)
        return self.model.get_output_embeddings()(controlled)

    def controlled_logits_sequence_from_features(
        self,
        features: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.controlled_logits_from_features(features, controls)

    @torch.no_grad()
    def native_logit_error(self, input_ids: torch.Tensor) -> float:
        """Measure whether the generic exposed head reproduces native model logits."""

        self._validate_input_ids(input_ids)
        native = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        decoded = self.controlled_logits_from_features(native.hidden_states[-1])
        return float((native.logits - decoded).abs().max())

    @staticmethod
    def _validate_input_ids(input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have non-empty [batch, sequence] shape")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
