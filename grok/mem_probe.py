"""One-shot CUDA memory probe for config selection on Aine."""

from __future__ import annotations

import torch

from .config import TrainConfig
from .corpus import CharCorpus
from .model import StreamingCreature


def peak(device_index: int = 1, **kwargs) -> None:
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    corpus = CharCorpus.from_shakespeare("data/tinyshakespeare")
    cfg = TrainConfig(**kwargs, use_attention=True)
    model = StreamingCreature(cfg, corpus.vocab_size).to(device)
    state = model.initial_state(cfg.batch_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    corpus.reset_cursors(cfg.batch_size, seed=0)
    window: torch.Tensor | None = None
    for _ in range(cfg.truncate_every):
        x, y = corpus.next_batch(cfg.batch_size, device)
        logits, state = model.step(x, state)
        loss = torch.nn.functional.cross_entropy(logits, y)
        window = loss if window is None else window + loss
    assert window is not None
    (window / cfg.truncate_every).backward()
    opt.step()
    peak_gb = torch.cuda.max_memory_allocated(device_index) / 1e9
    print(
        f"cells={cfg.n_cells} h={cfg.hidden} B={cfg.batch_size} "
        f"T={cfg.truncate_every} spt={cfg.steps_per_token} "
        f"params={model.count_parameters():,} peak_GB={peak_gb:.2f}"
    )


def main() -> None:
    configs = [
        dict(n_cells=64, hidden=64, batch_size=4, n_dendrites=8, n_input=8, n_output=8, n_mirror=16, truncate_every=32, steps_per_token=2),
        dict(n_cells=64, hidden=96, batch_size=8, n_dendrites=8, n_input=8, n_output=8, n_mirror=16, truncate_every=32, steps_per_token=3),
        dict(n_cells=96, hidden=96, batch_size=8, n_dendrites=10, n_input=12, n_output=12, n_mirror=24, truncate_every=32, steps_per_token=3),
        dict(n_cells=128, hidden=128, batch_size=4, n_dendrites=12, n_input=16, n_output=16, n_mirror=32, truncate_every=24, steps_per_token=3),
        dict(n_cells=128, hidden=128, batch_size=8, n_dendrites=12, n_input=16, n_output=16, n_mirror=32, truncate_every=16, steps_per_token=2),
        dict(n_cells=128, hidden=128, batch_size=16, n_dendrites=12, n_input=16, n_output=16, n_mirror=32, truncate_every=8, steps_per_token=2),
    ]
    for kwargs in configs:
        try:
            peak(1, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL cells={kwargs['n_cells']} B={kwargs['batch_size']}: {type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
