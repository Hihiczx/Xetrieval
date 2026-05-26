"""Reasoning internalizer modules for Xetrieval document-side views."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


REASONING_ASPECTS = ("qa", "summary", "purpose")


class ReasoningInternalizer(nn.Module):
    """One-hidden-layer MLP R_t(z) that maps a document embedding to a view."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(embedding), p=2, dim=-1)


def _extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Reasoning internalizer checkpoint must contain a state dict.")

    state = {}
    for key, value in checkpoint.items():
        clean_key = key.removeprefix("module.").removeprefix("_orig_mod.")
        state[clean_key] = value
    return state


def load_reasoning_internalizer(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> ReasoningInternalizer:
    """Load one aspect-specific reasoning internalizer."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)
    if "net.0.weight" not in state:
        raise KeyError("Missing `net.0.weight` in reasoning internalizer checkpoint.")

    hidden_dim, input_dim = state["net.0.weight"].shape
    internalizer = ReasoningInternalizer(input_dim=input_dim, hidden_dim=hidden_dim)
    internalizer.load_state_dict(state)
    internalizer.to(device)
    internalizer.eval()
    return internalizer


def load_reasoning_internalizers(
    model_dir: str | Path,
    *,
    device: torch.device,
) -> Dict[str, ReasoningInternalizer]:
    """Load QA, Summary, and Purpose reasoning internalizers from a directory."""
    model_dir = Path(model_dir)
    internalizers: Dict[str, ReasoningInternalizer] = {}
    for aspect in REASONING_ASPECTS:
        checkpoint_path = model_dir / f"model_{aspect}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing reasoning internalizer: {checkpoint_path}")
        internalizers[aspect] = load_reasoning_internalizer(
            checkpoint_path,
            device=device,
        )
    return internalizers


@torch.no_grad()
def build_document_view_embeddings(
    doc_embeddings: torch.Tensor,
    internalizers: Dict[str, ReasoningInternalizer],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build V(d) = {z_d, R_qa(z_d), R_summary(z_d), R_purpose(z_d)}."""
    document_views = {"original": doc_embeddings.cpu()}

    for aspect in REASONING_ASPECTS:
        if aspect not in internalizers:
            raise KeyError(f"Missing `{aspect}` reasoning internalizer.")
        chunks: list[torch.Tensor] = []
        model = internalizers[aspect]
        for start in range(0, doc_embeddings.shape[0], batch_size):
            end = min(start + batch_size, doc_embeddings.shape[0])
            batch = doc_embeddings[start:end].to(device)
            view_embedding = model(batch)
            chunks.append(view_embedding.detach().cpu())
        document_views[aspect] = torch.cat(chunks, dim=0)

    return document_views
