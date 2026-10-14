"""Feature Elicitor modules for Xetrieval document-side views."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


DOCUMENT_PERSPECTIVES = ("qa", "summary", "purpose")


class FeatureElicitor(nn.Module):
    """One-hidden-layer MLP P_t(z) that maps a document embedding to a view."""

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
        raise TypeError("Feature Elicitor checkpoint must contain a state dict.")

    state = {}
    for key, value in checkpoint.items():
        clean_key = key.removeprefix("module.").removeprefix("_orig_mod.")
        state[clean_key] = value
    return state


def load_feature_elicitor(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> FeatureElicitor:
    """Load one aspect-specific Feature Elicitor."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)
    if "net.0.weight" not in state:
        raise KeyError("Missing `net.0.weight` in Feature Elicitor checkpoint.")

    hidden_dim, input_dim = state["net.0.weight"].shape
    elicitor = FeatureElicitor(input_dim=input_dim, hidden_dim=hidden_dim)
    elicitor.load_state_dict(state)
    elicitor.to(device)
    elicitor.eval()
    return elicitor


def load_feature_elicitors(
    model_dir: str | Path,
    *,
    device: torch.device,
) -> Dict[str, FeatureElicitor]:
    """Load QA, Summary, and Purpose Feature Elicitors from a directory."""
    model_dir = Path(model_dir)
    elicitors: Dict[str, FeatureElicitor] = {}
    for aspect in DOCUMENT_PERSPECTIVES:
        checkpoint_path = model_dir / f"model_{aspect}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing Feature Elicitor: {checkpoint_path}")
        elicitors[aspect] = load_feature_elicitor(
            checkpoint_path,
            device=device,
        )
    return elicitors


@torch.no_grad()
def build_document_view_embeddings(
    doc_embeddings: torch.Tensor,
    elicitors: Dict[str, FeatureElicitor],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build V(d) = {z_d, P_qa(z_d), P_summary(z_d), P_purpose(z_d)}."""
    document_views = {"original": doc_embeddings.cpu()}

    for aspect in DOCUMENT_PERSPECTIVES:
        if aspect not in elicitors:
            raise KeyError(f"Missing `{aspect}` Feature Elicitor.")
        chunks: list[torch.Tensor] = []
        model = elicitors[aspect]
        for start in range(0, doc_embeddings.shape[0], batch_size):
            end = min(start + batch_size, doc_embeddings.shape[0])
            batch = doc_embeddings[start:end].to(device)
            view_embedding = model(batch)
            chunks.append(view_embedding.detach().cpu())
        document_views[aspect] = torch.cat(chunks, dim=0)

    return document_views
