"""TopK-SAE mechanistic explainer used by Xetrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKMechanisticExplainer(nn.Module):
    """TopK sparse autoencoder for decomposing embeddings into sparse codes."""

    def __init__(self, activation_dim: int, dict_size: int, k: int) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"TopK value must be positive, got {k}.")
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode_sparse_code(
        self,
        embedding: torch.Tensor,
        *,
        return_topk: bool = False,
        use_threshold: bool = False,
    ):
        """Encode dense embeddings into TopK sparse feature activations."""
        post_relu = F.relu(self.encoder(embedding - self.b_dec))
        if use_threshold:
            sparse_code = post_relu * (post_relu > self.threshold)
            if return_topk:
                topk = post_relu.topk(self.k.item(), sorted=False, dim=-1)
                return sparse_code, topk.values, topk.indices, post_relu
            return sparse_code

        topk = post_relu.topk(self.k.item(), sorted=False, dim=-1)
        sparse_code = torch.zeros_like(post_relu)
        sparse_code.scatter_(dim=-1, index=topk.indices, src=topk.values)
        if return_topk:
            return sparse_code, topk.values, topk.indices, post_relu
        return sparse_code

    def decode(self, sparse_code: torch.Tensor) -> torch.Tensor:
        return self.decoder(sparse_code) + self.b_dec

    def forward(self, embedding: torch.Tensor, *, return_sparse_code: bool = False):
        sparse_code = self.encode_sparse_code(embedding)
        reconstruction = self.decode(sparse_code)
        if return_sparse_code:
            return reconstruction, sparse_code
        return reconstruction


def active_support(
    sparse_code: torch.Tensor,
    *,
    activation_threshold: float = 0.0,
) -> List[set[int]]:
    """Return active sparse feature ids for each row of a sparse code tensor."""
    supports: List[set[int]] = []
    for row in sparse_code:
        feature_ids = torch.nonzero(row > activation_threshold, as_tuple=False)
        supports.append({int(i) for i in feature_ids.flatten().tolist()})
    return supports


@torch.no_grad()
def encode_sparse_codes_in_batches(
    embeddings: torch.Tensor,
    explainer: TopKMechanisticExplainer,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode a matrix of embeddings into sparse codes without keeping GPU outputs."""
    chunks: list[torch.Tensor] = []
    for start in range(0, embeddings.shape[0], batch_size):
        end = min(start + batch_size, embeddings.shape[0])
        batch = embeddings[start:end].to(device)
        sparse_code = explainer.encode_sparse_code(batch)
        chunks.append(sparse_code.detach().cpu())
    if not chunks:
        raise ValueError("Cannot encode an empty embedding matrix.")
    return torch.cat(chunks, dim=0)


def _extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Mechanistic explainer checkpoint must contain a state dict.")

    state = {}
    for key, value in checkpoint.items():
        clean_key = key.removeprefix("module.").removeprefix("_orig_mod.")
        state[clean_key] = value
    return state


def _read_topk_value(state: dict[str, torch.Tensor], default: int = 64) -> int:
    value = state.get("k")
    if value is None:
        return default
    try:
        return int(value.item())
    except AttributeError:
        return int(value)


def load_mechanistic_explainer(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> TopKMechanisticExplainer:
    """Load the TopK-SAE mechanistic explainer from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)
    if "encoder.weight" not in state:
        raise KeyError("Missing `encoder.weight` in mechanistic explainer checkpoint.")

    dict_size, activation_dim = state["encoder.weight"].shape
    explainer = TopKMechanisticExplainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        k=_read_topk_value(state),
    )
    explainer.load_state_dict(state)
    explainer.to(device)
    explainer.eval()
    return explainer
