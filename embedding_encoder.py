"""Sentence embedding wrapper used by Xetrieval inference."""

from __future__ import annotations

from typing import Sequence

import torch


QUERY_PREFIX = "query: "
DOC_PREFIX = "passage: "


class EmbeddingEncoder:
    """Encode queries and documents with an e5-style SentenceTransformer."""

    def __init__(
        self,
        model_name: str,
        *,
        device: torch.device,
        batch_size: int,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=str(device))

    def encode_queries(self, queries: Sequence[str]) -> torch.Tensor:
        return self._encode([f"{QUERY_PREFIX}{query}" for query in queries])

    def encode_documents(self, documents: Sequence[str]) -> torch.Tensor:
        return self._encode([f"{DOC_PREFIX}{doc}" for doc in documents])

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        return embeddings.detach().cpu().float()
