#!/usr/bin/env python3
"""Explain dense retrieval decisions with Xetrieval shared sparse features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

from embedding_encoder import EmbeddingEncoder
from mechanistic_explainer import (
    active_support,
    encode_sparse_codes_in_batches,
    load_mechanistic_explainer,
)
from reasoning_internalizer import (
    REASONING_ASPECTS,
    build_document_view_embeddings,
    load_reasoning_internalizers,
)


DOCUMENT_VIEW_NAMES = ("original", *REASONING_ASPECTS)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_name)


def read_query_doc_pairs(input_jsonl: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(input_jsonl).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "query" not in record or "doc" not in record:
                raise ValueError(
                    f"Line {line_number} must contain both `query` and `doc` fields."
                )
            records.append(record)
    if not records:
        raise ValueError(f"No query-document pairs found in {input_jsonl}.")
    return records


def load_feature_hypotheses(path: str | Path | None) -> list[str] | None:
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def feature_explanations(
    shared_features: Iterable[int],
    feature_hypotheses: list[str],
) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    for feature_id in shared_features:
        hypothesis = (
            feature_hypotheses[feature_id]
            if feature_id < len(feature_hypotheses)
            else ""
        )
        explanations.append(
            {
                "feature_id": int(feature_id),
                "hypothesis": hypothesis,
            }
        )
    return explanations


def compute_document_view_supports(
    document_views: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    mechanistic_explainer,
    activation_threshold: float,
) -> dict[str, list[set[int]]]:
    view_supports: dict[str, list[set[int]]] = {}
    for view_name in DOCUMENT_VIEW_NAMES:
        sparse_codes = encode_sparse_codes_in_batches(
            document_views[view_name],
            mechanistic_explainer,
            batch_size=batch_size,
            device=device,
        )
        view_supports[view_name] = active_support(
            sparse_codes,
            activation_threshold=activation_threshold,
        )
    return view_supports


def build_explanation_records(
    records: list[dict[str, Any]],
    *,
    query_supports: list[set[int]],
    document_view_supports: dict[str, list[set[int]]],
    feature_hypotheses: list[str] | None,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        per_view_shared_features: dict[str, list[int]] = {}
        shared_feature_set: set[int] = set()

        for view_name in DOCUMENT_VIEW_NAMES:
            shared = query_supports[index] & document_view_supports[view_name][index]
            sorted_shared = sorted(int(feature_id) for feature_id in shared)
            per_view_shared_features[view_name] = sorted_shared
            shared_feature_set.update(sorted_shared)

        shared_features = sorted(int(feature_id) for feature_id in shared_feature_set)
        output_record: dict[str, Any] = {
            "query_id": record.get("query_id"),
            "doc_id": record.get("doc_id"),
            "shared_features": shared_features,
            "per_view_shared_features": per_view_shared_features,
        }
        if feature_hypotheses is not None:
            output_record["feature_explanations"] = feature_explanations(
                shared_features,
                feature_hypotheses,
            )
        outputs.append(output_record)

    return outputs


def write_jsonl(records: Iterable[dict[str, Any]], output_jsonl: str | Path) -> None:
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain query-document pairs with Xetrieval shared features."
    )
    parser.add_argument("--input_jsonl", required=True, help="Input query-doc pair JSONL.")
    parser.add_argument("--output_jsonl", required=True, help="Output explanation JSONL.")
    parser.add_argument(
        "--embedding_model_name",
        default="intfloat/e5-large-v2",
        help="SentenceTransformer embedding model name or local path.",
    )
    parser.add_argument(
        "--reasoning_internalizer_dir",
        required=True,
        help="Directory containing model_qa.pt, model_summary.pt, and model_purpose.pt.",
    )
    parser.add_argument(
        "--mechanistic_explainer_checkpoint",
        required=True,
        help="TopK-SAE mechanistic explainer checkpoint.",
    )
    parser.add_argument(
        "--feature_hypotheses_path",
        default=None,
        help="Optional txt file where line i is the hypothesis for feature i.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--activation_threshold",
        type=float,
        default=0.0,
        help="Threshold tau used to binarize sparse feature activations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    records = read_query_doc_pairs(args.input_jsonl)
    feature_hypotheses = load_feature_hypotheses(args.feature_hypotheses_path)
    queries = [str(record["query"]) for record in records]
    docs = [str(record["doc"]) for record in records]

    print(f"Loaded {len(records)} query-document pairs.")
    print(f"Loading embedding encoder: {args.embedding_model_name}")
    embedding_encoder = EmbeddingEncoder(
        args.embedding_model_name,
        device=device,
        batch_size=args.batch_size,
    )
    query_embeddings = embedding_encoder.encode_queries(queries)
    doc_embeddings = embedding_encoder.encode_documents(docs)

    print("Loading reasoning internalizers.")
    reasoning_internalizers = load_reasoning_internalizers(
        args.reasoning_internalizer_dir,
        device=device,
    )
    print("Building document-side views.")
    document_views = build_document_view_embeddings(
        doc_embeddings,
        reasoning_internalizers,
        batch_size=args.batch_size,
        device=device,
    )

    print("Loading mechanistic explainer.")
    mechanistic_explainer = load_mechanistic_explainer(
        args.mechanistic_explainer_checkpoint,
        device=device,
    )
    if query_embeddings.shape[1] != mechanistic_explainer.activation_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"encoder produced {query_embeddings.shape[1]}, "
            f"but mechanistic explainer expects {mechanistic_explainer.activation_dim}."
        )

    if feature_hypotheses is not None and len(feature_hypotheses) < mechanistic_explainer.dict_size:
        print(
            "Warning: feature hypothesis file has fewer lines than the SAE dictionary size.",
            file=sys.stderr,
        )

    print("Encoding query sparse codes.")
    query_sparse_codes = encode_sparse_codes_in_batches(
        query_embeddings,
        mechanistic_explainer,
        batch_size=args.batch_size,
        device=device,
    )
    query_supports = active_support(
        query_sparse_codes,
        activation_threshold=args.activation_threshold,
    )

    print("Encoding document-view sparse codes.")
    document_view_supports = compute_document_view_supports(
        document_views,
        batch_size=args.batch_size,
        device=device,
        mechanistic_explainer=mechanistic_explainer,
        activation_threshold=args.activation_threshold,
    )

    output_records = build_explanation_records(
        records,
        query_supports=query_supports,
        document_view_supports=document_view_supports,
        feature_hypotheses=feature_hypotheses,
    )
    write_jsonl(output_records, args.output_jsonl)
    print(f"Wrote explanations to {args.output_jsonl}")


if __name__ == "__main__":
    main()
