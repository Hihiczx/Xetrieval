# Xetrieval: Explaining Dense Retrieval through Feature Elicitation

Xetrieval explains dense retrieval through sparse features shared by a query and
its document. **Feature Elicitors** express document semantics through
**elicited views**, while a **shared sparse dictionary** provides a common
feature vocabulary. The natural-language descriptions of shared features form
an explanation of the retriever's original query–document relevance score.

<p align="center">
  <img src="assets/overview.png" width="92%" alt="Xetrieval overview">
</p>

## Updates

- **[Released]** Inference code for feature-level retrieval explanation.
- **[Released]** shared TopK SAE checkpoint.
- **[Released]** Feature Elicitor checkpoints for `qa`, `summary`, and
  `purpose` document-side views.

## Method Overview

Given a query `q` and document `d`, Xetrieval:

1. Encodes `q` and `d` with an e5-style embedding model.
2. Applies three Feature Elicitors to the document embedding `z_d`,
   producing `qa`, `summary`, and `purpose` elicited views.
3. Applies the shared TopK SAE to the query embedding and all
   document-side views.
4. Computes shared active features between the query and each document view.
5. Aggregates those shared features across views and optionally attaches
   natural-language feature descriptions.

The returned `shared_features` field corresponds to `O(q,d)` in the paper.
The four document views include the original document embedding and the three
elicited views. A feature is active when its SAE coefficient is strictly positive
(the default activation threshold is zero).

## Installation

```bash
git clone https://github.com/Hihiczx/Xetrieval.git xetrieval
cd xetrieval
pip install -r requirements.txt
```

The default embedding model is [`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2),
loaded through `sentence-transformers`.

## Checkpoints

This release includes:

```text
checkpoints/
  sae_model.pt
  feature_elicitors/
    model_qa.pt
    model_summary.pt
    model_purpose.pt
```

If you use custom checkpoints, pass their paths through the CLI:

- `--sae_checkpoint`: TopK-SAE checkpoint.
- `--feature_elicitor_dir`: directory containing `model_qa.pt`,
  `model_summary.pt`, and `model_purpose.pt`.
- `--feature_descriptions_path`: optional text file where line `i` is the
  natural-language description for feature `i`.

## Input Format

Input is JSONL, with one query-document pair per line:

```json
{"query_id": "q1", "doc_id": "d1", "query": "why does ice float?", "doc": "Water expands as it freezes, lowering the density of ice."}
```

`query` and `doc` are required. `query_id` and `doc_id` are optional.

## Usage

Run Xetrieval with the released checkpoints:

```bash
python explain_retrieval.py \
  --input_jsonl examples/query_doc_pairs.jsonl \
  --output_jsonl outputs/explanations.jsonl \
  --embedding_model_name intfloat/e5-large-v2 \
  --feature_elicitor_dir checkpoints/feature_elicitors \
  --sae_checkpoint checkpoints/sae_model.pt \
  --device cuda \
  --batch_size 64
```

When `--feature_descriptions_path` is omitted, the output contains feature ids
only. After the description table is released, add:

```bash
--feature_descriptions_path path/to/feature_descriptions.txt
```

## Output Format

Each output line is a JSON object. The feature IDs and descriptions below are
illustrative:

```json
{
  "query_id": "q1",
  "doc_id": "d1",
  "shared_features": [12, 98],
  "per_view_shared_features": {
    "original": [12],
    "qa": [98],
    "summary": [],
    "purpose": [12]
  },
  "feature_descriptions": [
    {"feature_id": 12, "description": "density and phase-change reasoning"},
    {"feature_id": 98, "description": "water-related physical explanation"}
  ]
}
```

`shared_features` is the union of the four `per_view_shared_features` sets.

## Citation

If you find Xetrieval useful for your research, please cite:

```bibtex
@article{cai2026xetrieval,
  title={Xetrieval: Mechanistically Explaining Dense Retrieval},
  author={Cai, Zhixin and Bai, Jun and Liu, Yang and Li, Jiaqi and Zhang, Yichi and Li, Taichuan and Chen, Zhuofan and Jia, Zixia and Zheng, Zilong and Rong, Wenge},
  journal={arXiv preprint arXiv:2605.29507},
  year={2026}
}
```
