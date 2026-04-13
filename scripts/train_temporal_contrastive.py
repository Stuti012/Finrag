"""Train temporal-aware contrastive retriever pairs.

Builds positive/hard-negative pairs to fine-tune sentence embeddings for
financial temporal retrieval.
"""

import argparse
import json
import re
from typing import List, Dict

from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def extract_years(text: str) -> List[str]:
    return YEAR_RE.findall(text or "")


def build_pairs(data: List[Dict]) -> List[InputExample]:
    pairs: List[InputExample] = []
    for ex in data:
        q = ex.get("question", "")
        pos = ex.get("positive_passage", "")
        negs = ex.get("negative_passages", [])
        if not q or not pos:
            continue

        q_years = set(extract_years(q))
        pos_years = set(extract_years(pos))
        if q_years and pos_years and not (q_years & pos_years):
            continue

        pairs.append(InputExample(texts=[q, pos], label=1.0))

        for neg in negs[:3]:
            neg_years = set(extract_years(neg))
            # Hard negative: entity overlap but wrong time period.
            if q_years and neg_years and (q_years & neg_years):
                continue
            pairs.append(InputExample(texts=[q, neg], label=0.0))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_examples = build_pairs(data)
    model = SentenceTransformer(args.model_name)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=max(1, len(train_loader) // 10),
        output_path=args.output_dir,
    )


if __name__ == "__main__":
    main()
