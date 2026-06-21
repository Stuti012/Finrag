"""Component ablation study for FinTAG-RAG (CLI, reproducible thesis artifact).

Runs the rule-based (no-LLM) pipeline over a FinQA split across the full 2x2
grid of the two retrieval-stage components introduced beyond the original
hybrid-retrieval baseline:

    - QE: ontology-based query expansion              (§3.1, query_expansion.py)
    - TF: temporal-compatibility filtering             (§3.3, temporal_filter.py)

    Baseline   : QE off, TF off  — plain hybrid retrieval + reranking (§3.2)
    +QE        : QE on,  TF off
    +TF        : QE off, TF on
    Full model : QE on,  TF on

Usage:
    python scripts/run_ablation.py --split finqa_data/test.json --out results/ablation.json
    python scripts/run_ablation.py --split finqa_data/test.json --max-examples 200 --tau 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.data.finqa_loader import load_finqa_split
from src.evaluation.metrics import FinQAEvaluator
from src.pipeline import FinancialQAPipeline

CONFIGS = [
    ("baseline", False, False),
    ("+QE", True, False),
    ("+TF", False, True),
    ("full_model", True, True),
]


def run_config(
    examples: List[Any],
    query_expansion: bool,
    temporal_filter: bool,
    tau: float,
    evaluator: FinQAEvaluator,
) -> Dict[str, Any]:
    pipeline = FinancialQAPipeline(
        load_llm=False,
        query_expansion_enabled=query_expansion,
        temporal_filter_enabled=temporal_filter,
        temporal_filter_tau=tau,
    )
    t0 = time.time()
    results = pipeline.batch_answer(examples, verbose=False)
    report = evaluator.evaluate(results, examples)

    discarded = sum(
        r.get("retrieval", {}).get("temporal_filter", {}).get("discarded", 0) for r in results
    )
    gated = sum(
        1 for r in results if r.get("retrieval", {}).get("temporal_filter", {}).get("applied")
    )

    return {
        "numerical_accuracy": report["overall"]["accuracy"],
        "context_f1": report["context_filtering"].get("mean_f1", 0.0),
        "context_precision": report["context_filtering"].get("mean_precision", 0.0),
        "context_recall": report["context_filtering"].get("mean_recall", 0.0),
        "causality_detection_rate": report["causality_detection"].get("detection_rate", 0.0),
        "mean_temporal_score": report["temporal_reasoning"].get("mean_temporal_score", 0.0),
        "program_generation_rate": report["program_induction"].get("program_generation_rate", 0.0),
        "execution_success_rate": report["program_induction"].get("execution_success_rate", 0.0),
        "temporal_passages_discarded": discarded,
        "temporal_gate_applied_count": gated,
        "elapsed_seconds": round(time.time() - t0, 1),
        "num_examples": len(examples),
    }


def print_table(grid: Dict[str, Dict[str, Any]]) -> None:
    metrics = [
        ("Numerical accuracy", "numerical_accuracy", True),
        ("Context F1", "context_f1", True),
        ("Context precision", "context_precision", True),
        ("Context recall", "context_recall", True),
        ("Causality detect rate", "causality_detection_rate", True),
        ("Mean temporal score", "mean_temporal_score", False),
        ("Program gen rate", "program_generation_rate", True),
        ("Execution success", "execution_success_rate", True),
    ]
    names = list(grid.keys())
    print("\n  ── FinTAG-RAG Component Ablation ─────────────────────────────────────")
    header = f'  {"Metric":<24}' + "".join(f"{n:>14}" for n in names)
    print(header)
    baseline = grid[names[0]]
    for label, key, is_pct in metrics:
        row = f"  {label:<24}"
        for n in names:
            v = grid[n][key]
            row += f"{v:>14.1%}" if is_pct else f"{v:>14.3f}"
        delta = grid[names[-1]][key] - baseline[key]
        row += f"   (Δ full vs base: {delta:+.1%})" if is_pct else f"   (Δ full vs base: {delta:+.3f})"
        print(row)
    print(f'  {"Elapsed (s)":<24}' + "".join(f"{grid[n]['elapsed_seconds']:>14.0f}" for n in names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="Path to a FinQA split JSON file.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tau", type=float, default=0.5, help="Temporal-filter threshold τ.")
    parser.add_argument("--out", default=None, help="Optional path to write the results JSON.")
    args = parser.parse_args()

    examples = load_finqa_split(args.split, max_examples=args.max_examples)
    print(f"Loaded {len(examples)} examples from {args.split}")

    evaluator = FinQAEvaluator(tolerance=0.01)
    grid: Dict[str, Dict[str, Any]] = {}
    for label, qe, tf in CONFIGS:
        print(f"Running config '{label}' (QE={qe}, TF={tf})...")
        grid[label] = run_config(examples, qe, tf, args.tau, evaluator)

    print_table(grid)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(grid, indent=2))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
