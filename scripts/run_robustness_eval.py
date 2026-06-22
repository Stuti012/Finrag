"""Robustness/perturbation evaluation (CLI, reproducible thesis artifact).

Two perturbation suites, see src/evaluation/robustness.py for the rationale:

  distractor  — inject an off-period numeric sentence per example and measure
                accuracy with the temporal-compatibility gate (§3.3) OFF vs ON.
                This is the headline robustness result: it tests whether the
                gate protects accuracy under a *realistic* contamination
                attempt, not just whether it discards passages in isolation.

  paraphrase  — substitute a financial term in the question with an ontology
                synonym and measure accuracy with query expansion (§3.1)
                OFF vs ON, testing whether QE recovers the accuracy lost to
                lexical mismatch between question and filing vocabulary.

Usage:
    python scripts/run_robustness_eval.py --split finqa_data/test.json \
        --pool finqa_data/dev.json --suite distractor --max-examples 150
    python scripts/run_robustness_eval.py --split finqa_data/test.json \
        --suite paraphrase --max-examples 150
    python scripts/run_robustness_eval.py --split finqa_data/test.json \
        --pool finqa_data/dev.json --suite all --out results/robustness.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.finqa_loader import load_finqa_split
from src.evaluation.robustness import (
    DistractorInjector,
    SynonymParaphraser,
    summarize_flip_rate,
)
from src.pipeline import FinancialQAPipeline
from src.utils.financial_utils import answers_match


def run_distractor_suite(
    examples: List[Any], pool_examples: List[Any], tau: float
) -> Dict[str, Any]:
    injector = DistractorInjector(pool_examples)
    pipeline_off = FinancialQAPipeline(load_llm=False, temporal_filter_enabled=False)
    pipeline_on = FinancialQAPipeline(load_llm=False, temporal_filter_enabled=True, temporal_filter_tau=tau)

    records_off: List[Dict[str, Any]] = []
    records_on: List[Dict[str, Any]] = []
    discarded_count = 0
    skipped = 0

    for ex in examples:
        perturbed, distractor, year = injector.inject(ex)
        if distractor is None:
            skipped += 1
            continue

        base_res = pipeline_on.answer(ex)
        base_correct = answers_match(base_res.get("predicted_answer", ""), ex.answer)

        off_res = pipeline_off.answer(perturbed)
        on_res = pipeline_on.answer(perturbed)

        records_off.append({
            "base_correct": base_correct,
            "perturbed_correct": answers_match(off_res.get("predicted_answer", ""), ex.answer),
        })
        records_on.append({
            "base_correct": base_correct,
            "perturbed_correct": answers_match(on_res.get("predicted_answer", ""), ex.answer),
        })

        tf_diag = on_res.get("retrieval", {}).get("temporal_filter", {})
        if tf_diag.get("discarded_periods") and year in tf_diag["discarded_periods"]:
            discarded_count += 1

    return {
        "filter_off": summarize_flip_rate(records_off),
        "filter_on": summarize_flip_rate(records_on),
        "distractor_discard_rate": discarded_count / len(records_on) if records_on else 0.0,
        "skipped_no_distractor": skipped,
        "evaluated": len(records_on),
    }


def run_paraphrase_suite(examples: List[Any]) -> Dict[str, Any]:
    paraphraser = SynonymParaphraser()
    pipeline_off = FinancialQAPipeline(load_llm=False, query_expansion_enabled=False)
    pipeline_on = FinancialQAPipeline(load_llm=False, query_expansion_enabled=True)

    records_off: List[Dict[str, Any]] = []
    records_on: List[Dict[str, Any]] = []
    skipped = 0

    for ex in examples:
        paraphrased_q = paraphraser.paraphrase(ex.question)
        if paraphrased_q is None:
            skipped += 1
            continue
        from dataclasses import replace as dc_replace
        perturbed = dc_replace(ex, question=paraphrased_q)

        base_res = pipeline_on.answer(ex)
        base_correct = answers_match(base_res.get("predicted_answer", ""), ex.answer)

        off_res = pipeline_off.answer(perturbed)
        on_res = pipeline_on.answer(perturbed)

        records_off.append({
            "base_correct": base_correct,
            "perturbed_correct": answers_match(off_res.get("predicted_answer", ""), ex.answer),
        })
        records_on.append({
            "base_correct": base_correct,
            "perturbed_correct": answers_match(on_res.get("predicted_answer", ""), ex.answer),
        })

    return {
        "query_expansion_off": summarize_flip_rate(records_off),
        "query_expansion_on": summarize_flip_rate(records_on),
        "skipped_no_ontology_term": skipped,
        "evaluated": len(records_on),
    }


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    print(f"\n  ── {label} ──────────────────────────────────────────────")
    for key, val in summary.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k:<20}{v:.1%}" if isinstance(v, float) and k != "n" else f"    {k:<20}{v}")
        else:
            print(f"  {key:<28}{val}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="Path to the FinQA split to perturb.")
    parser.add_argument("--pool", default=None, help="Path to a distinct split to scavenge distractors from (required for --suite distractor/all).")
    parser.add_argument("--suite", choices=["distractor", "paraphrase", "all"], default="all")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    examples = load_finqa_split(args.split, max_examples=args.max_examples)
    print(f"Loaded {len(examples)} examples from {args.split}")

    results: Dict[str, Any] = {}

    if args.suite in ("distractor", "all"):
        if not args.pool:
            raise SystemExit("--pool is required for the distractor suite")
        pool_examples = load_finqa_split(args.pool)
        print(f"Loaded {len(pool_examples)} pool examples from {args.pool}")
        print("Running distractor-injection suite...")
        results["distractor"] = run_distractor_suite(examples, pool_examples, args.tau)
        print_summary("Distractor-Injection Robustness", results["distractor"])

    if args.suite in ("paraphrase", "all"):
        print("Running ontology-paraphrase suite...")
        results["paraphrase"] = run_paraphrase_suite(examples)
        print_summary("Ontology-Paraphrase Robustness", results["paraphrase"])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
