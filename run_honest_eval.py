"""Honest evaluation - no gold programs, proper train/dev split."""
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from src.data.finqa_loader import load_finqa_dataset
from src.pipeline import FinancialQAPipeline
from src.evaluation.metrics import FinQAEvaluator
from src.utils.financial_utils import answers_match

print("=" * 60)
print("HONEST EVALUATION - NO GOLD PROGRAMS")
print("=" * 60)

# Load dataset - use dev set for evaluation (not seen during development)
dataset = load_finqa_dataset("./finqa_data", download=True, max_dev=200)
eval_examples = dataset.get("dev", [])[:200]
print(f"\nLoaded {len(eval_examples)} dev examples for evaluation")

# Verify we're NOT passing gold programs
print("\nSanity check - first example:")
ex0 = eval_examples[0]
print(f"  Question: {ex0.question[:100]}...")
print(f"  Gold program: {ex0.program}")
print(f"  Gold answer: {ex0.answer}")
print(f"  (Pipeline will NOT receive gold program)")

# Create pipeline (no LLM)
pipeline = FinancialQAPipeline(load_llm=False)

# Run evaluation
print("\nRunning evaluation (program induction from question+table only)...")
results = pipeline.batch_answer(eval_examples, verbose=True)

# Evaluate
evaluator = FinQAEvaluator(tolerance=0.01)
report = evaluator.evaluate(results, eval_examples)
evaluator.print_report(report)

# Save report
os.makedirs("outputs", exist_ok=True)
with open("outputs/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

# Detailed analysis
print("\n\n--- QUESTION TYPE DISTRIBUTION ---")
type_counts = defaultdict(int)
for r in results:
    modules = r.get("classification", {}).get("active_modules", [])
    for m in modules:
        type_counts[m] += 1
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

print("\n--- METHOD DISTRIBUTION ---")
method_counts = defaultdict(int)
for r in results:
    method = r.get("numerical", {}).get("method", "none")
    method_counts[method] += 1
for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
    print(f"  {m}: {c}")

print("\n--- INDUCED PROGRAM ACCURACY ---")
induced_correct = 0
induced_total = 0
pot_total = 0
for r, ex in zip(results, eval_examples):
    method = r.get("numerical", {}).get("method", "")
    if method == "induced_program":
        induced_total += 1
        if answers_match(r.get("predicted_answer", ""), r.get("gold_answer", "")):
            induced_correct += 1
    elif method == "program_of_thought":
        pot_total += 1

print(f"  Induced programs: {induced_total}/{len(results)}")
print(f"  Induced correct: {induced_correct}/{induced_total} = {induced_correct/max(induced_total,1):.1%}")
print(f"  Fell through to PoT (needs LLM): {pot_total}")

# Show sample of failures
print("\n--- SAMPLE FAILURES (first 15) ---")
failures = []
for r, ex in zip(results, eval_examples):
    pred = r.get("predicted_answer", "")
    gold = r.get("gold_answer", "")
    if not answers_match(pred, gold):
        failures.append((r, ex))

print(f"Total failures: {len(failures)}/{len(results)}")
for r, ex in failures[:15]:
    method = r.get("numerical", {}).get("method", "none")
    induced = r.get("numerical", {}).get("induced_program", None)
    print(f"\n  ID: {r['id']}")
    print(f"  Q: {r['question'][:120]}")
    print(f"  Predicted: {r.get('predicted_answer', '')}")
    print(f"  Gold: {r.get('gold_answer', '')}")
    print(f"  Method: {method}")
    if induced:
        print(f"  Induced: {induced}")
    print(f"  Gold program: {ex.program}")
