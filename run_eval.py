"""Quick evaluation script to test accuracy improvements."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data.finqa_loader import load_finqa_dataset
from src.pipeline import FinancialQAPipeline
from src.evaluation.metrics import FinQAEvaluator

# Load dataset
dataset = load_finqa_dataset("./finqa_data", download=True, max_dev=100)
eval_examples = dataset.get("dev", dataset.get("train", []))[:100]
print(f"Loaded {len(eval_examples)} examples")

# Create pipeline without LLM (rule-based only)
pipeline = FinancialQAPipeline(load_llm=False)

# Run evaluation
results = pipeline.batch_answer(eval_examples, verbose=True)

# Evaluate
evaluator = FinQAEvaluator(tolerance=0.01)
report = evaluator.evaluate(results, eval_examples)
evaluator.print_report(report)

# Save report
os.makedirs("outputs", exist_ok=True)
with open("outputs/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print("\nReport saved to outputs/evaluation_report.json")

# Detailed failure analysis
print("\n\n--- FAILURE ANALYSIS ---")
failures = []
for r, ex in zip(results, eval_examples):
    pred = r.get("predicted_answer", "")
    gold = r.get("gold_answer", "")
    from src.utils.financial_utils import answers_match
    if not answers_match(pred, gold):
        failures.append({
            "id": r["id"],
            "predicted": pred,
            "gold": gold,
            "program": ex.program,
            "method": r.get("numerical", {}).get("method", ""),
            "success": r.get("numerical", {}).get("success", False),
            "error": r.get("numerical", {}).get("error", ""),
        })

print(f"Total failures: {len(failures)}/{len(results)}")
for f in failures[:20]:
    print(f"\n  ID: {f['id']}")
    print(f"  Predicted: {f['predicted']}")
    print(f"  Gold: {f['gold']}")
    print(f"  Program: {f['program']}")
    print(f"  Method: {f['method']}, Success: {f['success']}")
    if f['error']:
        print(f"  Error: {f['error']}")
