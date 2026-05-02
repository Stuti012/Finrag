"""Generate comparison report showing numerical, temporal, and causal accuracy improvements.

Produces:
  - Side-by-side comparison tables (baseline vs IRCoT)
  - Per-module improvement bar charts
  - Multi-dimensional radar comparison
  - Detailed markdown report
  - JSON data for further analysis
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from src.data.finqa_loader import load_finqa_dataset
from src.pipeline import FinancialQAPipeline
from src.evaluation.metrics import FinQAEvaluator
from src.utils.financial_utils import answers_match

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

SAVE_DIR = "./outputs/module_improvement"
FIG_DIR = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    "baseline": "#90A4AE",
    "ircot": "#2196F3",
    "numerical": "#FF9800",
    "temporal": "#4CAF50",
    "causal": "#9C27B0",
    "success": "#4CAF50",
    "danger": "#F44336",
}
PALETTE = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#009688"]

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# Load dataset
# ================================================================
print("=" * 70)
print("MODULE-LEVEL IMPROVEMENT ANALYSIS")
print("Numerical | Temporal | Causal Accuracy Comparison")
print("=" * 70)

dataset = load_finqa_dataset("./finqa_data", download=True, max_dev=200)
eval_examples = dataset.get("dev", [])[:200]
print(f"Loaded {len(eval_examples)} dev examples\n")

# ================================================================
# Helper: classify each example and compute per-module accuracy
# ================================================================
def per_module_accuracy(results, pipeline):
    """Compute accuracy broken down by which modules were active."""
    module_results = {
        "numerical": {"correct": 0, "total": 0, "examples": []},
        "temporal": {"correct": 0, "total": 0, "examples": []},
        "causal": {"correct": 0, "total": 0, "examples": []},
    }
    for r in results:
        pred = str(r.get("predicted_answer", ""))
        gold = str(r.get("gold_answer", ""))
        is_correct = bool(pred and gold and answers_match(pred, gold))
        active = r.get("classification", {}).get("active_modules", [])
        for mod in ["numerical", "temporal", "causal"]:
            if mod in active:
                module_results[mod]["total"] += 1
                if is_correct:
                    module_results[mod]["correct"] += 1
                module_results[mod]["examples"].append({
                    "id": r.get("id", ""),
                    "correct": is_correct,
                    "predicted": pred,
                    "gold": gold,
                })
    for mod in module_results:
        t = module_results[mod]["total"]
        c = module_results[mod]["correct"]
        module_results[mod]["accuracy"] = c / t if t > 0 else 0.0
    return module_results


def extract_module_metrics(results):
    """Extract detailed per-module quality metrics from results."""
    numerical_metrics = {
        "execution_success_rate": 0.0,
        "mean_relative_error": None,
        "program_generation_rate": 0.0,
    }
    temporal_metrics = {
        "mean_entity_count": 0.0,
        "trend_detection_rate": 0.0,
        "mean_temporal_score": 0.0,
        "deictic_resolution_rate": 0.0,
    }
    causal_metrics = {
        "detection_rate": 0.0,
        "mean_relations_per_q": 0.0,
        "mean_chain_confidence": 0.0,
        "discourse_detection_rate": 0.0,
        "mean_causal_overlap": 0.0,
    }

    num_count, num_success, num_has_program = 0, 0, 0
    rel_errors = []
    temp_count, trend_detected = 0, 0
    entity_counts, temp_scores = [], []
    deictic_found, deictic_resolved = 0, 0
    causal_count, causal_detected = 0, 0
    relation_counts, chain_confs = [], []
    disc_counts, overlap_vals = [], []

    for r in results:
        active = r.get("classification", {}).get("active_modules", [])
        num = r.get("numerical", {})
        temp = r.get("temporal", {})
        causal = r.get("causal", {})

        if "numerical" in active:
            num_count += 1
            if num.get("success"):
                num_success += 1
            if num.get("generated_code") or num.get("program"):
                num_has_program += 1
            if num.get("result") is not None and r.get("gold_answer"):
                try:
                    from src.utils.financial_utils import normalize_answer
                    pred_val = float(normalize_answer(str(num["result"])))
                    gold_val = float(normalize_answer(str(r["gold_answer"])))
                    if gold_val != 0:
                        rel_errors.append(abs(pred_val - gold_val) / abs(gold_val))
                except Exception:
                    pass

        if "temporal" in active:
            temp_count += 1
            entities = temp.get("temporal_entities", [])
            entity_counts.append(len(entities))
            trend = temp.get("trend_analysis", {})
            if isinstance(trend, dict) and trend.get("trend") not in {None, "", "insufficient_data"}:
                trend_detected += 1
            implicit = temp.get("implicit_temporal_entities", [])
            deictic_found += len(implicit)
            deictic_resolved += sum(1 for e in implicit if isinstance(e, dict) and e.get("value") is not None)

        if "causal" in active:
            causal_count += 1
            rels = causal.get("causal_relations", [])
            if rels:
                causal_detected += 1
            relation_counts.append(len(rels))
            chains = causal.get("causal_chains", [])
            for c in chains:
                if isinstance(c, dict) and c.get("propagated_confidence"):
                    chain_confs.append(float(c["propagated_confidence"]))
            disc = causal.get("discourse_analysis", {})
            disc_counts.append(disc.get("total_discourse_relations", 0))
            overlap_vals.append(float(causal.get("temporal_causal_overlap", 0)))

    if num_count:
        numerical_metrics["execution_success_rate"] = num_success / num_count
        numerical_metrics["program_generation_rate"] = num_has_program / num_count
        if rel_errors:
            numerical_metrics["mean_relative_error"] = float(np.mean(rel_errors))
            numerical_metrics["median_relative_error"] = float(np.median(rel_errors))
    if temp_count:
        temporal_metrics["mean_entity_count"] = float(np.mean(entity_counts)) if entity_counts else 0
        temporal_metrics["trend_detection_rate"] = trend_detected / temp_count
        temporal_metrics["deictic_resolution_rate"] = deictic_resolved / max(deictic_found, 1)
    if causal_count:
        causal_metrics["detection_rate"] = causal_detected / causal_count
        causal_metrics["mean_relations_per_q"] = float(np.mean(relation_counts)) if relation_counts else 0
        causal_metrics["mean_chain_confidence"] = float(np.mean(chain_confs)) if chain_confs else 0
        causal_metrics["discourse_detection_rate"] = float(np.mean([1 if d > 0 else 0 for d in disc_counts])) if disc_counts else 0
        causal_metrics["mean_causal_overlap"] = float(np.mean(overlap_vals)) if overlap_vals else 0

    return {
        "numerical": numerical_metrics,
        "temporal": temporal_metrics,
        "causal": causal_metrics,
        "counts": {"numerical": num_count, "temporal": temp_count, "causal": causal_count},
    }


# ================================================================
# MODE 1: Baseline — disable IRCoT (single pass, no iterative loop)
# ================================================================
print("--- MODE 1: Baseline (single-pass, no iterative retrieval) ---")
t0 = time.time()
pipeline_baseline = FinancialQAPipeline(load_llm=False)
# Disable IRCoT by setting threshold to 0 (always converges immediately)
pipeline_baseline.ircot_controller.confidence_threshold = 0.0
results_baseline = pipeline_baseline.batch_answer(eval_examples, verbose=True)
baseline_time = time.time() - t0

evaluator = FinQAEvaluator(tolerance=0.01)
report_baseline = evaluator.evaluate(results_baseline, eval_examples)
mod_acc_baseline = per_module_accuracy(results_baseline, pipeline_baseline)
mod_metrics_baseline = extract_module_metrics(results_baseline)

print(f"\nBaseline completed in {baseline_time:.1f}s")
print(f"  Overall accuracy: {report_baseline['overall']['accuracy']:.4f}")
for m in ["numerical", "temporal", "causal"]:
    info = mod_acc_baseline[m]
    print(f"  {m:12s}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")

# ================================================================
# MODE 2: With IRCoT (full interleaved retrieval)
# ================================================================
print("\n--- MODE 2: With IRCoT (interleaved retrieval-reasoning) ---")
t0 = time.time()
pipeline_ircot = FinancialQAPipeline(load_llm=False)
# IRCoT uses default settings (threshold=0.7, max_iter=3)
results_ircot = pipeline_ircot.batch_answer(eval_examples, verbose=True)
ircot_time = time.time() - t0

report_ircot = evaluator.evaluate(results_ircot, eval_examples)
mod_acc_ircot = per_module_accuracy(results_ircot, pipeline_ircot)
mod_metrics_ircot = extract_module_metrics(results_ircot)

print(f"\nIRCoT completed in {ircot_time:.1f}s")
print(f"  Overall accuracy: {report_ircot['overall']['accuracy']:.4f}")
for m in ["numerical", "temporal", "causal"]:
    info = mod_acc_ircot[m]
    print(f"  {m:12s}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")

# ================================================================
# Build comparison data
# ================================================================
comparison = {
    "overall": {
        "baseline": report_baseline["overall"]["accuracy"],
        "ircot": report_ircot["overall"]["accuracy"],
        "delta": report_ircot["overall"]["accuracy"] - report_baseline["overall"]["accuracy"],
        "baseline_correct": report_baseline["overall"]["correct"],
        "baseline_total": report_baseline["overall"]["total"],
        "ircot_correct": report_ircot["overall"]["correct"],
        "ircot_total": report_ircot["overall"]["total"],
    },
    "per_module_accuracy": {},
    "per_module_metrics": {"baseline": mod_metrics_baseline, "ircot": mod_metrics_ircot},
    "ircot_analytics": report_ircot.get("ircot", {}),
    "timing": {
        "baseline_seconds": round(baseline_time, 1),
        "ircot_seconds": round(ircot_time, 1),
        "overhead_pct": round((ircot_time - baseline_time) / max(baseline_time, 0.1) * 100, 1),
    },
}

for mod in ["numerical", "temporal", "causal"]:
    b = mod_acc_baseline[mod]
    n = mod_acc_ircot[mod]
    comparison["per_module_accuracy"][mod] = {
        "baseline_accuracy": b["accuracy"],
        "baseline_correct": b["correct"],
        "baseline_total": b["total"],
        "ircot_accuracy": n["accuracy"],
        "ircot_correct": n["correct"],
        "ircot_total": n["total"],
        "delta": n["accuracy"] - b["accuracy"],
        "relative_improvement": (n["accuracy"] - b["accuracy"]) / max(b["accuracy"], 0.001),
    }

# ================================================================
# Save JSON report
# ================================================================
full_report = {
    "title": "Module-Level Accuracy Improvement Analysis",
    "feature": "P3-4b IRCoT Interleaved Retrieval with Chain-of-Thought",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "num_examples": len(eval_examples),
    "comparison": comparison,
    "baseline_report": report_baseline,
    "ircot_report": report_ircot,
}

json_path = os.path.join(SAVE_DIR, "module_improvement_report.json")
with open(json_path, "w") as f:
    json.dump(full_report, f, indent=2, default=str)
print(f"\nJSON report saved to {json_path}")

# ================================================================
# Generate comparison plots
# ================================================================
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)

    # ---- Plot 1: Per-module accuracy comparison (grouped bar) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    modules = ["Numerical", "Temporal", "Causal", "Overall"]
    baseline_accs = [
        mod_acc_baseline["numerical"]["accuracy"],
        mod_acc_baseline["temporal"]["accuracy"],
        mod_acc_baseline["causal"]["accuracy"],
        report_baseline["overall"]["accuracy"],
    ]
    ircot_accs = [
        mod_acc_ircot["numerical"]["accuracy"],
        mod_acc_ircot["temporal"]["accuracy"],
        mod_acc_ircot["causal"]["accuracy"],
        report_ircot["overall"]["accuracy"],
    ]

    x = np.arange(len(modules))
    width = 0.35
    bars1 = ax.bar(x - width / 2, baseline_accs, width, label="Baseline (Single-Pass)",
                   color=COLORS["baseline"], alpha=0.85, edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width / 2, ircot_accs, width, label="With IRCoT",
                   color=COLORS["ircot"], alpha=0.85, edgecolor="white", linewidth=1.2)

    for bar, acc in zip(bars1, baseline_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=10, color=COLORS["baseline"], fontweight="bold")
    for bar, acc, bl_acc in zip(bars2, ircot_accs, baseline_accs):
        delta = acc - bl_acc
        sign = "+" if delta >= 0 else ""
        label = f"{acc:.1%}\n({sign}{delta:.1%})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                label, ha="center", va="bottom", fontsize=10, color=COLORS["ircot"], fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Module Accuracy: Baseline vs IRCoT", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=12)
    ax.set_ylim(0, max(max(baseline_accs), max(ircot_accs)) + 0.15)
    ax.legend(fontsize=11, loc="upper right")
    save_fig(fig, "module_accuracy_comparison.png")

    # ---- Plot 2: Improvement waterfall ----
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Baseline\nAccuracy"]
    values = [report_baseline["overall"]["accuracy"]]
    bottoms = [0]
    colors = [COLORS["baseline"]]

    # Compute approximate contribution of each module improvement
    num_delta = mod_acc_ircot["numerical"]["accuracy"] - mod_acc_baseline["numerical"]["accuracy"]
    temp_delta = mod_acc_ircot["temporal"]["accuracy"] - mod_acc_baseline["temporal"]["accuracy"]
    causal_delta = mod_acc_ircot["causal"]["accuracy"] - mod_acc_baseline["causal"]["accuracy"]

    num_weight = mod_acc_ircot["numerical"]["total"] / max(report_ircot["overall"]["total"], 1)
    temp_weight = mod_acc_ircot["temporal"]["total"] / max(report_ircot["overall"]["total"], 1)
    causal_weight = mod_acc_ircot["causal"]["total"] / max(report_ircot["overall"]["total"], 1)

    overall_delta = report_ircot["overall"]["accuracy"] - report_baseline["overall"]["accuracy"]
    num_contrib = num_delta * num_weight
    temp_contrib = temp_delta * temp_weight
    causal_contrib = causal_delta * causal_weight
    other_contrib = overall_delta - num_contrib - temp_contrib - causal_contrib

    running = report_baseline["overall"]["accuracy"]
    contributions = [
        ("+ Numerical\nImprovement", num_contrib, COLORS["numerical"]),
        ("+ Temporal\nImprovement", temp_contrib, COLORS["temporal"]),
        ("+ Causal\nImprovement", causal_contrib, COLORS["causal"]),
    ]
    if abs(other_contrib) > 0.001:
        contributions.append(("+ Cross-Module\nEffects", other_contrib, "#607D8B"))

    for name, contrib, color in contributions:
        categories.append(name)
        bottoms.append(running)
        values.append(abs(contrib))
        colors.append(color if contrib >= 0 else COLORS["danger"])
        running += contrib

    categories.append("IRCoT\nAccuracy")
    bottoms.append(0)
    values.append(report_ircot["overall"]["accuracy"])
    colors.append(COLORS["ircot"])

    bars = ax.bar(categories, values, bottom=bottoms, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5, width=0.6)

    for i, (cat, val, bot) in enumerate(zip(categories, values, bottoms)):
        if i == 0 or i == len(categories) - 1:
            ax.text(i, bot + val + 0.005, f"{bot + val:.1%}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        elif val > 0.001:
            sign = "+" if val >= 0 else ""
            ax.text(i, bot + val + 0.005, f"{sign}{val:.1%}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Improvement Waterfall by Module", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) + 0.1)
    save_fig(fig, "improvement_waterfall_by_module.png")

    # ---- Plot 3: Numerical module deep-dive ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Numerical accuracy + success rate
    ax = axes[0]
    metrics = ["Accuracy", "Execution\nSuccess", "Program\nGeneration"]
    bl_vals = [
        mod_acc_baseline["numerical"]["accuracy"],
        mod_metrics_baseline["numerical"]["execution_success_rate"],
        mod_metrics_baseline["numerical"]["program_generation_rate"],
    ]
    ir_vals = [
        mod_acc_ircot["numerical"]["accuracy"],
        mod_metrics_ircot["numerical"]["execution_success_rate"],
        mod_metrics_ircot["numerical"]["program_generation_rate"],
    ]
    x = np.arange(len(metrics))
    ax.bar(x - 0.17, bl_vals, 0.34, label="Baseline", color=COLORS["baseline"], alpha=0.85)
    ax.bar(x + 0.17, ir_vals, 0.34, label="IRCoT", color=COLORS["numerical"], alpha=0.85)
    for xi, bv, iv in zip(x, bl_vals, ir_vals):
        ax.text(xi - 0.17, bv + 0.01, f"{bv:.1%}", ha="center", fontsize=9, color=COLORS["baseline"], fontweight="bold")
        ax.text(xi + 0.17, iv + 0.01, f"{iv:.1%}", ha="center", fontsize=9, color=COLORS["numerical"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Rate")
    ax.set_title("Numerical Module", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(bl_vals), max(ir_vals)) + 0.15)

    # Right: Temporal + Causal metrics
    ax = axes[1]
    metrics2 = ["Temporal\nTrend Det.", "Causal\nDetection", "Discourse\nCausality", "Temporal-Causal\nOverlap"]
    bl_vals2 = [
        mod_metrics_baseline["temporal"]["trend_detection_rate"],
        mod_metrics_baseline["causal"]["detection_rate"],
        mod_metrics_baseline["causal"]["discourse_detection_rate"],
        mod_metrics_baseline["causal"]["mean_causal_overlap"],
    ]
    ir_vals2 = [
        mod_metrics_ircot["temporal"]["trend_detection_rate"],
        mod_metrics_ircot["causal"]["detection_rate"],
        mod_metrics_ircot["causal"]["discourse_detection_rate"],
        mod_metrics_ircot["causal"]["mean_causal_overlap"],
    ]
    x2 = np.arange(len(metrics2))
    ax.bar(x2 - 0.17, bl_vals2, 0.34, label="Baseline", color=COLORS["baseline"], alpha=0.85)
    ax.bar(x2 + 0.17, ir_vals2, 0.34, label="IRCoT", color=COLORS["temporal"], alpha=0.85)
    for xi, bv, iv in zip(x2, bl_vals2, ir_vals2):
        ax.text(xi - 0.17, bv + 0.01, f"{bv:.2f}", ha="center", fontsize=9, color=COLORS["baseline"], fontweight="bold")
        ax.text(xi + 0.17, iv + 0.01, f"{iv:.2f}", ha="center", fontsize=9, color=COLORS["temporal"], fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics2, fontsize=9)
    ax.set_ylabel("Rate / Score")
    ax.set_title("Temporal & Causal Modules", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(bl_vals2), max(ir_vals2)) + 0.15)

    fig.suptitle("Detailed Module-Level Metric Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "module_metrics_deep_dive.png")

    # ---- Plot 4: Multi-dimensional radar comparison ----
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    categories = [
        "Numerical\nAccuracy",
        "Temporal\nAccuracy",
        "Causal\nAccuracy",
        "Context\nRetrieval F1",
        "Execution\nSuccess",
        "Causality\nDetection",
    ]
    bl_radar = [
        mod_acc_baseline["numerical"]["accuracy"],
        mod_acc_baseline["temporal"]["accuracy"],
        mod_acc_baseline["causal"]["accuracy"],
        report_baseline["context_filtering"].get("mean_f1", 0),
        mod_metrics_baseline["numerical"]["execution_success_rate"],
        mod_metrics_baseline["causal"]["detection_rate"],
    ]
    ir_radar = [
        mod_acc_ircot["numerical"]["accuracy"],
        mod_acc_ircot["temporal"]["accuracy"],
        mod_acc_ircot["causal"]["accuracy"],
        report_ircot["context_filtering"].get("mean_f1", 0),
        mod_metrics_ircot["numerical"]["execution_success_rate"],
        mod_metrics_ircot["causal"]["detection_rate"],
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    bl_plot = bl_radar + [bl_radar[0]]
    ir_plot = ir_radar + [ir_radar[0]]

    ax.fill(angles, bl_plot, alpha=0.15, color=COLORS["baseline"])
    ax.plot(angles, bl_plot, linewidth=2, label="Baseline", color=COLORS["baseline"],
            marker="o", markersize=7)
    ax.fill(angles, ir_plot, alpha=0.15, color=COLORS["ircot"])
    ax.plot(angles, ir_plot, linewidth=2.5, label="With IRCoT", color=COLORS["ircot"],
            marker="o", markersize=8)

    for angle, bl_v, ir_v in zip(angles[:-1], bl_radar, ir_radar):
        ax.annotate(f"{ir_v:.2f}", xy=(angle, ir_v), xytext=(12, 8),
                    textcoords="offset points", fontsize=9, fontweight="bold", color=COLORS["ircot"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title("Multi-Dimensional: Baseline vs IRCoT", fontsize=14, fontweight="bold", pad=25)
    save_fig(fig, "radar_baseline_vs_ircot.png")

    # ---- Plot 5: Summary comparison table as figure ----
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    rows = []
    for mod in ["numerical", "temporal", "causal"]:
        b = comparison["per_module_accuracy"][mod]
        delta = b["delta"]
        sign = "+" if delta >= 0 else ""
        rel = b["relative_improvement"]
        rows.append([
            mod.capitalize(),
            f"{b['baseline_accuracy']:.1%} ({b['baseline_correct']}/{b['baseline_total']})",
            f"{b['ircot_accuracy']:.1%} ({b['ircot_correct']}/{b['ircot_total']})",
            f"{sign}{delta:.1%}",
            f"{sign}{rel:.0%}" if abs(rel) < 100 else f"{sign}{rel:.1f}x",
        ])
    # Overall
    o = comparison["overall"]
    o_delta = o["delta"]
    o_rel = o_delta / max(o["baseline"], 0.001)
    o_sign = "+" if o_delta >= 0 else ""
    rows.append([
        "Overall",
        f"{o['baseline']:.1%} ({o['baseline_correct']}/{o['baseline_total']})",
        f"{o['ircot']:.1%} ({o['ircot_correct']}/{o['ircot_total']})",
        f"{o_sign}{o_delta:.1%}",
        f"{o_sign}{o_rel:.0%}" if abs(o_rel) < 100 else f"{o_sign}{o_rel:.1f}x",
    ])

    col_labels = ["Module", "Baseline Accuracy", "IRCoT Accuracy", "Absolute Δ", "Relative Δ"]
    table = ax.table(
        cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
        colWidths=[0.15, 0.22, 0.22, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(COLORS["ircot"])
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i == len(rows):
                cell.set_facecolor("#E3F2FD")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#f5f5f5")
            # Color the delta column
            if j == 3:
                delta_val = [mod_acc_ircot[m]["accuracy"] - mod_acc_baseline[m]["accuracy"]
                             for m in ["numerical", "temporal", "causal"]]
                delta_val.append(o_delta)
                if delta_val[i - 1] > 0:
                    cell.set_text_props(color=COLORS["success"], fontweight="bold")
                elif delta_val[i - 1] < 0:
                    cell.set_text_props(color=COLORS["danger"], fontweight="bold")

    ax.set_title("Module-Level Accuracy Improvement Summary",
                 fontsize=14, fontweight="bold", pad=30)
    save_fig(fig, "module_improvement_summary_table.png")

    print(f"\nAll comparison plots saved to {FIG_DIR}/")


# ================================================================
# Generate Markdown report
# ================================================================
md = []
md.append("# Module-Level Accuracy Improvement: Baseline vs IRCoT")
md.append("")
md.append("**Feature:** P3-4b IRCoT Interleaved Retrieval with Chain-of-Thought")
md.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
md.append(f"**Dataset:** FinQA dev set ({len(eval_examples)} examples)")
md.append("")
md.append("## Accuracy Comparison")
md.append("")
md.append("| Module | Baseline | With IRCoT | Absolute Delta | Relative Delta |")
md.append("|--------|----------|------------|---------------|----------------|")
for mod in ["numerical", "temporal", "causal"]:
    b = comparison["per_module_accuracy"][mod]
    delta = b["delta"]
    sign = "+" if delta >= 0 else ""
    rel = b["relative_improvement"]
    rel_str = f"{sign}{rel:.0%}" if abs(rel) < 100 else f"{sign}{rel:.1f}x"
    md.append(f"| **{mod.capitalize()}** | {b['baseline_accuracy']:.1%} ({b['baseline_correct']}/{b['baseline_total']}) | {b['ircot_accuracy']:.1%} ({b['ircot_correct']}/{b['ircot_total']}) | {sign}{delta:.1%} | {rel_str} |")
o = comparison["overall"]
o_delta = o["delta"]
o_sign = "+" if o_delta >= 0 else ""
o_rel = o_delta / max(o["baseline"], 0.001)
o_rel_str = f"{o_sign}{o_rel:.0%}" if abs(o_rel) < 100 else f"{o_sign}{o_rel:.1f}x"
md.append(f"| **Overall** | {o['baseline']:.1%} ({o['baseline_correct']}/{o['baseline_total']}) | {o['ircot']:.1%} ({o['ircot_correct']}/{o['ircot_total']}) | {o_sign}{o_delta:.1%} | {o_rel_str} |")
md.append("")

md.append("## Numerical Module Details")
md.append("")
md.append("| Metric | Baseline | IRCoT |")
md.append("|--------|----------|-------|")
md.append(f"| Accuracy | {mod_acc_baseline['numerical']['accuracy']:.1%} | {mod_acc_ircot['numerical']['accuracy']:.1%} |")
md.append(f"| Execution success rate | {mod_metrics_baseline['numerical']['execution_success_rate']:.1%} | {mod_metrics_ircot['numerical']['execution_success_rate']:.1%} |")
md.append(f"| Program generation rate | {mod_metrics_baseline['numerical']['program_generation_rate']:.1%} | {mod_metrics_ircot['numerical']['program_generation_rate']:.1%} |")
bl_mre = mod_metrics_baseline["numerical"].get("median_relative_error")
ir_mre = mod_metrics_ircot["numerical"].get("median_relative_error")
md.append(f"| Median relative error | {bl_mre:.2f} | {ir_mre:.2f} |" if bl_mre and ir_mre else f"| Median relative error | N/A | N/A |")
md.append("")

md.append("## Temporal Module Details")
md.append("")
md.append("| Metric | Baseline | IRCoT |")
md.append("|--------|----------|-------|")
md.append(f"| Accuracy | {mod_acc_baseline['temporal']['accuracy']:.1%} | {mod_acc_ircot['temporal']['accuracy']:.1%} |")
md.append(f"| Mean entity count | {mod_metrics_baseline['temporal']['mean_entity_count']:.1f} | {mod_metrics_ircot['temporal']['mean_entity_count']:.1f} |")
md.append(f"| Trend detection rate | {mod_metrics_baseline['temporal']['trend_detection_rate']:.1%} | {mod_metrics_ircot['temporal']['trend_detection_rate']:.1%} |")
md.append(f"| Deictic resolution rate | {mod_metrics_baseline['temporal']['deictic_resolution_rate']:.1%} | {mod_metrics_ircot['temporal']['deictic_resolution_rate']:.1%} |")
md.append("")

md.append("## Causal Module Details")
md.append("")
md.append("| Metric | Baseline | IRCoT |")
md.append("|--------|----------|-------|")
md.append(f"| Accuracy | {mod_acc_baseline['causal']['accuracy']:.1%} | {mod_acc_ircot['causal']['accuracy']:.1%} |")
md.append(f"| Detection rate | {mod_metrics_baseline['causal']['detection_rate']:.1%} | {mod_metrics_ircot['causal']['detection_rate']:.1%} |")
md.append(f"| Mean relations per question | {mod_metrics_baseline['causal']['mean_relations_per_q']:.1f} | {mod_metrics_ircot['causal']['mean_relations_per_q']:.1f} |")
md.append(f"| Mean chain confidence | {mod_metrics_baseline['causal']['mean_chain_confidence']:.3f} | {mod_metrics_ircot['causal']['mean_chain_confidence']:.3f} |")
md.append(f"| Discourse causality rate | {mod_metrics_baseline['causal']['discourse_detection_rate']:.1%} | {mod_metrics_ircot['causal']['discourse_detection_rate']:.1%} |")
md.append(f"| Temporal-causal overlap | {mod_metrics_baseline['causal']['mean_causal_overlap']:.3f} | {mod_metrics_ircot['causal']['mean_causal_overlap']:.3f} |")
md.append("")

md.append("## IRCoT Performance")
md.append("")
ircot_metrics = report_ircot.get("ircot", {})
md.append(f"- Mean iterations: {ircot_metrics.get('mean_ircot_iterations', 0):.2f}")
md.append(f"- Convergence rate: {ircot_metrics.get('ircot_convergence_rate', 0):.1%}")
md.append(f"- Mean confidence: {ircot_metrics.get('mean_ircot_confidence', 0):.4f}")
md.append(f"- Mean improvement: {ircot_metrics.get('mean_ircot_improvement', 0):.4f}")
md.append(f"- Retrieval rate: {ircot_metrics.get('ircot_retrieval_rate', 0):.1%}")
md.append("")

md.append("## Timing")
md.append("")
md.append(f"- Baseline: {baseline_time:.1f}s ({baseline_time/len(eval_examples):.2f}s/example)")
md.append(f"- IRCoT: {ircot_time:.1f}s ({ircot_time/len(eval_examples):.2f}s/example)")
md.append(f"- Overhead: {comparison['timing']['overhead_pct']}%")
md.append("")

md.append("## Figures")
md.append("")
md.append("| Figure | Description |")
md.append("|--------|-------------|")
md.append("| `module_accuracy_comparison.png` | Side-by-side accuracy bars per module |")
md.append("| `improvement_waterfall_by_module.png` | Waterfall showing each module's contribution |")
md.append("| `module_metrics_deep_dive.png` | Detailed numerical + temporal/causal metrics |")
md.append("| `radar_baseline_vs_ircot.png` | Multi-dimensional radar comparison |")
md.append("| `module_improvement_summary_table.png` | Publication-ready summary table |")

md_text = "\n".join(md)
md_path = os.path.join(SAVE_DIR, "MODULE_IMPROVEMENT.md")
with open(md_path, "w") as f:
    f.write(md_text)
print(f"\nMarkdown report saved to {md_path}")

# ================================================================
# Print final summary
# ================================================================
print("\n" + "=" * 70)
print("MODULE IMPROVEMENT SUMMARY")
print("=" * 70)
print(f"\n{'Module':<12} {'Baseline':>12} {'IRCoT':>12} {'Delta':>10}")
print("-" * 48)
for mod in ["numerical", "temporal", "causal"]:
    b = comparison["per_module_accuracy"][mod]
    delta = b["delta"]
    sign = "+" if delta >= 0 else ""
    print(f"{mod.capitalize():<12} {b['baseline_accuracy']:>11.1%} {b['ircot_accuracy']:>11.1%} {sign}{delta:>9.1%}")
print("-" * 48)
print(f"{'Overall':<12} {o['baseline']:>11.1%} {o['ircot']:>11.1%} {o_sign}{o_delta:>9.1%}")
print(f"""
Output:
  JSON:     {json_path}
  Markdown: {md_path}
  Figures:  {FIG_DIR}/
""")
print("=" * 70)
