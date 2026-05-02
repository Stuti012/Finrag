"""P3-4b IRCoT Evaluation: run pipeline with interleaved retrieval, compare to baseline, generate plots."""
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

SAVE_DIR = "./outputs/ircot_evaluation"
FIG_DIR = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF9800",
    "success": "#4CAF50",
    "danger": "#F44336",
    "purple": "#9C27B0",
    "teal": "#009688",
    "grey": "#607D8B",
    "blue_light": "#64B5F6",
}
PALETTE = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#009688", "#607D8B"]

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
# Load data + baseline
# ================================================================
print("=" * 70)
print("P3-4b IRCoT EVALUATION — INTERLEAVED RETRIEVAL IMPROVEMENT ANALYSIS")
print("=" * 70)

baseline_path = "./outputs/evaluation_report.json"
baseline = {}
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    print(f"Loaded baseline report from {baseline_path}")
else:
    print("No baseline report found — will generate fresh comparison.")

dataset = load_finqa_dataset("./finqa_data", download=True, max_dev=200)
eval_examples = dataset.get("dev", [])[:200]
print(f"Loaded {len(eval_examples)} dev examples")

# ================================================================
# Run pipeline WITH IRCoT
# ================================================================
print("\n--- Running pipeline with IRCoT interleaved retrieval ---")
t0 = time.time()
pipeline = FinancialQAPipeline(load_llm=False)
results = pipeline.batch_answer(eval_examples, verbose=True)
elapsed = time.time() - t0
print(f"Pipeline completed in {elapsed:.1f}s ({elapsed/len(eval_examples):.2f}s/example)")

# ================================================================
# Evaluate
# ================================================================
evaluator = FinQAEvaluator(tolerance=0.01)
report = evaluator.evaluate(results, eval_examples)
evaluator.print_report(report)

# ================================================================
# Extract IRCoT-specific analytics
# ================================================================
ircot_data = []
for r in results:
    ic = r.get("ircot", {})
    ircot_data.append({
        "id": r.get("id", ""),
        "question": r.get("question", ""),
        "total_iterations": ic.get("total_iterations", 0),
        "final_confidence": ic.get("final_confidence", 0.0),
        "converged": ic.get("converged", False),
        "termination_reason": ic.get("termination_reason", "none"),
        "total_improvement": ic.get("total_improvement", 0.0),
        "module_confidences": ic.get("module_confidences", {}),
        "final_gaps": ic.get("final_gaps", []),
        "iterations": ic.get("iterations", []),
        "correct": bool(answers_match(
            str(r.get("predicted_answer", "")),
            str(r.get("gold_answer", "")),
        )),
    })

# Aggregate IRCoT stats
total = len(ircot_data)
iterations_list = [d["total_iterations"] for d in ircot_data]
confidence_list = [d["final_confidence"] for d in ircot_data]
improvement_list = [d["total_improvement"] for d in ircot_data]
converged_count = sum(1 for d in ircot_data if d["converged"])
multi_iter = sum(1 for d in ircot_data if d["total_iterations"] > 1)

termination_counts = defaultdict(int)
for d in ircot_data:
    termination_counts[d["termination_reason"]] += 1

module_conf_agg = defaultdict(list)
for d in ircot_data:
    for mod, conf in d.get("module_confidences", {}).items():
        module_conf_agg[mod].append(conf)

gap_counts = defaultdict(int)
for d in ircot_data:
    for gap in d.get("final_gaps", []):
        gap_counts[gap] += 1

# Confidence progression per iteration index
iter_confidences = defaultdict(list)
for d in ircot_data:
    for it in d.get("iterations", []):
        iter_confidences[it.get("iteration", 0)].append(it.get("confidence", 0))

# Accuracy by convergence status
converged_correct = sum(1 for d in ircot_data if d["converged"] and d["correct"])
converged_total = sum(1 for d in ircot_data if d["converged"])
not_converged_correct = sum(1 for d in ircot_data if not d["converged"] and d["correct"])
not_converged_total = sum(1 for d in ircot_data if not d["converged"])

ircot_analytics = {
    "total_examples": total,
    "mean_iterations": float(np.mean(iterations_list)),
    "median_iterations": float(np.median(iterations_list)),
    "max_iterations": int(np.max(iterations_list)),
    "mean_final_confidence": float(np.mean(confidence_list)),
    "median_final_confidence": float(np.median(confidence_list)),
    "std_final_confidence": float(np.std(confidence_list)),
    "mean_improvement": float(np.mean(improvement_list)),
    "convergence_rate": converged_count / max(total, 1),
    "multi_iteration_rate": multi_iter / max(total, 1),
    "termination_reasons": dict(termination_counts),
    "module_mean_confidence": {k: float(np.mean(v)) for k, v in module_conf_agg.items()},
    "remaining_gap_distribution": dict(gap_counts),
    "confidence_by_iteration": {str(k): float(np.mean(v)) for k, v in sorted(iter_confidences.items())},
    "accuracy_by_convergence": {
        "converged": {"correct": converged_correct, "total": converged_total,
                      "accuracy": converged_correct / max(converged_total, 1)},
        "not_converged": {"correct": not_converged_correct, "total": not_converged_total,
                          "accuracy": not_converged_correct / max(not_converged_total, 1)},
    },
}

# ================================================================
# Build comparison with baseline
# ================================================================
baseline_induced = baseline.get("induced_program", {})
baseline_acc = baseline_induced.get("overall", {}).get("accuracy", 0)
new_acc = report["overall"]["accuracy"]

comparison = {
    "baseline_accuracy": baseline_acc,
    "new_accuracy": new_acc,
    "accuracy_delta": new_acc - baseline_acc,
    "baseline_context_f1": baseline_induced.get("context_filtering", {}).get("mean_f1", 0),
    "new_context_f1": report["context_filtering"].get("mean_f1", 0),
    "baseline_temporal_score": baseline_induced.get("temporal_reasoning", {}).get("mean_temporal_score", 0),
    "new_temporal_score": report["temporal_reasoning"].get("mean_temporal_score", 0),
    "baseline_causality_detection": baseline_induced.get("causality_detection", {}).get("detection_rate", 0),
    "new_causality_detection": report["causality_detection"].get("detection_rate", 0),
    "new_ircot_convergence_rate": ircot_analytics["convergence_rate"],
    "new_ircot_mean_confidence": ircot_analytics["mean_final_confidence"],
    "new_ircot_mean_iterations": ircot_analytics["mean_iterations"],
}

# ================================================================
# Save comprehensive report
# ================================================================
full_report = {
    "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "feature": "P3-4b IRCoT Interleaved Retrieval with Chain-of-Thought",
    "reference": "Trivedi et al. (2023) — Interleaving Retrieval with Chain-of-Thought Reasoning",
    "num_examples": len(eval_examples),
    "pipeline_time_seconds": round(elapsed, 1),
    "evaluation_report": report,
    "ircot_analytics": ircot_analytics,
    "baseline_comparison": comparison,
    "per_example_ircot": ircot_data,
}

report_path = os.path.join(SAVE_DIR, "ircot_evaluation_report.json")
with open(report_path, "w") as f:
    json.dump(full_report, f, indent=2, default=str)
print(f"\nFull report saved to {report_path}")

# ================================================================
# Generate IRCoT-specific plots
# ================================================================
if not HAS_MATPLOTLIB:
    print("matplotlib not available — skipping plots")
else:
    print("\n" + "=" * 70)
    print("GENERATING IRCoT EVALUATION PLOTS")
    print("=" * 70)

    # --- Plot 1: Termination reason distribution (pie chart) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(termination_counts.keys())
    sizes = list(termination_counts.values())
    color_map = {
        "threshold_met_initial": COLORS["success"],
        "threshold_met": COLORS["primary"],
        "plateau": COLORS["secondary"],
        "max_iterations": COLORS["danger"],
    }
    pie_colors = [color_map.get(l, COLORS["grey"]) for l in labels]
    display_labels = [l.replace("_", " ").title() for l in labels]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=display_labels, autopct="%1.1f%%", colors=pie_colors,
        startangle=90, pctdistance=0.85, textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("IRCoT Termination Reasons\n(Trivedi et al. 2023)", fontsize=14, fontweight="bold")
    legend_labels = [f"{dl} (n={s})" for dl, s in zip(display_labels, sizes)]
    ax.legend(legend_labels, loc="lower right", fontsize=9)
    save_fig(fig, "ircot_termination_reasons.png")

    # --- Plot 2: Confidence progression across iterations ---
    fig, ax = plt.subplots(figsize=(9, 6))
    sorted_iters = sorted(iter_confidences.keys())
    means = [np.mean(iter_confidences[k]) for k in sorted_iters]
    stds = [np.std(iter_confidences[k]) for k in sorted_iters]
    counts = [len(iter_confidences[k]) for k in sorted_iters]
    x = list(sorted_iters)
    ax.plot(x, means, "o-", color=COLORS["primary"], linewidth=2.5, markersize=10, label="Mean Confidence")
    ax.fill_between(x, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color=COLORS["primary"])
    ax.axhline(y=0.7, color=COLORS["danger"], linestyle="--", linewidth=1.5,
               label="Convergence Threshold (0.7)")
    for xi, mi, ci in zip(x, means, counts):
        ax.annotate(f"{mi:.3f}\n(n={ci})", (xi, mi),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=10, fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("IRCoT Iteration", fontsize=12)
    ax.set_ylabel("Mean Confidence Score", fontsize=12)
    ax.set_title("Confidence Progression Across IRCoT Iterations", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Iter {i}" for i in x])
    ax.legend(fontsize=11)
    save_fig(fig, "ircot_confidence_progression.png")

    # --- Plot 3: Module confidence distribution (grouped bar) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    mods = sorted(module_conf_agg.keys())
    if mods:
        mod_means = [np.mean(module_conf_agg[m]) for m in mods]
        mod_stds = [np.std(module_conf_agg[m]) for m in mods]
        mod_counts = [len(module_conf_agg[m]) for m in mods]
        x_pos = np.arange(len(mods))
        bars = ax.bar(x_pos, mod_means, yerr=mod_stds, capsize=5,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(mods))],
                      alpha=0.85, width=0.6, edgecolor="white", linewidth=1.2)
        for bar, mean, cnt in zip(bars, mod_means, mod_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{mean:.3f}\n(n={cnt})", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.capitalize() for m in mods], fontsize=11)
        ax.set_ylabel("Mean Confidence", fontsize=12)
        ax.set_title("Per-Module Confidence Scores (IRCoT Assessment)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.7, color=COLORS["danger"], linestyle="--", linewidth=1.5, alpha=0.6,
                    label="Convergence Threshold")
        ax.legend(fontsize=10)
    save_fig(fig, "ircot_module_confidence.png")

    # --- Plot 4: Remaining gaps distribution ---
    if gap_counts:
        fig, ax = plt.subplots(figsize=(9, 5))
        gap_labels = [g.replace("_", " ").title() for g in gap_counts.keys()]
        gap_values = list(gap_counts.values())
        sorted_pairs = sorted(zip(gap_values, gap_labels), reverse=True)
        gap_values, gap_labels = zip(*sorted_pairs)
        y_pos = np.arange(len(gap_labels))
        bars = ax.barh(y_pos, gap_values, color=COLORS["secondary"], alpha=0.85,
                       height=0.6, edgecolor="white")
        for bar, val in zip(bars, gap_values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=11, fontweight="bold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(gap_labels, fontsize=11)
        ax.set_xlabel("Count", fontsize=12)
        ax.set_title("Remaining Reasoning Gaps After IRCoT", fontsize=14, fontweight="bold")
        save_fig(fig, "ircot_remaining_gaps.png")

    # --- Plot 5: Accuracy comparison (converged vs not converged) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = ["Converged", "Not Converged", "Overall"]
    accs = [
        ircot_analytics["accuracy_by_convergence"]["converged"]["accuracy"],
        ircot_analytics["accuracy_by_convergence"]["not_converged"]["accuracy"],
        new_acc,
    ]
    totals = [
        ircot_analytics["accuracy_by_convergence"]["converged"]["total"],
        ircot_analytics["accuracy_by_convergence"]["not_converged"]["total"],
        report["overall"]["total"],
    ]
    corrects = [
        ircot_analytics["accuracy_by_convergence"]["converged"]["correct"],
        ircot_analytics["accuracy_by_convergence"]["not_converged"]["correct"],
        report["overall"]["correct"],
    ]
    bar_colors = [COLORS["success"], COLORS["danger"], COLORS["primary"]]
    bars = ax.bar(groups, accs, color=bar_colors, alpha=0.85, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, acc, cor, tot in zip(bars, accs, corrects, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}\n({cor}/{tot})", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Accuracy", fontsize=12)
    ax.set_title("Accuracy by IRCoT Convergence Status", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(accs) + 0.15 if max(accs) > 0 else 0.2)
    save_fig(fig, "ircot_accuracy_by_convergence.png")

    # --- Plot 6: Iteration count distribution (histogram) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_iters = sorted(set(iterations_list))
    iter_hist = [iterations_list.count(i) for i in unique_iters]
    bars = ax.bar(unique_iters, iter_hist, color=COLORS["teal"], alpha=0.85,
                  width=0.6, edgecolor="white", linewidth=1.2)
    for bar, cnt in zip(bars, iter_hist):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of IRCoT Iterations", fontsize=12)
    ax.set_ylabel("Number of Examples", fontsize=12)
    ax.set_title("Distribution of IRCoT Iteration Counts", fontsize=14, fontweight="bold")
    ax.set_xticks(unique_iters)
    save_fig(fig, "ircot_iteration_distribution.png")

    # --- Plot 7: Improvement summary radar ---
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = [
        "Convergence\nRate",
        "Mean\nConfidence",
        "Multi-Iteration\nRate",
        "Context\nRetrieval F1",
        "Temporal\nScore",
    ]
    values = [
        ircot_analytics["convergence_rate"],
        ircot_analytics["mean_final_confidence"],
        ircot_analytics["multi_iteration_rate"],
        report["context_filtering"].get("mean_f1", 0),
        report["temporal_reasoning"].get("mean_temporal_score", 0),
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]
    ax.fill(angles, values_plot, color=COLORS["primary"], alpha=0.25)
    ax.plot(angles, values_plot, color=COLORS["primary"], linewidth=2.5, marker="o", markersize=10)
    for angle, val in zip(angles[:-1], values):
        ax.annotate(f"{val:.3f}", xy=(angle, val), xytext=(10, 10),
                    textcoords="offset points", fontsize=10, fontweight="bold", color=COLORS["primary"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("IRCoT System Performance Overview", fontsize=14, fontweight="bold", pad=20)
    save_fig(fig, "ircot_performance_radar.png")

    print(f"\nAll IRCoT plots saved to {FIG_DIR}/")


# ================================================================
# Generate Markdown summary report
# ================================================================
md_lines = []
md_lines.append("# P3-4b: IRCoT Interleaved Retrieval with Chain-of-Thought")
md_lines.append("")
md_lines.append("**Reference:** Trivedi et al. (2023) — *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*")
md_lines.append("")
md_lines.append(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
md_lines.append(f"**Dataset:** FinQA dev set ({len(eval_examples)} examples)")
md_lines.append(f"**Pipeline Time:** {elapsed:.1f}s ({elapsed/len(eval_examples):.2f}s/example)")
md_lines.append("")

md_lines.append("## Executive Summary")
md_lines.append("")
md_lines.append("IRCoT replaces the previous single-pass re-retrieval with an iterative")
md_lines.append("retrieval-reasoning loop that uses confidence-based termination, CoT-guided")
md_lines.append("query reformulation, and cross-module gap detection.")
md_lines.append("")

md_lines.append("## Key IRCoT Metrics")
md_lines.append("")
md_lines.append("| Metric | Value |")
md_lines.append("|--------|-------|")
md_lines.append(f"| Mean iterations | {ircot_analytics['mean_iterations']:.2f} |")
md_lines.append(f"| Median iterations | {ircot_analytics['median_iterations']:.1f} |")
md_lines.append(f"| Mean final confidence | {ircot_analytics['mean_final_confidence']:.4f} |")
md_lines.append(f"| Confidence std dev | {ircot_analytics['std_final_confidence']:.4f} |")
md_lines.append(f"| Mean improvement | {ircot_analytics['mean_improvement']:.4f} |")
md_lines.append(f"| Convergence rate | {ircot_analytics['convergence_rate']:.1%} |")
md_lines.append(f"| Multi-iteration rate | {ircot_analytics['multi_iteration_rate']:.1%} |")
md_lines.append("")

md_lines.append("## Termination Reasons")
md_lines.append("")
md_lines.append("| Reason | Count | Percentage |")
md_lines.append("|--------|-------|-----------|")
for reason, count in sorted(termination_counts.items(), key=lambda x: -x[1]):
    pct = count / total * 100
    md_lines.append(f"| {reason.replace('_', ' ').title()} | {count} | {pct:.1f}% |")
md_lines.append("")

md_lines.append("## Confidence Progression Across Iterations")
md_lines.append("")
md_lines.append("| Iteration | Mean Confidence | Examples |")
md_lines.append("|-----------|----------------|----------|")
for k in sorted(iter_confidences.keys()):
    v = iter_confidences[k]
    md_lines.append(f"| {k} | {np.mean(v):.4f} | {len(v)} |")
md_lines.append("")

md_lines.append("## Per-Module Confidence")
md_lines.append("")
md_lines.append("| Module | Mean Confidence | Examples |")
md_lines.append("|--------|----------------|----------|")
for mod in sorted(module_conf_agg.keys()):
    vals = module_conf_agg[mod]
    md_lines.append(f"| {mod.capitalize()} | {np.mean(vals):.4f} | {len(vals)} |")
md_lines.append("")

if gap_counts:
    md_lines.append("## Remaining Gaps After IRCoT")
    md_lines.append("")
    md_lines.append("| Gap Type | Count |")
    md_lines.append("|----------|-------|")
    for gap, cnt in sorted(gap_counts.items(), key=lambda x: -x[1]):
        md_lines.append(f"| {gap.replace('_', ' ').title()} | {cnt} |")
    md_lines.append("")

md_lines.append("## Accuracy by Convergence Status")
md_lines.append("")
md_lines.append("| Group | Correct | Total | Accuracy |")
md_lines.append("|-------|---------|-------|----------|")
for group in ["converged", "not_converged"]:
    info = ircot_analytics["accuracy_by_convergence"][group]
    md_lines.append(f"| {group.replace('_', ' ').title()} | {info['correct']} | {info['total']} | {info['accuracy']:.1%} |")
md_lines.append(f"| **Overall** | **{report['overall']['correct']}** | **{report['overall']['total']}** | **{new_acc:.1%}** |")
md_lines.append("")

md_lines.append("## Baseline Comparison")
md_lines.append("")
md_lines.append("| Metric | Baseline (Pre-IRCoT) | With IRCoT | Delta |")
md_lines.append("|--------|---------------------|------------|-------|")
for key_nice, bk, nk in [
    ("Overall Accuracy", "baseline_accuracy", "new_accuracy"),
    ("Context F1", "baseline_context_f1", "new_context_f1"),
    ("Temporal Score", "baseline_temporal_score", "new_temporal_score"),
    ("Causality Detection", "baseline_causality_detection", "new_causality_detection"),
]:
    bv = comparison[bk]
    nv = comparison[nk]
    delta = nv - bv
    sign = "+" if delta >= 0 else ""
    md_lines.append(f"| {key_nice} | {bv:.4f} | {nv:.4f} | {sign}{delta:.4f} |")
md_lines.append("")

md_lines.append("## New Capabilities (IRCoT-Specific)")
md_lines.append("")
md_lines.append(f"- **Convergence rate:** {ircot_analytics['convergence_rate']:.1%} of examples reach confidence threshold")
md_lines.append(f"- **Mean confidence score:** {ircot_analytics['mean_final_confidence']:.4f}")
md_lines.append(f"- **Iterative retrieval:** {ircot_analytics['multi_iteration_rate']:.1%} trigger additional retrieval passes")
md_lines.append(f"- **Plateau detection:** avoids wasted iterations when improvement stalls")
md_lines.append(f"- **Gap-guided query reformulation:** targets specific reasoning weaknesses per iteration")
md_lines.append("")

md_lines.append("## Pipeline Metrics")
md_lines.append("")
md_lines.append("| Metric | Value |")
md_lines.append("|--------|-------|")
md_lines.append(f"| Overall accuracy | {new_acc:.4f} ({report['overall']['correct']}/{report['overall']['total']}) |")
md_lines.append(f"| Execution accuracy | {report['numerical_reasoning'].get('execution_accuracy', 0):.4f} |")
md_lines.append(f"| Context precision | {report['context_filtering'].get('mean_precision', 0):.4f} |")
md_lines.append(f"| Context recall | {report['context_filtering'].get('mean_recall', 0):.4f} |")
md_lines.append(f"| Context F1 | {report['context_filtering'].get('mean_f1', 0):.4f} |")
md_lines.append(f"| Temporal score | {report['temporal_reasoning'].get('mean_temporal_score', 0):.4f} |")
md_lines.append(f"| Trend detection | {report['temporal_reasoning'].get('trend_detection_rate', 0):.4f} |")
md_lines.append(f"| Causality detection | {report['causality_detection'].get('detection_rate', 0):.4f} |")
md_lines.append("")

md_lines.append("## Figures")
md_lines.append("")
md_lines.append("| Figure | Description |")
md_lines.append("|--------|-------------|")
md_lines.append("| `ircot_termination_reasons.png` | Distribution of loop termination reasons |")
md_lines.append("| `ircot_confidence_progression.png` | Mean confidence across IRCoT iterations |")
md_lines.append("| `ircot_module_confidence.png` | Per-module confidence from IRCoT assessment |")
md_lines.append("| `ircot_remaining_gaps.png` | Reasoning gaps remaining after IRCoT |")
md_lines.append("| `ircot_accuracy_by_convergence.png` | Accuracy for converged vs non-converged |")
md_lines.append("| `ircot_iteration_distribution.png` | Histogram of iteration counts |")
md_lines.append("| `ircot_performance_radar.png` | Overall IRCoT system performance radar |")
md_lines.append("")

md_text = "\n".join(md_lines)
md_path = os.path.join(SAVE_DIR, "IRCOT_RESULTS.md")
with open(md_path, "w") as f:
    f.write(md_text)
print(f"\nMarkdown report saved to {md_path}")

# ================================================================
# Print final summary
# ================================================================
print("\n" + "=" * 70)
print("IRCoT EVALUATION RESULTS SUMMARY")
print("=" * 70)
print(f"""
IRCoT Configuration:
  Max iterations:         3
  Confidence threshold:   0.7
  Min improvement:        0.02

IRCoT Performance:
  Mean iterations:        {ircot_analytics['mean_iterations']:.2f}
  Convergence rate:       {ircot_analytics['convergence_rate']:.1%}
  Mean final confidence:  {ircot_analytics['mean_final_confidence']:.4f}
  Mean improvement:       {ircot_analytics['mean_improvement']:.4f}
  Multi-iteration rate:   {ircot_analytics['multi_iteration_rate']:.1%}

Termination Reasons:""")
for reason, count in sorted(termination_counts.items(), key=lambda x: -x[1]):
    print(f"  {reason:30s}: {count:4d} ({count/total*100:.1f}%)")

print(f"""
Pipeline Accuracy:
  Overall:                {new_acc:.1%} ({report['overall']['correct']}/{report['overall']['total']})
  Baseline (pre-IRCoT):   {baseline_acc:.1%}
  Delta:                  {new_acc - baseline_acc:+.4f}

Output Files:
  Report:    {report_path}
  Markdown:  {md_path}
  Figures:   {FIG_DIR}/
""")
print("=" * 70)
