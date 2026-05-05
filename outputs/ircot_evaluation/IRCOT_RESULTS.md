# P3-4b: IRCoT Interleaved Retrieval with Chain-of-Thought

**Reference:** Trivedi et al. (2023) — *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*

**Evaluation Date:** 2026-05-02 16:10:03
**Dataset:** FinQA dev set (200 examples)
**Pipeline Time:** 53.0s (0.26s/example)

## Executive Summary

IRCoT replaces the previous single-pass re-retrieval with an iterative
retrieval-reasoning loop that uses confidence-based termination, CoT-guided
query reformulation, and cross-module gap detection.

## Key IRCoT Metrics

| Metric | Value |
|--------|-------|
| Mean iterations | 1.32 |
| Median iterations | 1.0 |
| Mean final confidence | 0.8435 |
| Confidence std dev | 0.0972 |
| Mean improvement | 0.0387 |
| Convergence rate | 92.0% |
| Multi-iteration rate | 22.0% |

## Termination Reasons

| Reason | Count | Percentage |
|--------|-------|-----------|
| Threshold Met Initial | 156 | 78.0% |
| Threshold Met | 28 | 14.0% |
| Plateau | 16 | 8.0% |

## Confidence Progression Across Iterations

| Iteration | Mean Confidence | Examples |
|-----------|----------------|----------|
| 0 | 0.8048 | 200 |
| 1 | 0.7818 | 44 |
| 2 | 0.6724 | 19 |
| 3 | 0.4500 | 1 |

## Per-Module Confidence

| Module | Mean Confidence | Examples |
|--------|----------------|----------|
| Causal | 0.7769 | 104 |
| Numerical | 0.8500 | 130 |
| Temporal | 0.8667 | 138 |

## Remaining Gaps After IRCoT

| Gap Type | Count |
|----------|-------|
| No Trend | 64 |
| No Relations | 26 |
| Insufficient Entities | 12 |

## Accuracy by Convergence Status

| Group | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Converged | 184 | 184 | 100.0% |
| Not Converged | 16 | 16 | 100.0% |
| **Overall** | **200** | **200** | **100.0%** |

## Baseline Comparison

| Metric | Baseline (Pre-IRCoT) | With IRCoT | Delta |
|--------|---------------------|------------|-------|
| Overall Accuracy | 0.0718 | 1.0000 | +0.9282 |
| Context F1 | 0.4830 | 0.5365 | +0.0535 |
| Temporal Score | 0.8591 | 0.4237 | -0.4354 |
| Causality Detection | 1.0000 | 0.8000 | -0.2000 |

## New Capabilities (IRCoT-Specific)

- **Convergence rate:** 92.0% of examples reach confidence threshold
- **Mean confidence score:** 0.8435
- **Iterative retrieval:** 22.0% trigger additional retrieval passes
- **Plateau detection:** avoids wasted iterations when improvement stalls
- **Gap-guided query reformulation:** targets specific reasoning weaknesses per iteration

## Pipeline Metrics

| Metric | Value |
|--------|-------|
| Overall accuracy | 1.0000 (200/200) |
| Execution accuracy | 1.0000 |
| Context precision | 0.3945 |
| Context recall | 0.9221 |
| Context F1 | 0.5365 |
| Temporal score | 0.4237 |
| Trend detection | 0.3700 |
| Causality detection | 0.8000 |

## Figures

| Figure | Description |
|--------|-------------|
| `ircot_termination_reasons.png` | Distribution of loop termination reasons |
| `ircot_confidence_progression.png` | Mean confidence across IRCoT iterations |
| `ircot_module_confidence.png` | Per-module confidence from IRCoT assessment |
| `ircot_remaining_gaps.png` | Reasoning gaps remaining after IRCoT |
| `ircot_accuracy_by_convergence.png` | Accuracy for converged vs non-converged |
| `ircot_iteration_distribution.png` | Histogram of iteration counts |
| `ircot_performance_radar.png` | Overall IRCoT system performance radar |
