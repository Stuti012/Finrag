# P3-4b: IRCoT Interleaved Retrieval with Chain-of-Thought

**Reference:** Trivedi et al. (2023) — *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*

**Evaluation Date:** 2026-05-02 08:36:38
**Dataset:** FinQA dev set (200 examples)
**Pipeline Time:** 83.3s (0.42s/example)

## Executive Summary

IRCoT replaces the previous single-pass re-retrieval with an iterative
retrieval-reasoning loop that uses confidence-based termination, CoT-guided
query reformulation, and cross-module gap detection.

## Key IRCoT Metrics

| Metric | Value |
|--------|-------|
| Mean iterations | 1.53 |
| Median iterations | 1.0 |
| Mean final confidence | 0.7652 |
| Confidence std dev | 0.1602 |
| Mean improvement | 0.0509 |
| Convergence rate | 84.5% |
| Multi-iteration rate | 33.5% |

## Termination Reasons

| Reason | Count | Percentage |
|--------|-------|-----------|
| Threshold Met Initial | 133 | 66.5% |
| Threshold Met | 36 | 18.0% |
| Plateau | 31 | 15.5% |

## Confidence Progression Across Iterations

| Iteration | Mean Confidence | Examples |
|-----------|----------------|----------|
| 0 | 0.7143 | 200 |
| 1 | 0.6823 | 67 |
| 2 | 0.5762 | 36 |
| 3 | 0.5333 | 3 |

## Per-Module Confidence

| Module | Mean Confidence | Examples |
|--------|----------------|----------|
| Causal | 0.8409 | 104 |
| Numerical | 0.6408 | 130 |
| Temporal | 0.8674 | 138 |

## Remaining Gaps After IRCoT

| Gap Type | Count |
|----------|-------|
| No Trend | 66 |
| No Relations | 16 |
| Execution Failure | 14 |
| No Result | 14 |
| Insufficient Entities | 9 |

## Accuracy by Convergence Status

| Group | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Converged | 25 | 169 | 14.8% |
| Not Converged | 0 | 31 | 0.0% |
| **Overall** | **25** | **116** | **21.6%** |

## Baseline Comparison

| Metric | Baseline (Pre-IRCoT) | With IRCoT | Delta |
|--------|---------------------|------------|-------|
| Overall Accuracy | 0.0718 | 0.2155 | +0.1437 |
| Context F1 | 0.4830 | 0.5305 | +0.0475 |
| Temporal Score | 0.8591 | 0.4308 | -0.4283 |
| Causality Detection | 1.0000 | 1.0000 | +0.0000 |

## New Capabilities (IRCoT-Specific)

- **Convergence rate:** 84.5% of examples reach confidence threshold
- **Mean confidence score:** 0.7652
- **Iterative retrieval:** 33.5% trigger additional retrieval passes
- **Plateau detection:** avoids wasted iterations when improvement stalls
- **Gap-guided query reformulation:** targets specific reasoning weaknesses per iteration

## Pipeline Metrics

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.2155 (25/116) |
| Execution accuracy | 0.2155 |
| Context precision | 0.3878 |
| Context recall | 0.9196 |
| Context F1 | 0.5305 |
| Temporal score | 0.4308 |
| Trend detection | 0.3600 |
| Causality detection | 1.0000 |

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
