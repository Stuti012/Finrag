# Module-Level Accuracy Improvement: Baseline vs IRCoT

**Feature:** P3-4b IRCoT Interleaved Retrieval with Chain-of-Thought
**Date:** 2026-05-02 11:21:13
**Dataset:** FinQA dev set (200 examples)

## Accuracy Comparison

| Module | Baseline | With IRCoT | Absolute Delta | Relative Delta |
|--------|----------|------------|---------------|----------------|
| **Numerical** | 19.2% (25/130) | 19.2% (25/130) | +0.0% | +0% |
| **Temporal** | 4.3% (6/138) | 4.3% (6/138) | +0.0% | +0% |
| **Causal** | 1.9% (2/104) | 1.9% (2/104) | +0.0% | +0% |
| **Overall** | 21.7% (25/115) | 21.6% (25/116) | -0.2% | -1% |

## Numerical Module Details

| Metric | Baseline | IRCoT |
|--------|----------|-------|
| Accuracy | 19.2% | 19.2% |
| Execution success rate | 88.5% | 89.2% |
| Program generation rate | 0.0% | 0.0% |
| Median relative error | 99.00 | 99.00 |

## Temporal Module Details

| Metric | Baseline | IRCoT |
|--------|----------|-------|
| Accuracy | 4.3% | 4.3% |
| Mean entity count | 4.4 | 6.6 |
| Trend detection rate | 50.7% | 52.2% |
| Deictic resolution rate | 100.0% | 100.0% |

## Causal Module Details

| Metric | Baseline | IRCoT |
|--------|----------|-------|
| Accuracy | 1.9% | 1.9% |
| Detection rate | 51.9% | 84.6% |
| Mean relations per question | 0.9 | 1.9 |
| Mean chain confidence | 0.642 | 0.594 |
| Discourse causality rate | 51.9% | 84.6% |
| Temporal-causal overlap | 0.587 | 1.240 |

## IRCoT Performance

- Mean iterations: 1.53
- Convergence rate: 84.5%
- Mean confidence: 0.7652
- Mean improvement: 0.0509
- Retrieval rate: 33.5%

## Timing

- Baseline: 19.0s (0.09s/example)
- IRCoT: 79.1s (0.40s/example)
- Overhead: 316.5%

## Figures

| Figure | Description |
|--------|-------------|
| `module_accuracy_comparison.png` | Side-by-side accuracy bars per module |
| `improvement_waterfall_by_module.png` | Waterfall showing each module's contribution |
| `module_metrics_deep_dive.png` | Detailed numerical + temporal/causal metrics |
| `radar_baseline_vs_ircot.png` | Multi-dimensional radar comparison |
| `module_improvement_summary_table.png` | Publication-ready summary table |