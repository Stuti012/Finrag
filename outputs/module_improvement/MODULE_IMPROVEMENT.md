# Module-Level Accuracy Improvement: Baseline vs IRCoT

**Feature:** P3-4b IRCoT Interleaved Retrieval with Chain-of-Thought
**Date:** 2026-05-02 16:09:03
**Dataset:** FinQA dev set (200 examples)

## Accuracy Comparison

| Module | Baseline | With IRCoT | Absolute Delta | Relative Delta |
|--------|----------|------------|---------------|----------------|
| **Numerical** | 100.0% (130/130) | 100.0% (130/130) | +0.0% | +0% |
| **Temporal** | 100.0% (138/138) | 100.0% (138/138) | +0.0% | +0% |
| **Causal** | 100.0% (104/104) | 100.0% (104/104) | +0.0% | +0% |
| **Overall** | 100.0% (200/200) | 100.0% (200/200) | +0.0% | +0% |

## Numerical Module Details

| Metric | Baseline | IRCoT |
|--------|----------|-------|
| Accuracy | 100.0% | 100.0% |
| Execution success rate | 100.0% | 100.0% |
| Program generation rate | 100.0% | 100.0% |
| Median relative error | 0.00 | 0.00 |

## Temporal Module Details

| Metric | Baseline | IRCoT |
|--------|----------|-------|
| Accuracy | 100.0% | 100.0% |
| Mean entity count | 4.4 | 6.1 |
| Trend detection rate | 52.9% | 53.6% |
| Deictic resolution rate | 100.0% | 100.0% |

## Causal Module Details

| Metric | Baseline | IRCoT |
|--------|----------|-------|
| Accuracy | 100.0% | 100.0% |
| Detection rate | 50.0% | 75.0% |
| Mean relations per question | 0.8 | 1.5 |
| Mean chain confidence | 0.652 | 0.625 |
| Discourse causality rate | 50.0% | 75.0% |
| Temporal-causal overlap | 0.577 | 1.077 |

## IRCoT Performance

- Mean iterations: 1.32
- Convergence rate: 92.0%
- Mean confidence: 0.8435
- Mean improvement: 0.0387
- Retrieval rate: 22.0%

## Timing

- Baseline: 17.3s (0.09s/example)
- IRCoT: 53.5s (0.27s/example)
- Overhead: 209.6%

## Figures

| Figure | Description |
|--------|-------------|
| `module_accuracy_comparison.png` | Side-by-side accuracy bars per module |
| `improvement_waterfall_by_module.png` | Waterfall showing each module's contribution |
| `module_metrics_deep_dive.png` | Detailed numerical + temporal/causal metrics |
| `radar_baseline_vs_ircot.png` | Multi-dimensional radar comparison |
| `module_improvement_summary_table.png` | Publication-ready summary table |