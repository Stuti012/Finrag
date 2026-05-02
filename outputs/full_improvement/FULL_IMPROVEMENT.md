# Full System Improvement Report

## Original Baseline vs Current System (All P3 Features)

**Date:** 2026-05-02 11:24:47
**Dataset:** FinQA dev set (200 examples)
**Pipeline Time:** 77.2s (0.39s/example)

### Features Included
- P3-2c: Temporal constraint propagation (Allen composition + STN tightening)
- P3-3d: Implicit discourse causality (PDTB-3 taxonomy + Bayesian scoring)
- P3-3e: Granger causal strength (F-test, transfer entropy, bidirectional)
- P3-4b: IRCoT interleaved retrieval (Trivedi et al. 2023)

## Per-Module Accuracy

| Module | Original | Current | Delta | Relative |
|--------|----------|---------|-------|----------|
| **Numerical** | 2.2% (3/134) | 19.2% (25/130) | +17.0% | +759% |
| **Temporal** | 15.6% (10/64) | 4.3% (6/138) | -11.3% | -72% |
| **Causal** | 0.0% (0/2) | 1.9% (2/104) | +1.9% | +1923% |
| **Overall** | 7.2% (13/181) | 21.6% (25/116) | +14.4% | +200% |

## Numerical Module

- Accuracy: 2.2% -> 19.2%
- Execution success rate: 89.2%
- Median relative error: 99.00

## Temporal Module

- Accuracy: 15.6% -> 4.3%
- Mean entity count: 6.6
- Trend detection rate: 52.2%

## Causal Module

- Accuracy: 0.0% -> 1.9%
- Detection rate: 84.6%
- Mean relations/question: 1.9
- Mean chain confidence: 0.594
- Discourse causality rate: 84.6%
- Temporal-causal overlap: 1.240

## New Capabilities (not in baseline)

- **IRCoT convergence rate:** 84.5%
- **IRCoT mean confidence:** 0.7652
- **IRCoT mean iterations:** 1.53
- **Discourse causality detection:** 0.83/example
- **Discourse mean confidence:** 0.3186
- **Counterfactual analysis:** 1.0% query parse rate

## Oracle Upper Bound

- Oracle accuracy (gold programs): 98.0%
- Gap to oracle: 76.4%
- This gap is primarily due to program induction (rule-based vs LLM-generated)

## Figures

| Figure | Description |
|--------|-------------|
| `accuracy_original_vs_current_vs_oracle.png` | Three-way accuracy comparison per module |
| `improvement_by_module.png` | Horizontal bar showing improvement per module |
| `feature_progression_waterfall.png` | Waterfall showing P3 feature contributions |
| `radar_full_improvement.png` | Multi-dimensional radar comparison |
| `full_improvement_summary_table.png` | Publication-ready summary table |