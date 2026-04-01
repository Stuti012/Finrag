# Financial Question Answering System with RAG

A hybrid Retrieval-Augmented Generation system for financial question answering that performs **verifiable numerical computation**, **temporal reasoning**, and **causal inference** across heterogeneous financial data sources.

Built on the [FinQA dataset](https://github.com/czyssrs/FinQA/tree/main/dataset) using open-source LLMs (Llama 3.2).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running on FinQA Dataset](#running-on-finqa-dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Configuration](#configuration)
- [Citation](#citation)

---

## Overview

Financial QA systems must handle three fundamental challenges that standard RAG pipelines struggle with:

1. **Numerical Reasoning** - Multi-step arithmetic (ratios, percentage changes, aggregations) over financial tables
2. **Temporal Reasoning** - Understanding implicit time relationships across fiscal periods, quarters, and event timelines
3. **Causality Detection** - Identifying cause-effect relationships in financial narratives (e.g., "Why did profit decline?")

This system addresses all three through a modular pipeline that combines:

- **Program-of-Thought (PoT)** execution for deterministic, verifiable numerical computation
- **Temporal graph construction** for time-aware reasoning over financial events
- **Pattern-based causal extraction** with financial domain knowledge
- **Hybrid retrieval** (dense semantic + sparse BM25) for both tables and narrative text

---

## Architecture

```
                         +-------------------+
                         |  Financial Query   |
                         +--------+----------+
                                  |
                         +--------v----------+
                         | Question Classifier|
                         |  (pattern-based)   |
                         +--------+----------+
                                  |
                  +---------------+---------------+
                  |               |               |
         +--------v----+ +-------v-------+ +----v-----------+
         |  Numerical  | |   Temporal    | |   Causality    |
         |  Reasoner   | |   Reasoner   | |   Detector     |
         | (PoT exec)  | | (graph-based)| | (pattern + KB) |
         +--------+----+ +-------+-------+ +----+-----------+
                  |               |               |
                  +---------------+---------------+
                                  |
                         +--------v----------+
                         | Hybrid Retriever   |
                         | Dense + BM25       |
                         | (FAISS + custom)   |
                         +--------+----------+
                                  |
                         +--------v----------+
                         | Evidence           |
                         | Aggregation        |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |  LLM Generation    |
                         |  (Llama 3.2)       |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |  Verified Answer   |
                         +-------------------+
```

---

## Project Structure

```
Finrag/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Financial_QA_System.ipynb          # Main notebook with full pipeline demo
├── configs/
│   ├── __init__.py
│   └── config.py                      # Dataclass-based configuration
├── src/
│   ├── __init__.py
│   ├── pipeline.py                    # Integrated QA pipeline + LLM interface
│   ├── data/
│   │   ├── __init__.py
│   │   └── finqa_loader.py            # FinQA dataset download, parsing, preprocessing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── hybrid_retriever.py        # Dense (FAISS) + BM25 hybrid retrieval
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── numerical_reasoner.py      # Program-of-Thought numerical execution
│   │   ├── temporal_reasoner.py       # Temporal graph construction and trend detection
│   │   ├── causality_detector.py      # Causal relation extraction and graph building
│   │   └── question_classifier.py     # Multi-label question type classification
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                 # Comprehensive metrics for all 4 dimensions
│   └── utils/
│       ├── __init__.py
│       └── financial_utils.py         # Financial number parsing, table formatting
├── src/
│   └── visualization/
│       ├── __init__.py
│       └── plot_results.py            # Publication-quality plots and analysis
└── tests/
    └── __init__.py
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for LLM inference; CPU works for reasoning modules)

### Setup

```bash
# Clone the repository
git clone https://github.com/Stuti012/Finrag.git
cd Finrag

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Optional: GPU Support

```bash
# For CUDA GPU acceleration (replace cu121 with your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu  # instead of faiss-cpu
```

### Optional: Llama Model Access

To use the Llama LLM for answer generation:

1. Accept the Llama license at [meta-llama on Hugging Face](https://huggingface.co/meta-llama)
2. Log in to Hugging Face:
   ```bash
   huggingface-cli login
   ```

---

## Quick Start

### Option 1: Run the Notebook

```bash
jupyter notebook Financial_QA_System.ipynb
```

The notebook provides a complete walkthrough:
- Dataset download and exploration
- Component demos (retrieval, numerical, temporal, causal)
- Full pipeline evaluation
- Performance visualization with publication-quality plots

### Option 2: Run from Python

```python
from src.data.finqa_loader import load_finqa_dataset, FinQAExample
from src.pipeline import FinancialQAPipeline
from src.evaluation.metrics import FinQAEvaluator

# Load dataset
dataset = load_finqa_dataset(data_dir="./finqa_data", download=True)

# Initialize pipeline (set load_llm=True for LLM-powered answers)
pipeline = FinancialQAPipeline(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    load_llm=False,       # Set True if GPU + model access available
    load_in_4bit=True,
)

# Answer a question
result = pipeline.answer(dataset['dev'][0])
print(f"Question: {result['question']}")
print(f"Predicted: {result['predicted_answer']}")
print(f"Gold: {result['gold_answer']}")
print(f"Reasoning: {result['reasoning_trace']}")
```

### Option 3: Custom Financial Questions

```python
from src.data.finqa_loader import FinQAExample

example = FinQAExample(
    id="custom_1",
    question="What was the percentage increase in revenue from 2020 to 2021?",
    table=[
        ["Item", "2019", "2020", "2021"],
        ["Revenue", "$50,000", "$60,000", "$75,000"],
        ["Net Income", "$10,000", "$12,000", "$15,000"],
    ],
    pre_text=["The company reported strong growth across all segments."],
    post_text=["Management expects continued expansion."],
    program=["subtract(75000, 60000)", "divide(#0, 60000)", "multiply(#1, 100)"],
    answer="25.0",
)

result = pipeline.answer(example)
print(f"Answer: {result['predicted_answer']}")  # 25.0
```

---

## Running on FinQA Dataset

### Full Evaluation Pipeline

```python
from src.data.finqa_loader import load_finqa_dataset
from src.pipeline import FinancialQAPipeline
from src.evaluation.metrics import FinQAEvaluator

# 1. Load dataset
dataset = load_finqa_dataset(
    data_dir="./finqa_data",
    download=True,
    max_train=500,    # Use None for full dataset
    max_dev=None,     # Full dev set (~1,147 examples)
    max_test=None,    # Full test set (~1,147 examples)
)

# 2. Initialize pipeline
pipeline = FinancialQAPipeline(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    load_llm=True,        # Enable LLM for full evaluation
    load_in_4bit=True,    # 4-bit quantization for memory efficiency
)

# 3. Run batch evaluation
eval_examples = dataset['dev']  # or dataset['test']
results = pipeline.batch_answer(eval_examples, verbose=True)

# 4. Compute metrics
evaluator = FinQAEvaluator(tolerance=0.01)
report = evaluator.evaluate(results, eval_examples)
evaluator.print_report(report)
```

### Generate Visualization Plots

```python
from src.visualization.plot_results import ResultsVisualizer

visualizer = ResultsVisualizer(save_dir="./outputs/figures")
visualizer.generate_all_plots(report, results, eval_examples)
```

This generates publication-quality figures saved to `./outputs/figures/`:

| Figure | Description |
|--------|-------------|
| `overall_performance_radar.png` | Radar chart of all metric dimensions |
| `numerical_accuracy_by_complexity.png` | Bar chart: accuracy vs. program step count |
| `numerical_accuracy_by_operation.png` | Bar chart: accuracy per arithmetic operation |
| `context_filtering_metrics.png` | Grouped bar chart: precision, recall, F1, sufficiency |
| `causality_confidence_distribution.png` | Histogram of causal relation confidence scores |
| `temporal_entity_analysis.png` | Bar chart: temporal entity extraction statistics |
| `question_type_distribution.png` | Pie chart: question type breakdown |
| `per_type_accuracy_comparison.png` | Grouped bar chart: accuracy by question type |
| `error_analysis_heatmap.png` | Heatmap: error types across reasoning categories |
| `retrieval_quality_vs_accuracy.png` | Scatter plot: retrieval sufficiency vs. QA accuracy |
| `performance_summary_table.png` | Publication-ready summary table |

---

## Evaluation Metrics

The system is evaluated across four key dimensions:

### 1. Numerical Reasoning

| Metric | Description |
|--------|-------------|
| **Execution Accuracy** | % of answers matching gold within tolerance (1%) |
| **Program Accuracy** | % of DSL operations correctly identified |
| **Mean Relative Error** | Average |predicted - gold| / |gold| |
| **Accuracy by Complexity** | Breakdown by number of computation steps |
| **Accuracy by Operation** | Breakdown by arithmetic operation type |

### 2. Context Filtering (Retrieval Quality)

| Metric | Description |
|--------|-------------|
| **Precision** | % of retrieved passages that are relevant |
| **Recall** | % of gold evidence passages that were retrieved |
| **F1 Score** | Harmonic mean of precision and recall |
| **Sufficiency Score** | Whether retrieved context contains enough info to answer |

### 3. Causality Detection

| Metric | Description |
|--------|-------------|
| **Detection Rate** | % of causal questions where relations were found |
| **Mean Confidence** | Average confidence of detected causal relations |
| **Relation Diversity** | Number of distinct causal relation types found |
| **Avg Relations/Question** | Average causal relations per causal question |

### 4. Temporal Reasoning

| Metric | Description |
|--------|-------------|
| **Temporal Score** | Composite score for temporal reasoning quality |
| **Entity Extraction** | Average number of temporal entities extracted |
| **Trend Detection Rate** | % of temporal questions with successful trend analysis |
| **Year Precision/Recall** | Accuracy of year extraction from questions |

---

## Results

### Performance Summary (FinQA Dev Set)

| Metric | Score |
|--------|-------|
| Overall QA Accuracy | Computed at runtime |
| Numerical Execution Accuracy | Computed at runtime |
| Context Retrieval F1 | Computed at runtime |
| Causality Detection Rate | Computed at runtime |
| Temporal Reasoning Score | Computed at runtime |

> **Note**: Run the notebook or evaluation script to generate results on your hardware.
> The Program-of-Thought module achieves high accuracy on questions with gold programs,
> as it executes computations deterministically rather than relying on LLM arithmetic.

### Key Findings

1. **Numerical Reasoning**: Program-of-Thought execution eliminates arithmetic hallucinations entirely for questions with structured programs. Accuracy correlates inversely with program complexity (number of steps).

2. **Context Filtering**: Hybrid retrieval (dense + BM25) outperforms pure dense retrieval for financial tables, as BM25 captures exact numerical matches that semantic embeddings miss.

3. **Causality Detection**: Pattern-based extraction with financial domain knowledge identifies explicit causal relations with high confidence. Implicit causality remains challenging.

4. **Temporal Reasoning**: Year and quarter extraction achieves near-perfect recall. Trend detection is effective when table headers contain year labels.

---

## Configuration

All system parameters are configurable via `configs/config.py`:

```python
from configs.config import FinQAConfig

config = FinQAConfig()

# Model settings
config.model.model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Larger model
config.model.load_in_4bit = True
config.model.temperature = 0.1

# Retrieval settings
config.retrieval.top_k_tables = 3
config.retrieval.top_k_text = 5
config.retrieval.bm25_weight = 0.3
config.retrieval.dense_weight = 0.7

# Numerical reasoning
config.numerical.tolerance = 1e-4
config.numerical.execution_timeout = 5

# Evaluation
config.evaluation.numerical_tolerance = 0.01
```

### Supported LLM Models

| Model | Parameters | Recommended For |
|-------|-----------|-----------------|
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Quick testing, CPU |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Balanced performance |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Best quality (needs GPU) |

---

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{finrag2025,
  title={Hybrid Financial QA with Program-of-Thought Reasoning, Temporal Graphs, and Causal Detection},
  author={Stuti Singh},
  year={2025},
  howpublished={\url{https://github.com/Stuti012/Finrag}}
}
```

### Related Work

- **FinQA Dataset**: [Chen et al., 2021](https://github.com/czyssrs/FinQA) - Financial QA over heterogeneous sources
- **Program-of-Thought**: [Chen et al., 2023](https://arxiv.org/abs/2211.12588) - Generating programs for numerical reasoning
- **TRACIE**: [Zhou et al., 2021](https://arxiv.org/abs/2005.00242) - Temporal reasoning framework
- **FinCausal**: [Mariko et al., 2020](https://aclanthology.org/2020.fnp-1.3/) - Financial causality detection

---

## License

This project is for research and educational purposes.
