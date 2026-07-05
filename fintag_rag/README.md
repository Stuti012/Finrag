# FinTAG-RAG

A clean, self-contained implementation of **FinTAG-RAG: A Temporal-Aware
Hybrid Retrieval and Reasoning Framework for Robust Financial Question
Answering** — the thesis / paper by Stuti Singh et al. (MANIT Bhopal).

This is an independent project inside the `Finrag` repository: it does not
depend on the rest of the repo's `src/` tree, and can be copied out on its
own.

## Why this exists

The thesis identifies the **Integration Gap** in financial QA: systems
generally optimize retrieval, temporal reasoning, or numerical reasoning in
isolation, but rarely all three together. FinTAG-RAG closes that gap with a
six-stage pipeline where each stage's output constrains the next:

```
question
   │
   ▼
1. Query Processing        — financial ontology expansion, entity/temporal normalization
   │
   ▼
2. Hybrid Retrieval         — FAISS dense + BM25 sparse retrieval, Reciprocal Rank Fusion,
   │                          Cross-Encoder re-ranking
   ▼
3. Context Filtering        — relevance threshold + near-duplicate removal
   │
   ▼
4. Temporal Compatibility   — discards passages from the wrong fiscal year/quarter
   │
   ▼
5. Symbolic Reasoning       — deterministic (Decimal) arithmetic over verified operands,
   │                          not LLM-generated arithmetic
   ▼
6. Answer Generation        — LLM synthesizes the final answer from verified evidence
                               + the already-computed number
```

A **causal reasoning** module (cue-phrase cause/effect extraction) is added on
top, since it was requested as a fourth accuracy dimension alongside the
thesis's context / temporal / numerical metrics.

## Project layout

```
fintag_rag/
├── FinTAG_RAG_Colab.ipynb   # Colab Pro notebook: build, evaluate, save, demo
├── requirements.txt
├── src/
│   ├── config.py            # FinTAGRAGConfig (hyperparameters, Table 4.3)
│   ├── data.py               # FinQA download/parsing, corpus + fact extraction
│   ├── ontology.py           # Module 1: Query Processing
│   ├── retrieval.py          # Modules 2-3: Hybrid Retrieval + Context Filtering
│   ├── temporal.py           # Module 4: Temporal Compatibility
│   ├── symbolic.py           # Module 5: Symbolic Reasoning
│   ├── causal.py             # Causal reasoning extension
│   ├── generation.py         # Module 6: Answer Generation (local LLM or OpenAI)
│   ├── pipeline.py           # Orchestrator + baseline modes (bm25/dense/llm_only/full)
│   └── evaluation.py         # Metrics: context, temporal, numerical, causal accuracy
└── tests/                    # Unit tests for every module (no ML downloads required)
```

## Quick start

Open `FinTAG_RAG_Colab.ipynb` in Google Colab (**Runtime → Change runtime
type → T4 GPU**) and run the cells top to bottom. It will:

1. Install dependencies and clone this repo.
2. Download the [FinQA dataset](https://github.com/czyssrs/FinQA) and build a
   shared, multi-document retrieval corpus (so retrieval has to contend with
   real cross-year / cross-company ambiguity, per Section 3.4 of the thesis).
3. Walk through each of the six modules on a single example.
4. Run the full pipeline end-to-end with a live reasoning trace.
5. **Reproduce the thesis's baseline comparison live** (BM25 / Dense / LLM-only
   / FinTAG-RAG) on a sample of the test set — genuinely computed on this run,
   not copied from the paper.
6. Report the four accuracy dimensions: context, temporal, numerical, causal.
7. Save the retrieval index to Google Drive so a future session can skip
   straight to the demo without rebuilding it.
8. Run a **committee demo**: a per-question four-dimension scorecard, a
   scripted walkthrough, a live free-form question cell, and an optional
   shareable Gradio web app.

### Running locally (no Colab)

```bash
cd fintag_rag
pip install -r requirements.txt
python -m pytest tests/ -v
```

```python
from src.config import FinTAGRAGConfig
from src.data import load_finqa_dataset, build_corpus
from src.retrieval import HybridRetriever
from src.pipeline import FinTAGRAGPipeline

dataset = load_finqa_dataset("./finqa_data", download=True)
config = FinTAGRAGConfig()
chunks = build_corpus(dataset["test"][:200], config.chunk_size_tokens, config.chunk_overlap_tokens)
retriever = HybridRetriever(chunks, config)
pipeline = FinTAGRAGPipeline(retriever, config)

result = pipeline.answer(dataset["test"][0].question)
print(result.answer_text)
```

## Honesty notes on the numbers this project reports

- **Context / temporal / numerical accuracy** are computed live against the
  FinQA test split using the thesis's own definitions (Eq. 4.1 for temporal
  accuracy; exact/tolerance match for numerical accuracy). They will differ
  from the thesis's reported 0.88 / 0.81 / 0.82 because this is a from-scratch
  reproduction on a configurable sample size and corpus size, using an open
  1.5B-parameter local model by default (the thesis used GPT-3.5-Turbo).
- **Causal accuracy** is *not* one of the thesis's reported metrics. It is
  evaluated here against a small, hand-labeled set of 10 representative
  financial-narrative sentences (`src/causal.py::CAUSAL_EVAL_SET`) so the
  number is genuine, if small-sample, rather than presented as a benchmark
  result.
- The **implicit temporal reference heuristic** (`src/symbolic.py::resolve_comparison_years`)
  is a best-effort resolution of phrases like "last year" that the thesis's
  own error analysis (Section 5.7) identifies as ~40% of failures and
  explicitly leaves unresolved. It is flagged in every result
  (`heuristic_temporal_resolution=True`) rather than silently trusted.

## Citation

If you use this implementation, please cite the thesis:

> Stuti Singh, Akhtar Rasool, Rajesh Wadhvani, Abhishek Jadhav. *FinTAG-RAG: A
> Temporal-Aware Hybrid Retrieval and Reasoning Framework for Robust Financial
> Question Answering.* Maulana Azad National Institute of Technology, Bhopal.
