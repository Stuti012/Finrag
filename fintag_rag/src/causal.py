"""Causal Reasoning extension.

The thesis's own metrics are context filtering, temporal reasoning, and
numerical reasoning (Chapter 4). Causal accuracy is not part of the
published FinTAG-RAG evaluation, but the thesis's literature review
(Section 2.5, and reference [17]/[18] FinCausal) motivates cue-phrase-based
cause-effect extraction as the standard lightweight approach for financial
causality detection, so it is added here as a fourth reasoning dimension.

Because FinQA carries no gold causal annotations, `CAUSAL_EVAL_SET` below is
a small, hand-labeled set of representative financial sentences used to
report a genuine (if small-sample) causal accuracy number, rather than an
unverifiable one -- see `evaluate_causal_accuracy`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")

# (cue phrase, direction) where direction says which side of the cue is the
# cause vs. the effect in typical financial-narrative usage.
_CUE_PHRASES = [
    ("as a result of", "effect_first", 0.9),
    ("as a consequence of", "effect_first", 0.85),
    ("primarily due to", "effect_first", 0.9),
    ("mainly due to", "effect_first", 0.9),
    ("due to", "effect_first", 0.85),
    ("owing to", "effect_first", 0.8),
    ("on account of", "effect_first", 0.75),
    ("because of", "effect_first", 0.85),
    ("attributable to", "effect_first", 0.85),
    ("driven by", "effect_first", 0.8),
    ("reflecting", "effect_first", 0.55),
    ("resulted in", "cause_first", 0.85),
    ("resulting in", "cause_first", 0.85),
    ("led to", "cause_first", 0.8),
    ("contributed to", "cause_first", 0.7),
    ("caused", "cause_first", 0.75),
]

_FINANCIAL_KEYWORD_RE = re.compile(
    r"\b(revenue|income|profit|margin|earnings|sales|cost|expense|cash flow|demand|growth|"
    r"decline|increase|decrease|price|volume|rate|debt|asset)\b",
    re.IGNORECASE,
)


@dataclass
class CausalRelation:
    cause: str
    effect: str
    cue_phrase: str
    confidence: float
    source_sentence: str


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _trim_boundary(span: str) -> str:
    return span.strip(" ,.;:-")


class CausalExtractor:
    """Cue-phrase-based cause-effect extraction (FinCausal-style, thesis ref [17])."""

    def extract_from_sentence(self, sentence: str) -> Optional[CausalRelation]:
        lowered = sentence.lower()
        best = None
        for cue, direction, base_conf in _CUE_PHRASES:
            idx = lowered.find(cue)
            if idx == -1:
                continue
            # Prefer the longest / most specific matching cue phrase.
            if best is not None and len(cue) <= len(best[0]):
                continue
            best = (cue, direction, base_conf, idx)
        if best is None:
            return None

        cue, direction, base_conf, idx = best
        left = _trim_boundary(sentence[:idx])
        right = _trim_boundary(sentence[idx + len(cue) :])
        if not left or not right:
            return None
        cause, effect = (left, right) if direction == "cause_first" else (right, left)

        confidence = base_conf
        if _FINANCIAL_KEYWORD_RE.search(cause) and _FINANCIAL_KEYWORD_RE.search(effect):
            confidence = min(1.0, confidence + 0.1)

        return CausalRelation(cause=cause, effect=effect, cue_phrase=cue, confidence=round(confidence, 3), source_sentence=sentence)

    def extract(self, text: str) -> List[CausalRelation]:
        relations = []
        for sentence in split_sentences(text):
            rel = self.extract_from_sentence(sentence)
            if rel:
                relations.append(rel)
        return relations


_CAUSAL_QUESTION_MARKERS = ("why", "reason", "caused", "cause of", "driven by", "due to", "what led", "factors")


def is_causal_question(question: str) -> bool:
    q = question.lower()
    return any(marker in q for marker in _CAUSAL_QUESTION_MARKERS)


# ---------------------------------------------------------------------------
# Small hand-labeled evaluation set (financial-narrative sentences typical of
# 10-K / earnings-report language) used to report a genuine causal accuracy
# number instead of an unverifiable one.
# ---------------------------------------------------------------------------

CAUSAL_EVAL_SET = [
    {
        "text": "Operating margin declined due to rising raw material costs.",
        "gold_cause": "rising raw material costs",
        "gold_effect": "operating margin declined",
    },
    {
        "text": "Revenue increased primarily due to strong demand in international markets.",
        "gold_cause": "strong demand in international markets",
        "gold_effect": "revenue increased",
    },
    {
        "text": "The reduction in headcount resulted in lower selling and administrative expenses.",
        "gold_cause": "the reduction in headcount",
        "gold_effect": "lower selling and administrative expenses",
    },
    {
        "text": "Gross profit fell because of unfavorable currency translation effects.",
        "gold_cause": "unfavorable currency translation effects",
        "gold_effect": "gross profit fell",
    },
    {
        "text": "Higher interest rates contributed to a decline in mortgage originations.",
        "gold_cause": "higher interest rates",
        "gold_effect": "a decline in mortgage originations",
    },
    {
        "text": "The increase in operating income was attributable to cost discipline across all segments.",
        "gold_cause": "cost discipline across all segments",
        "gold_effect": "the increase in operating income",
    },
    {
        "text": "Supply chain disruptions led to higher input costs and delayed shipments.",
        "gold_cause": "supply chain disruptions",
        "gold_effect": "higher input costs and delayed shipments",
    },
    {
        "text": "Net sales grew, driven by robust unit volume growth in the retail channel.",
        "gold_cause": "robust unit volume growth in the retail channel",
        "gold_effect": "net sales grew",
    },
    {
        "text": "The decline in cash flow was mainly due to increased capital expenditures.",
        "gold_cause": "increased capital expenditures",
        "gold_effect": "the decline in cash flow",
    },
    {
        "text": "As a result of the divestiture, total assets decreased by 8 percent.",
        "gold_cause": "the divestiture",
        "gold_effect": "total assets decreased by 8 percent",
    },
]


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(re.findall(r"[a-z]+", pred.lower()))
    gold_tokens = set(re.findall(r"[a-z]+", gold.lower()))
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_causal_accuracy(extractor: CausalExtractor = None) -> dict:
    """Runs the extractor on CAUSAL_EVAL_SET and reports real, computed metrics.

    `span_f1` is the mean token-level F1 between the extracted cause/effect
    spans and the hand-labeled gold spans -- a genuine, if small-sample,
    causal accuracy number (n=len(CAUSAL_EVAL_SET)), not a figure copied from
    the thesis (which does not report a causal metric).
    """
    extractor = extractor or CausalExtractor()
    detected = 0
    cause_f1s, effect_f1s = [], []
    per_example = []
    for ex in CAUSAL_EVAL_SET:
        rel = extractor.extract_from_sentence(ex["text"])
        if rel is None:
            per_example.append({**ex, "predicted": None, "cause_f1": 0.0, "effect_f1": 0.0})
            cause_f1s.append(0.0)
            effect_f1s.append(0.0)
            continue
        detected += 1
        c_f1 = _token_f1(rel.cause, ex["gold_cause"])
        e_f1 = _token_f1(rel.effect, ex["gold_effect"])
        cause_f1s.append(c_f1)
        effect_f1s.append(e_f1)
        per_example.append({**ex, "predicted": rel, "cause_f1": c_f1, "effect_f1": e_f1})

    n = len(CAUSAL_EVAL_SET)
    return {
        "n_examples": n,
        "detection_rate": detected / n,
        "mean_cause_span_f1": sum(cause_f1s) / n,
        "mean_effect_span_f1": sum(effect_f1s) / n,
        "mean_span_f1": (sum(cause_f1s) + sum(effect_f1s)) / (2 * n),
        "per_example": per_example,
    }
