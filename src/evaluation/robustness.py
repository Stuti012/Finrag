"""Robustness/perturbation evaluation harness.

Targets the open challenge named in the companion review paper ("A Review on
Financial Question Answering Systems", §3.2): that benchmark accuracy alone
doesn't establish *genuine* reasoning, since models may be exploiting
statistical shortcuts that break under small, realistic input perturbations
(cf. Stolfo et al.'s causal-robustness framework for mathematical reasoning).

Two perturbation families are implemented, each probing a specific component
of FinTAG-RAG:

1. **Distractor-year injection** (:class:`DistractorInjector`) — appends an
   off-period numeric sentence, borrowed from a different filing, into an
   example's prose. This recreates the exact failure mode the temporal-
   compatibility gate (§3.3) is designed to prevent — cross-year operand
   contamination — and lets us measure whether the gate actually protects
   accuracy under a realistic distractor, rather than just discarding
   passages in a vacuum.

2. **Ontology paraphrase** (:class:`SynonymParaphraser`) — substitutes a
   financial term in the question with one of its ontology synonyms (the
   same ontology query expansion draws on, §3.1), testing whether retrieval
   is robust to the colloquial/formal vocabulary mismatch the expansion step
   is meant to close.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple

from ..data.finqa_loader import FinQAExample
from ..retrieval.query_expansion import DEFAULT_ONTOLOGY
from ..retrieval.temporal_filter import TemporalCompatibilityFilter

_NUMBER_RE = re.compile(r"\d[\d,.]*")


class DistractorInjector:
    """Builds a pool of off-period numeric sentences and injects one per example.

    Args:
        pool_examples: Examples to scavenge candidate distractor sentences
            from (typically a held-out split, distinct from the examples
            being perturbed, to avoid leaking the target's own correct
            evidence back in as a "distractor").
        seed: RNG seed for reproducible distractor selection.
    """

    def __init__(self, pool_examples: List[FinQAExample], seed: int = 13):
        self._rng = random.Random(seed)
        self._filter = TemporalCompatibilityFilter()
        self._sentences = self._build_pool(pool_examples)

    def _build_pool(self, examples: List[FinQAExample]) -> List[Tuple[str, set]]:
        pool: List[Tuple[str, set]] = []
        for ex in examples:
            for sent in ex.pre_text + ex.post_text:
                sent = sent.strip()
                if not sent or not _NUMBER_RE.search(sent):
                    continue
                periods = self._filter.extract_periods(sent)
                if periods:
                    pool.append((sent, {p.year for p in periods}))
        return pool

    def inject(
        self, example: FinQAExample
    ) -> Tuple[FinQAExample, Optional[str], Optional[int]]:
        """Append an off-period distractor sentence to ``example``.

        Returns ``(perturbed_example, distractor_sentence, distractor_year)``.
        If no distractor whose years don't overlap the example's own query
        periods can be found, returns the example unchanged with ``(None,
        None)`` so callers can skip it.
        """
        own_years = {p.year for p in self._filter.query_periods(example.question, example.table)}
        if not own_years:
            return example, None, None

        candidates = [
            (sent, years) for sent, years in self._sentences if years and not (years & own_years)
        ]
        if not candidates:
            return example, None, None

        sentence, years = self._rng.choice(candidates)
        perturbed = dc_replace(example, post_text=example.post_text + [sentence])
        return perturbed, sentence, min(years)


class SynonymParaphraser:
    """Rewrites a question by substituting a financial term with its ontology synonym."""

    def __init__(self, ontology: Optional[Dict[str, List[str]]] = None):
        self.ontology = ontology or DEFAULT_ONTOLOGY

    def paraphrase(self, question: str) -> Optional[str]:
        """Return a paraphrased question, or None if no ontology term is present."""
        for canonical, synonyms in self.ontology.items():
            if not synonyms:
                continue
            pattern = re.compile(r"\b" + re.escape(canonical) + r"\b", re.IGNORECASE)
            if pattern.search(question):
                return pattern.sub(synonyms[0], question, count=1)
        return None


def summarize_flip_rate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-example perturbation records into headline robustness metrics.

    Each record is expected to have boolean keys ``base_correct`` and
    ``perturbed_correct``. A "flip" is a base-correct example that becomes
    incorrect after perturbation — the metric that actually matters for
    robustness, since accuracy alone conflates pre-existing failures with
    perturbation-induced ones.
    """
    if not records:
        return {"n": 0, "base_accuracy": 0.0, "perturbed_accuracy": 0.0, "flip_rate": 0.0}

    n = len(records)
    base_correct = sum(1 for r in records if r["base_correct"])
    perturbed_correct = sum(1 for r in records if r["perturbed_correct"])
    flips = sum(1 for r in records if r["base_correct"] and not r["perturbed_correct"])

    return {
        "n": n,
        "base_accuracy": base_correct / n,
        "perturbed_accuracy": perturbed_correct / n,
        "flip_rate": flips / base_correct if base_correct else 0.0,
    }
