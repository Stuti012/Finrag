"""Temporal compatibility filtering for retrieved passages (FinTAG-RAG §3.3).

Implements the paper's core novelty: an explicit temporal-compatibility gate
placed *between* hybrid retrieval and symbolic reasoning. For a question ``q``
and a candidate passage ``d`` it computes

    T(d, q) = |T(d) ∩ T(q)| / |T(q)|

where ``T(·)`` is the set of fiscal periods (years / quarters) mentioned in the
text, and discards passages whose score falls below a threshold ``τ``. This
prevents *cross-year operand contamination* — answering a 2021 question with
2019 figures — which is the dominant source of numerical error in financial QA
when adjacent fiscal periods carry similar-looking operands.

Reference: Singh et al., "FinTAG-RAG: A Temporal-Aware Hybrid Retrieval and
Reasoning Framework for Robust Financial Question Answering", §3.3.

Design notes
------------
The paper specifies the score and the τ gate; the two recall-protecting
refinements below are required to make the gate help rather than hurt:

1. **Period-neutral passages are retained.** A passage with *no* temporal token
   (e.g. an accounting-policy definition, a row label) is valid across all
   periods and must not be discarded. Only passages that explicitly name a
   *competing* period are dropped.
2. **Query constraint falls back to the table.** When the question carries no
   explicit year (``"what was the change in revenue?"``), the relevant periods
   are taken from the table column headers — the implicit ``T(q)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# 4-digit fiscal years, 1900–2099.
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
# Explicit quarter mentions: "Q3", "Q3 2020", "third quarter", "3Q19".
_QUARTER_RE = re.compile(
    r"\bq([1-4])\b|\b([1-4])q\b|\b(first|second|third|fourth)\s+quarter\b",
    re.IGNORECASE,
)
# Two-digit fiscal-year shorthand: "FY19", "fiscal 2019", "FY2019".
_FY_SHORT_RE = re.compile(r"\bfy\s?'?(\d{2})\b", re.IGNORECASE)

_QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


@dataclass(frozen=True)
class FiscalPeriod:
    """A fiscal period at year and optional quarter granularity."""

    year: int
    quarter: Optional[int] = None

    def compatible_with(self, other: "FiscalPeriod") -> bool:
        """Two periods are compatible when years match and quarters don't conflict.

        A quarter-less period (annual figure) is compatible with any quarter of
        the same year, since the annual operand subsumes the quarterly one.
        """
        if self.year != other.year:
            return False
        if self.quarter is None or other.quarter is None:
            return True
        return self.quarter == other.quarter


@dataclass
class FilterDiagnostics:
    """Per-call diagnostics, useful for the context-filtering metrics in §4.1."""

    applied: bool = False
    query_periods: List[int] = field(default_factory=list)
    kept: int = 0
    discarded: int = 0
    neutral_kept: int = 0
    mean_compatibility: float = 0.0
    discarded_periods: List[int] = field(default_factory=list)

    @property
    def noise_reduction(self) -> float:
        """Fraction of period-bearing passages removed as off-period noise."""
        total = self.kept + self.discarded
        return self.discarded / total if total else 0.0


class TemporalCompatibilityFilter:
    """Gate retrieved passages by fiscal-period overlap with the query.

    Args:
        tau: Compatibility threshold τ in [0, 1]. Passages scoring below τ are
            discarded. The paper's reported configuration uses τ = 0.5, i.e. a
            passage must cover at least half of the query's fiscal periods.
        keep_neutral: When True (default), passages with no detectable period are
            retained (see design note 1). Set False for a strict, paper-literal
            gate that drops everything below τ including neutral passages.
        min_keep: Always retain at least this many top-ranked passages, even if
            the gate would empty the list. Guards against destroying recall on
            noisy temporal extraction.
    """

    def __init__(self, tau: float = 0.5, keep_neutral: bool = True, min_keep: int = 1):
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.tau = tau
        self.keep_neutral = keep_neutral
        self.min_keep = max(0, min_keep)

    # ── period extraction: T(·) ───────────────────────────────────────────────

    def extract_periods(self, text: str) -> Set[FiscalPeriod]:
        """Extract the set of fiscal periods T(d) mentioned in a span of text."""
        if not text:
            return set()

        years: Set[int] = {int(y) for y in _YEAR_RE.findall(text)}
        for two in _FY_SHORT_RE.findall(text):
            # FY19 → 2019, FY98 → 1998 (pivot at 50 like POSIX strptime %y).
            n = int(two)
            years.add(2000 + n if n < 50 else 1900 + n)

        quarters: Set[int] = set()
        for q_digit, digit_q, word in _QUARTER_RE.findall(text):
            if q_digit:
                quarters.add(int(q_digit))
            elif digit_q:
                quarters.add(int(digit_q))
            elif word:
                quarters.add(_QUARTER_WORDS[word.lower()])

        periods: Set[FiscalPeriod] = set()
        if years and quarters:
            for y in years:
                for q in quarters:
                    periods.add(FiscalPeriod(y, q))
        elif years:
            periods.update(FiscalPeriod(y) for y in years)
        # Quarters with no year are too ambiguous to anchor; ignored.
        return periods

    def query_periods(
        self, question: str, table: Optional[List[List[str]]] = None
    ) -> Set[FiscalPeriod]:
        """Build the query constraint T(q), falling back to the table headers.

        If the question names explicit periods, those are T(q). Otherwise the
        table's column-header years stand in as the implicit constraint.
        """
        q_periods = self.extract_periods(question or "")
        if q_periods:
            return q_periods

        if table and table[0]:
            header_text = " ".join(str(c) for c in table[0])
            return self.extract_periods(header_text)
        return set()

    # ── compatibility score: T(d, q) ──────────────────────────────────────────

    def compatibility(
        self, passage: str, q_periods: Set[FiscalPeriod]
    ) -> Tuple[float, bool]:
        """Compute T(d, q) for one passage.

        Returns:
            (score, is_neutral). ``score`` is |T(d)∩T(q)| / |T(q)|; ``is_neutral``
            is True when the passage carries no period at all.
        """
        if not q_periods:
            return 1.0, False  # no constraint → everything compatible

        d_periods = self.extract_periods(passage)
        if not d_periods:
            return 0.0, True

        matched = {
            qp for qp in q_periods if any(dp.compatible_with(qp) for dp in d_periods)
        }
        return len(matched) / len(q_periods), False

    # ── the gate ──────────────────────────────────────────────────────────────

    def filter(
        self,
        passages: List[Dict[str, Any]],
        question: str,
        table: Optional[List[List[str]]] = None,
        text_key: str = "text",
    ) -> Tuple[List[Dict[str, Any]], FilterDiagnostics]:
        """Apply the temporal-compatibility gate to a list of retrieved passages.

        Each passage is annotated in-place with ``temporal_compatibility`` (its
        T(d,q) score) and ``temporal_neutral`` (bool). Returns the surviving
        passages in their original order plus a :class:`FilterDiagnostics`.

        Passages whose score < τ are discarded, except period-neutral passages
        when ``keep_neutral`` is set. At least ``min_keep`` top-ranked passages
        are always returned.
        """
        diag = FilterDiagnostics()
        if not passages:
            return passages, diag

        q_periods = self.query_periods(question, table)
        diag.query_periods = sorted({p.year for p in q_periods})

        if not q_periods:
            # No determinable constraint — gate is a no-op, annotate and return.
            for p in passages:
                p["temporal_compatibility"] = 1.0
                p["temporal_neutral"] = False
            diag.kept = len(passages)
            diag.mean_compatibility = 1.0
            return passages, diag

        diag.applied = True
        kept: List[Dict[str, Any]] = []
        scores: List[float] = []
        for p in passages:
            score, is_neutral = self.compatibility(p.get(text_key, ""), q_periods)
            p["temporal_compatibility"] = round(score, 4)
            p["temporal_neutral"] = is_neutral

            if is_neutral and self.keep_neutral:
                kept.append(p)
                diag.neutral_kept += 1
                continue
            if score >= self.tau:
                kept.append(p)
                scores.append(score)
            else:
                diag.discarded += 1
                diag.discarded_periods.extend(
                    sorted({fp.year for fp in self.extract_periods(p.get(text_key, ""))})
                )

        diag.kept = len(kept)
        diag.mean_compatibility = sum(scores) / len(scores) if scores else 0.0
        diag.discarded_periods = sorted(set(diag.discarded_periods))

        # Recall guard: never empty the list below min_keep.
        if len(kept) < self.min_keep:
            for p in passages:
                if p not in kept:
                    kept.append(p)
                    diag.kept += 1
                    diag.discarded = max(0, diag.discarded - 1)
                if len(kept) >= self.min_keep:
                    break
            # Preserve original ordering of the salvaged set.
            kept = [p for p in passages if p in kept]

        return kept, diag
