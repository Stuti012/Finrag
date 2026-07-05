"""Symbolic Reasoning Module (thesis Section 3.5.5, Algorithm 5).

Separates operand extraction from arithmetic execution so financial answers
are computed deterministically (Decimal arithmetic) rather than through
probabilistic language-model generation -- the central claim of the thesis
(Section 3.6.4): "Symbolic Reasoning over Neural Arithmetic".

Every computation records a full audit trail: which fact was used, from
which source chunk, and what operation was applied -- matching the
"verifiable reasoning traces" described in the Introduction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import List, Optional, Tuple

from .config import FinTAGRAGConfig
from .data import Fact
from .ontology import EnrichedQuery
from .retrieval import RetrievedChunk

_RELATIVE_PRIOR = {"last year", "the prior year", "previous year", "prior year"}
_RELATIVE_NEXT = {"the following year", "next year"}


@dataclass
class OperandMatch:
    fact: Fact
    source_chunk_id: str
    match_score: float


@dataclass
class SymbolicResult:
    success: bool
    operation: Optional[str]
    value: Optional[float] = None
    operands: List[OperandMatch] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    error: Optional[str] = None
    heuristic_temporal_resolution: bool = False


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-z]+", a.lower()))
    b_tokens = set(re.findall(r"[a-z]+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def resolve_comparison_years(enriched_query: EnrichedQuery) -> Tuple[List[int], bool]:
    """Best-effort resolution of a two-point comparison's fiscal years.

    If the query names both years explicitly, use them as-is. If it names
    only one year and uses a relative phrase ("last year", "the following
    year"), heuristically infer the second year as +/-1. This addresses the
    single largest error category identified in the thesis's error analysis
    (Section 5.7, ~40% of failures: unresolved implicit temporal references)
    -- but the heuristic is always flagged so it is never silently trusted.
    """
    years = list(enriched_query.explicit_years)
    if len(years) >= 2:
        return sorted(years)[:2], False
    if len(years) == 1:
        refs = set(enriched_query.implicit_temporal_refs)
        if refs & _RELATIVE_PRIOR:
            return sorted([years[0] - 1, years[0]]), True
        if refs & _RELATIVE_NEXT:
            return sorted([years[0], years[0] + 1]), True
    return years, False


class SymbolicReasoner:
    """Implements Algorithm 5: operand extraction + deterministic arithmetic."""

    def __init__(self, config: FinTAGRAGConfig = None):
        self.config = config or FinTAGRAGConfig()

    @staticmethod
    def _collect_facts(chunks: List[RetrievedChunk]) -> List[Tuple[Fact, str]]:
        out = []
        for r in chunks:
            for f in r.chunk.facts:
                out.append((f, r.chunk.chunk_id))
        return out

    @staticmethod
    def _best_match_for_year(
        facts_with_src: List[Tuple[Fact, str]], query_text: str, year: Optional[int]
    ) -> Optional[OperandMatch]:
        candidates = [(f, cid) for f, cid in facts_with_src if f.year == year]
        if not candidates:
            return None
        scored = sorted(candidates, key=lambda fc: -_token_overlap(query_text, fc[0].metric))
        best_fact, best_cid = scored[0]
        return OperandMatch(fact=best_fact, source_chunk_id=best_cid, match_score=_token_overlap(query_text, best_fact.metric))

    def reason(self, enriched_query: EnrichedQuery, chunks: List[RetrievedChunk]) -> SymbolicResult:
        facts_with_src = self._collect_facts(chunks)
        if not facts_with_src:
            return SymbolicResult(
                success=False,
                operation=enriched_query.operation_hint,
                trace=["No numeric facts found in the temporally-filtered context."],
                error="insufficient_data",
            )

        query_text = enriched_query.expanded_text()
        op = enriched_query.operation_hint
        trace: List[str] = []

        if op in ("percentage_change", "difference", "ratio"):
            years, heuristic = resolve_comparison_years(enriched_query)
            if len(years) < 2:
                return SymbolicResult(
                    success=False, operation=op,
                    trace=["Comparison operation requires two fiscal years; only one (or none) could be resolved."],
                    error="unresolved_temporal_reference",
                )
            y_old, y_new = years[0], years[1]
            old_match = self._best_match_for_year(facts_with_src, query_text, y_old)
            new_match = self._best_match_for_year(facts_with_src, query_text, y_new)
            if not old_match or not new_match:
                missing = [str(y) for y, m in ((y_old, old_match), (y_new, new_match)) if not m]
                return SymbolicResult(
                    success=False, operation=op,
                    trace=trace + [f"Missing operand(s) for fiscal year(s): {', '.join(missing)}"],
                    error="insufficient_data", heuristic_temporal_resolution=heuristic,
                )
            try:
                old_v, new_v = Decimal(str(old_match.fact.value)), Decimal(str(new_match.fact.value))
            except InvalidOperation:
                return SymbolicResult(success=False, operation=op, error="invalid_operand", trace=trace)

            trace.append(f"operand_old = {old_match.fact.metric} ({y_old}) = {old_v}  [source: {old_match.source_chunk_id}]")
            trace.append(f"operand_new = {new_match.fact.metric} ({y_new}) = {new_v}  [source: {new_match.source_chunk_id}]")
            try:
                if op == "percentage_change":
                    result = (new_v - old_v) / old_v * Decimal("100")
                    trace.append(f"result = (new - old) / old * 100 = {result}")
                elif op == "difference":
                    result = new_v - old_v
                    trace.append(f"result = new - old = {result}")
                else:  # ratio
                    result = new_v / old_v
                    trace.append(f"result = new / old = {result}")
            except (DivisionByZero, InvalidOperation, ZeroDivisionError):
                return SymbolicResult(
                    success=False, operation=op, operands=[old_match, new_match],
                    trace=trace + ["Division by zero: base-year operand is 0."],
                    error="division_by_zero", heuristic_temporal_resolution=heuristic,
                )
            return SymbolicResult(
                success=True, operation=op, value=float(result),
                operands=[old_match, new_match], trace=trace, heuristic_temporal_resolution=heuristic,
            )

        if op in ("sum", "average"):
            years = enriched_query.explicit_years or sorted({f.year for f, _ in facts_with_src if f.year is not None})
            matches = [m for y in years if (m := self._best_match_for_year(facts_with_src, query_text, y))]
            if not matches:
                return SymbolicResult(success=False, operation=op, trace=["No matching operands found."], error="insufficient_data")
            values = [Decimal(str(m.fact.value)) for m in matches]
            for m in matches:
                trace.append(f"operand = {m.fact.metric} ({m.fact.year}) = {m.fact.value}  [source: {m.source_chunk_id}]")
            if op == "sum":
                result = sum(values)
                trace.append(f"result = sum(operands) = {result}")
            else:
                result = sum(values) / Decimal(len(values))
                trace.append(f"result = mean(operands) = {result}")
            return SymbolicResult(success=True, operation=op, value=float(result), operands=matches, trace=trace)

        # Default: direct single-value lookup (no comparison operation detected).
        target_year = enriched_query.explicit_years[0] if enriched_query.explicit_years else None
        match = self._best_match_for_year(facts_with_src, query_text, target_year) if target_year is not None else None
        if match is None:
            scored = sorted(facts_with_src, key=lambda fc: -_token_overlap(query_text, fc[0].metric))
            if scored and _token_overlap(query_text, scored[0][0].metric) > 0:
                f, cid = scored[0]
                match = OperandMatch(fact=f, source_chunk_id=cid, match_score=_token_overlap(query_text, f.metric))
        if match is None:
            return SymbolicResult(success=False, operation="lookup", trace=["No matching operand found for a direct lookup."], error="insufficient_data")
        trace.append(f"operand = {match.fact.metric} ({match.fact.year}) = {match.fact.value}  [source: {match.source_chunk_id}]")
        return SymbolicResult(success=True, operation="lookup", value=match.fact.value, operands=[match], trace=trace)
