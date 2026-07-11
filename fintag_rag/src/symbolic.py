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

_CHANGE_INDICATOR_WORDS = {
    "increase", "increased", "decrease", "decreased", "change", "changed",
    "variance", "growth", "decline", "declined", "increases", "decreases",
}
_MIN_STATED_VALUE_MATCH_SCORE = 0.15


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
    tier_used: Optional[str] = None


_MATCH_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "what", "which", "that", "this", "as", "by", "from",
}


def _content_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9&]+", text.lower()) if t not in _MATCH_STOPWORDS]


def _metric_similarity(query_text: str, metric: str, fact_kind: str = "table") -> float:
    """Score how well a candidate operand's metric label matches the query.

    Plain bag-of-words Jaccard overlap treats every shared word equally, so a
    noisy text-derived fact like "paid in two
    installments on the" can coincidentally out-score the actual correct table
    row on nothing but stopword-adjacent overlap. This instead heavily rewards
    the metric appearing as a contiguous phrase inside the query (e.g. "net
    revenue" as a 2-gram is a far stronger signal than two isolated word
    matches), and gives a small preference to clean, structured table facts
    over noisier sentence-extracted ones when scores are otherwise close.
    """
    q_tokens = _content_tokens(query_text)
    m_tokens = _content_tokens(metric)
    if not q_tokens or not m_tokens:
        return 0.0
    q_set, m_set = set(q_tokens), set(m_tokens)
    union = q_set | m_set
    jaccard = len(q_set & m_set) / len(union) if union else 0.0

    q_joined = " ".join(q_tokens)
    best_run = 0
    for i in range(len(m_tokens)):
        for j in range(i + 1, len(m_tokens) + 1):
            run_len = j - i
            if run_len <= best_run:
                continue
            if " ".join(m_tokens[i:j]) in q_joined:
                best_run = run_len
    phrase_bonus = min(0.6, best_run * 0.2)

    kind_bonus = 0.05 if fact_kind == "table" else 0.0
    return jaccard + phrase_bonus + kind_bonus


def resolve_comparison_years(enriched_query: EnrichedQuery, default_to_prior_year: bool = True) -> Tuple[List[int], bool]:
    """Best-effort resolution of a two-point comparison's fiscal years.

    If the query names both years explicitly, use them as-is. If it names
    only one year and uses a relative phrase ("last year", "the following
    year"), heuristically infer the second year as +/-1. This addresses the
    single largest error category identified in the thesis's error analysis
    (Section 5.7, ~40% of failures: unresolved implicit temporal references)
    -- but the heuristic is always flagged so it is never silently trusted.

    When `default_to_prior_year` is True (the default), a single named year
    with no relative phrase at all still falls back to comparing it against
    the immediately preceding year -- "year-over-year" is by far the most
    common comparison basis in FinQA-style questions, so this is a reasonable
    last resort rather than giving up outright. It is still flagged as a
    heuristic in the return value.
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
        if default_to_prior_year:
            return sorted([years[0] - 1, years[0]]), True
    return years, False


class SymbolicReasoner:
    """Implements Algorithm 5: operand extraction + deterministic arithmetic."""

    def __init__(self, config: FinTAGRAGConfig = None):
        self.config = config or FinTAGRAGConfig()

    @staticmethod
    def _collect_facts(chunks: List[RetrievedChunk]) -> List[Tuple[Fact, RetrievedChunk]]:
        out = []
        for r in chunks:
            for f in r.chunk.facts:
                out.append((f, r))
        return out

    @staticmethod
    def _restrict_to_entity(
        facts_with_src: List[Tuple[Fact, "RetrievedChunk"]], entity_phrase: Optional[str]
    ) -> List[Tuple[Fact, "RetrievedChunk"]]:
        """Restrict candidates to those whose source chunk names the query's
        entity (e.g. "entergy corporation"), when any do. Without this, an
        operand-matching pass over a wide fallback tier (many companies'
        chunks at once) can match a same-named metric from a completely
        different company -- e.g. picking up Lockheed Martin's "net sales"
        for a question about Entergy, since both happen to report a line
        item with that name. Falls back to the unrestricted candidate list
        only if none of them actually name the entity (so this never causes
        a total failure by itself)."""
        if not entity_phrase:
            return facts_with_src
        matching = [(f, rc) for f, rc in facts_with_src if entity_phrase in rc.chunk.text.lower()]
        return matching or facts_with_src

    @staticmethod
    def _best_match_for_year(
        facts_with_src: List[Tuple[Fact, RetrievedChunk]], query_text: str, year: Optional[int],
        entity_phrase: Optional[str] = None,
    ) -> Optional[OperandMatch]:
        candidates = [(f, rc) for f, rc in facts_with_src if f.year == year]
        candidates = SymbolicReasoner._restrict_to_entity(candidates, entity_phrase)
        if not candidates:
            return None
        scored = sorted(candidates, key=lambda fc: -_metric_similarity(query_text, fc[0].metric, fc[0].kind))
        best_fact, best_rc = scored[0]
        return OperandMatch(
            fact=best_fact, source_chunk_id=best_rc.chunk.chunk_id,
            match_score=_metric_similarity(query_text, best_fact.metric, best_fact.kind),
        )

    @staticmethod
    def _best_match_any_year(
        facts_with_src: List[Tuple[Fact, RetrievedChunk]], query_text: str, exclude: Optional[Fact] = None,
        entity_phrase: Optional[str] = None,
    ) -> Optional[OperandMatch]:
        """Best metric match regardless of year -- used for part/whole ratio
        questions ("what percentage of X is Y") where the two operands are
        different metrics, not the same metric in two different years."""
        candidates = [(f, rc) for f, rc in facts_with_src if f is not exclude]
        candidates = SymbolicReasoner._restrict_to_entity(candidates, entity_phrase)
        if not candidates:
            return None
        scored = sorted(candidates, key=lambda fc: -_metric_similarity(query_text, fc[0].metric, fc[0].kind))
        best_fact, best_rc = scored[0]
        score = _metric_similarity(query_text, best_fact.metric, best_fact.kind)
        if score <= 0:
            return None
        return OperandMatch(fact=best_fact, source_chunk_id=best_rc.chunk.chunk_id, match_score=score)

    def _two_endpoint_result(
        self, op: str, facts_with_src: List[Tuple[Fact, RetrievedChunk]], query_text: str, enriched_query: EnrichedQuery
    ) -> Optional[SymbolicResult]:
        """Reconstruct percentage_change/difference/ratio from two matched
        endpoint operands. Returns None (not a failed SymbolicResult) when it
        can't, so the caller can fall back to another strategy."""
        years, heuristic = resolve_comparison_years(enriched_query)
        if len(years) < 2:
            return None
        y_old, y_new = years[0], years[1]
        entity_phrase = enriched_query.entity_phrase
        old_match = self._best_match_for_year(facts_with_src, query_text, y_old, entity_phrase)
        new_match = self._best_match_for_year(facts_with_src, query_text, y_new, entity_phrase)
        if not old_match or not new_match:
            return None
        try:
            old_v, new_v = Decimal(str(old_match.fact.value)), Decimal(str(new_match.fact.value))
        except InvalidOperation:
            return None

        trace = [
            f"operand_old = {old_match.fact.metric} ({y_old}) = {old_v}  [source: {old_match.source_chunk_id}]",
            f"operand_new = {new_match.fact.metric} ({y_new}) = {new_v}  [source: {new_match.source_chunk_id}]",
        ]
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

    @staticmethod
    def _find_stated_change_value(
        facts_with_src: List[Tuple[Fact, RetrievedChunk]], query_text: str, query_years: List[int],
        entity_phrase: Optional[str] = None,
    ) -> Optional[OperandMatch]:
        """Many FinQA "what was the change/growth in X during Y" questions have
        the delta already stated directly in the narrative (e.g. "net revenue
        increased $94 million"), rather than requiring two endpoint values to
        be located and subtracted. Prefer a clearly-labeled, well-matched
        stated value over reconstructing the delta ourselves when one exists."""
        facts_with_src = SymbolicReasoner._restrict_to_entity(facts_with_src, entity_phrase)
        candidates = []
        for f, rc in facts_with_src:
            metric_words = set(re.findall(r"[a-z]+", f.metric))
            if not (metric_words & _CHANGE_INDICATOR_WORDS):
                continue
            if query_years and f.year is not None and f.year not in query_years:
                continue
            score = _metric_similarity(query_text, f.metric, f.kind)
            if score > 0:
                candidates.append((score, f, rc))
        if not candidates:
            return None
        candidates.sort(key=lambda c: -c[0])
        best_score, best_fact, best_rc = candidates[0]
        if best_score < _MIN_STATED_VALUE_MATCH_SCORE:
            return None
        return OperandMatch(fact=best_fact, source_chunk_id=best_rc.chunk.chunk_id, match_score=best_score)

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

        if op == "ratio" and enriched_query.ratio_phrase:
            # Part/whole ratio question ("what percentage of X is Y"): the two
            # operands are different metrics in the (usually) same period, not
            # the same metric across two years -- match each phrase against
            # its own best-scoring fact rather than filtering by year.
            whole_phrase, part_phrase = enriched_query.ratio_phrase
            whole_match = self._best_match_any_year(facts_with_src, whole_phrase, entity_phrase=enriched_query.entity_phrase)
            part_match = self._best_match_any_year(
                facts_with_src, part_phrase, exclude=whole_match.fact if whole_match else None,
                entity_phrase=enriched_query.entity_phrase,
            )
            if whole_match and part_match:
                try:
                    whole_v = Decimal(str(whole_match.fact.value))
                    part_v = Decimal(str(part_match.fact.value))
                    result = part_v / whole_v
                except (DivisionByZero, InvalidOperation, ZeroDivisionError):
                    return SymbolicResult(
                        success=False, operation="ratio", operands=[whole_match, part_match],
                        trace=[f'Division by zero: "{whole_phrase}" resolved to 0.'], error="division_by_zero",
                    )
                trace.append(
                    f"operand_whole = {whole_match.fact.metric} ({whole_match.fact.year}) = {whole_v}  "
                    f"[source: {whole_match.source_chunk_id}]  (matched phrase: \"{whole_phrase}\")"
                )
                trace.append(
                    f"operand_part = {part_match.fact.metric} ({part_match.fact.year}) = {part_v}  "
                    f"[source: {part_match.source_chunk_id}]  (matched phrase: \"{part_phrase}\")"
                )
                trace.append(f"result = part / whole = {result}")
                return SymbolicResult(success=True, operation="ratio", value=float(result), operands=[whole_match, part_match], trace=trace)

        if op in ("percentage_change", "difference", "ratio"):
            # Prefer reconstructing the result from two clean, structured
            # endpoint values (deterministic and verifiable) over the noisier
            # "directly stated in the narrative" shortcut below -- try this
            # first and only fall back to the narrative shortcut if it fails.
            endpoint_result = self._two_endpoint_result(op, facts_with_src, query_text, enriched_query)
            if endpoint_result is not None and endpoint_result.success:
                return endpoint_result

            if op in ("percentage_change", "difference"):
                stated = self._find_stated_change_value(
                    facts_with_src, query_text, enriched_query.explicit_years, enriched_query.entity_phrase
                )
                if stated is not None:
                    trace.append(
                        f"operand_stated_change = {stated.fact.metric} ({stated.fact.year}) = {stated.fact.value}  "
                        f"[source: {stated.source_chunk_id}]  (directly stated in evidence, not recomputed; "
                        "used as a fallback because a clean table-based computation was not available)"
                    )
                    return SymbolicResult(success=True, operation="stated_value", value=stated.fact.value, operands=[stated], trace=trace)

            if endpoint_result is not None:
                return endpoint_result  # a specific failure (e.g. division by zero) beats a generic one

            return SymbolicResult(
                success=False, operation=op,
                trace=["Could not resolve two comparable operands, and no directly-stated value was found."],
                error="insufficient_data",
            )

        if op in ("sum", "average"):
            years = enriched_query.explicit_years or sorted({f.year for f, _ in facts_with_src if f.year is not None})
            matches = [
                m for y in years
                if (m := self._best_match_for_year(facts_with_src, query_text, y, enriched_query.entity_phrase))
            ]
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
        match = (
            self._best_match_for_year(facts_with_src, query_text, target_year, enriched_query.entity_phrase)
            if target_year is not None else None
        )
        if match is None:
            match = self._best_match_any_year(facts_with_src, query_text, entity_phrase=enriched_query.entity_phrase)
        if match is None:
            return SymbolicResult(success=False, operation="lookup", trace=["No matching operand found for a direct lookup."], error="insufficient_data")
        trace.append(f"operand = {match.fact.metric} ({match.fact.year}) = {match.fact.value}  [source: {match.source_chunk_id}]")
        return SymbolicResult(success=True, operation="lookup", value=match.fact.value, operands=[match], trace=trace)

    def reason_with_fallback(
        self, enriched_query: EnrichedQuery, tiers: List[Tuple[str, List[RetrievedChunk]]]
    ) -> SymbolicResult:
        """Try progressively wider evidence sets until one yields an answer.

        Context filtering and temporal filtering are deliberately strict (that
        is the point -- Section 3.5.3/3.5.4), but a strict filter can
        occasionally exclude the one chunk that actually carries the answer.
        Rather than reporting "insufficient data" the moment the narrowest,
        most-trusted tier comes up empty, retry against progressively wider
        (less filtered) tiers, recording which tier ultimately succeeded so
        the trade-off stays visible rather than silently swept under a single
        "full mode" result.
        """
        last_result: Optional[SymbolicResult] = None
        for tier_name, chunks in tiers:
            if not chunks:
                continue
            result = self.reason(enriched_query, chunks)
            result.tier_used = tier_name
            last_result = result
            if result.success:
                return result
        return last_result or SymbolicResult(
            success=False, operation=enriched_query.operation_hint,
            trace=["No evidence available at any retrieval tier."], error="insufficient_data",
        )
