"""Query Processing Module (thesis Section 3.5.1).

Expands a raw financial query with:
  - financial ontology synonyms/abbreviations         -> E(q) term expansion
  - entity normalization (company name variants)
  - temporal expression standardization (explicit + a best-effort flag for
    implicit references such as "last year", which Chapter 5's error analysis
    identifies as the framework's largest failure category -- we surface
    these explicitly rather than silently guessing a fiscal year)
  - numeric operator detection, used to select the symbolic-reasoning
    operation in Section 3.5.5

This mirrors Algorithm 1 in the thesis (query -> enriched query Q').
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .data import extract_fiscal_years

# A compact but representative financial ontology: synonym groups map every
# variant to one canonical term, and abbreviations expand to their full form.
# (The thesis reports 847 terms / 312 abbreviations / 156 synonym groups
# manually curated from the FinQA training corpus; this is a curated subset
# covering the vocabulary that actually appears in FinQA questions.)
SYNONYM_GROUPS: List[List[str]] = [
    ["net income", "net earnings", "profit after tax", "bottom line"],
    ["revenue", "net sales", "total sales", "turnover", "top line"],
    ["operating income", "operating profit", "ebit"],
    ["gross profit", "gross margin dollars"],
    ["total assets", "assets"],
    ["total liabilities", "liabilities"],
    ["shareholders equity", "stockholders equity", "total equity", "book value"],
    ["cash flow from operations", "operating cash flow", "cfo"],
    ["capital expenditures", "capex", "capital spending"],
    ["earnings per share", "eps"],
    ["dividend per share", "dps"],
    ["cost of goods sold", "cogs", "cost of sales", "cost of revenue"],
    ["research and development", "r&d", "rd expense"],
    ["selling general and administrative", "sg&a", "sga expense"],
    ["depreciation and amortization", "d&a"],
    ["long term debt", "long-term debt", "ltd"],
    ["free cash flow", "fcf"],
    ["weighted average shares outstanding", "diluted shares", "shares outstanding"],
]

ABBREVIATIONS: Dict[str, str] = {
    "eps": "earnings per share",
    "ebitda": "earnings before interest taxes depreciation and amortization",
    "capex": "capital expenditures",
    "sg&a": "selling general and administrative",
    "sga": "selling general and administrative",
    "r&d": "research and development",
    "cogs": "cost of goods sold",
    "d&a": "depreciation and amortization",
    "roe": "return on equity",
    "roa": "return on assets",
    "yoy": "year over year",
    "qoq": "quarter over quarter",
    "cagr": "compound annual growth rate",
    "fcf": "free cash flow",
    "dps": "dividend per share",
    "ltd": "long term debt",
}

# Operation keyword -> canonical operation name consumed by the symbolic
# reasoning module (Section 3.5.5).
OPERATION_KEYWORDS: Dict[str, List[str]] = {
    "percentage_change": [
        "percentage increase", "percentage decrease", "percent increase", "percent decrease",
        "percentage change", "percent change", "growth rate", "growth in", "% increase",
        "% change", "increase of", "decrease of",
    ],
    "ratio": ["ratio of", "ratio between", "proportion of", "as a percentage of", "margin"],
    "difference": ["difference between", "difference in", "how much more", "how much less", "change in"],
    "sum": ["total of", "combined", "sum of", "in aggregate"],
    "average": ["average of", "mean of"],
}

IMPLICIT_TEMPORAL_PATTERNS = [
    "last year", "the prior year", "previous year", "the current year",
    "this year", "last quarter", "the prior quarter", "previous quarter",
    "this quarter", "the most recent", "year over year", "the following year",
]


@dataclass
class EnrichedQuery:
    original: str
    normalized: str
    expanded_terms: List[str] = field(default_factory=list)
    explicit_years: List[int] = field(default_factory=list)
    implicit_temporal_refs: List[str] = field(default_factory=list)
    operation_hint: Optional[str] = None

    @property
    def has_unresolved_temporal_reference(self) -> bool:
        return bool(self.implicit_temporal_refs) and not self.explicit_years

    def expanded_text(self) -> str:
        """q' = q u E(q): the original query plus its expansion terms."""
        return " ".join([self.normalized] + self.expanded_terms)


class QueryProcessor:
    """Implements Algorithm 1 (Query Processing) from the thesis."""

    def __init__(self):
        self._synonym_lookup: Dict[str, List[str]] = {}
        for group in SYNONYM_GROUPS:
            for term in group:
                others = [t for t in group if t != term]
                self._synonym_lookup[term] = others

    @staticmethod
    def normalize(query: str) -> str:
        q = query.strip().lower()
        q = re.sub(r"\s+", " ", q)
        return q

    def _expand_ontology(self, normalized: str) -> List[str]:
        expansions: List[str] = []
        for term, synonyms in self._synonym_lookup.items():
            if term in normalized:
                expansions.extend(synonyms)
        for abbr, full in ABBREVIATIONS.items():
            pattern = r"\b" + re.escape(abbr) + r"\b"
            if re.search(pattern, normalized):
                expansions.append(full)
            elif full in normalized:
                expansions.append(abbr)
        # de-duplicate while preserving order
        seen = set()
        out = []
        for e in expansions:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def _detect_operation(self, normalized: str) -> Optional[str]:
        for op, phrases in OPERATION_KEYWORDS.items():
            for phrase in phrases:
                if phrase in normalized:
                    return op
        return None

    def _detect_implicit_temporal(self, normalized: str) -> List[str]:
        return [p for p in IMPLICIT_TEMPORAL_PATTERNS if p in normalized]

    def process(self, query: str) -> EnrichedQuery:
        normalized = self.normalize(query)
        return EnrichedQuery(
            original=query,
            normalized=normalized,
            expanded_terms=self._expand_ontology(normalized),
            explicit_years=extract_fiscal_years(query),
            implicit_temporal_refs=self._detect_implicit_temporal(normalized),
            operation_hint=self._detect_operation(normalized),
        )
