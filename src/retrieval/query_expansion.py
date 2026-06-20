"""Ontology-based query expansion (FinTAG-RAG §3.1).

Implements the paper's retrieval-entry-point step

    q' = q ∪ E(q)

where ``E(q)`` is the set of domain-synonym terms triggered by financial
vocabulary already present in the query ``q``. Financial questions routinely
use a metric's colloquial form ("sales") while the source filing uses the
formal GAAP/IFRS term ("net revenue"), or vice versa — sparse (BM25) retrieval
in particular misses these passages entirely since it matches on surface
tokens. Expanding the query with the metric's synonym set closes that gap
without touching the dense retriever's embedding space.

Reference: Singh et al., "FinTAG-RAG: A Temporal-Aware Hybrid Retrieval and
Reasoning Framework for Robust Financial Question Answering", §3.1.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

# Canonical financial metric → its synonym/abbreviation surface forms.
# Each group (canonical + synonyms) is treated as mutually substitutable: any
# member appearing in the query triggers expansion with the other members.
DEFAULT_ONTOLOGY: Dict[str, List[str]] = {
    "revenue": ["sales", "net sales", "turnover", "total revenue"],
    "net income": ["net earnings", "net profit", "profit after tax"],
    "operating income": ["operating profit", "ebit"],
    "gross profit": ["gross margin"],
    "cost of goods sold": ["cogs", "cost of sales", "cost of revenue"],
    "operating expenses": ["opex"],
    "research and development": ["r&d", "r & d"],
    "selling general and administrative": ["sg&a", "sga"],
    "depreciation and amortization": ["d&a", "depreciation", "amortization"],
    "earnings per share": ["eps"],
    "ebitda": ["earnings before interest taxes depreciation and amortization"],
    "total assets": ["assets"],
    "total liabilities": ["liabilities"],
    "shareholders equity": ["stockholders equity", "stockholders' equity", "net assets"],
    "cash and cash equivalents": ["cash equivalents"],
    "accounts receivable": ["receivables", "trade receivables"],
    "accounts payable": ["payables", "trade payables"],
    "capital expenditures": ["capex", "capital spending"],
    "free cash flow": ["fcf"],
    "long term debt": ["long-term debt", "long term borrowings"],
    "dividend": ["dividends paid", "dividend payout"],
    "increase": ["rose", "grew", "increase", "growth", "rise"],
    "decrease": ["declined", "fell", "decrease", "drop", "reduction"],
    "percentage change": ["pct change", "% change", "percent change"],
}


class QueryExpander:
    """Expand a question with ontology-driven financial synonym terms.

    Args:
        ontology: Mapping of canonical metric name → list of synonyms. Falls
            back to :data:`DEFAULT_ONTOLOGY` (general financial-statement
            vocabulary) when omitted.
        max_terms: Cap on the number of expansion terms appended, so a query
            touching many ontology groups doesn't balloon past what the
            retriever can usefully weight.
    """

    def __init__(
        self,
        ontology: Optional[Dict[str, List[str]]] = None,
        max_terms: int = 6,
    ):
        self.ontology = ontology or DEFAULT_ONTOLOGY
        self.max_terms = max_terms
        self._groups: List[Set[str]] = [
            {canonical, *synonyms} for canonical, synonyms in self.ontology.items()
        ]
        self._patterns = [
            (
                group,
                re.compile(
                    r"\b(" + "|".join(re.escape(t) for t in sorted(group, key=len, reverse=True)) + r")\b",
                    re.IGNORECASE,
                ),
            )
            for group in self._groups
        ]

    def expand_terms(self, query: str) -> List[str]:
        """Compute E(q): the ontology terms to add, in trigger order, capped at max_terms."""
        if not query:
            return []

        terms: List[str] = []
        seen: Set[str] = set()
        for group, pattern in self._patterns:
            if not pattern.search(query):
                continue
            for term in group:
                term_lower = term.lower()
                if term_lower in seen:
                    continue
                if re.search(r"\b" + re.escape(term_lower) + r"\b", query.lower()):
                    continue  # already present verbatim, nothing to add
                seen.add(term_lower)
                terms.append(term)
                if len(terms) >= self.max_terms:
                    return terms
        return terms

    def expand(self, query: str) -> str:
        """Build q' = q ∪ E(q) as a single string for downstream retrieval."""
        terms = self.expand_terms(query)
        if not terms:
            return query
        return f"{query} {' '.join(terms)}"
