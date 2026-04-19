"""Temporal Reasoning Module for financial QA.

Handles time-based reasoning including:
- Temporal ordering of financial events
- Trend detection across reporting periods
- Implicit temporal reference resolution
- Temporal graph construction for multi-hop reasoning
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..utils.financial_utils import extract_years_from_text, parse_financial_number


class TemporalEntity:
    """Represents a temporal entity (time point, period, or event)."""

    def __init__(
        self,
        entity_type: str,
        value: Any,
        label: str = "",
        metadata: Dict = None,
    ):
        self.entity_type = entity_type  # 'year', 'quarter', 'date', 'event'
        self.value = value
        self.label = label or str(value)
        self.metadata = metadata or {}

    def __repr__(self):
        return f"TemporalEntity({self.entity_type}: {self.label})"

    def __lt__(self, other):
        if self.entity_type == other.entity_type:
            return self.value < other.value
        return str(self.value) < str(other.value)


class TemporalExpressionNormalizer:
    """TIMEX3-style temporal expression normalizer for financial text.

    Resolves deictic (relative) temporal expressions to absolute time references
    given a document anchor date. Handles financial-domain-specific patterns
    including fiscal periods, comparative quarters, and duration expressions.

    Reference: SUTime (Chang & Manning, LREC 2012), adapted for financial domain.
    """

    RELATIVE_YEAR_MAP = {
        "last year": -1,
        "prior year": -1,
        "previous year": -1,
        "preceding year": -1,
        "the year before": -1,
        "this year": 0,
        "current year": 0,
        "the current year": 0,
        "next year": 1,
        "the following year": 1,
    }

    RELATIVE_QUARTER_MAP = {
        "last quarter": -1,
        "prior quarter": -1,
        "previous quarter": -1,
        "preceding quarter": -1,
        "the quarter before": -1,
        "this quarter": 0,
        "current quarter": 0,
        "next quarter": 1,
        "the following quarter": 1,
    }

    HALF_YEAR_MAP = {
        "first half": (1, 2),
        "1st half": (1, 2),
        "h1": (1, 2),
        "second half": (3, 4),
        "2nd half": (3, 4),
        "h2": (3, 4),
    }

    ORDINAL_QUARTER = {
        "first quarter": 1, "1st quarter": 1,
        "second quarter": 2, "2nd quarter": 2,
        "third quarter": 3, "3rd quarter": 3,
        "fourth quarter": 4, "4th quarter": 4,
    }

    def __init__(self, fiscal_year_end_month: int = 12):
        self.fiscal_year_end_month = fiscal_year_end_month

    @staticmethod
    def _advance_quarter(year: int, quarter: int, delta: int) -> Tuple[int, int]:
        """Advance a (year, quarter) pair by delta quarters."""
        total = (year * 4 + (quarter - 1)) + delta
        return total // 4, (total % 4) + 1

    def normalize(
        self,
        text: str,
        anchor_year: Optional[int],
        known_quarters: Optional[List[Tuple[int, int]]] = None,
        anchor_quarter: Optional[int] = None,
    ) -> List[TemporalEntity]:
        """Normalize temporal expressions in text to absolute references.

        Args:
            text: Input text containing temporal expressions.
            anchor_year: Document reference year (e.g., from filing date or table header).
            known_quarters: List of (year, quarter) tuples found in the document.
            anchor_quarter: Current quarter if known (1-4).
        """
        if not text or anchor_year is None:
            return []

        q = text.lower()
        entities: List[TemporalEntity] = []

        if anchor_quarter is None and known_quarters:
            anchor_quarter = max(known_quarters, key=lambda x: (x[0], x[1]))[1]

        for phrase, delta in self.RELATIVE_YEAR_MAP.items():
            if phrase in q:
                resolved = anchor_year + delta
                entities.append(
                    TemporalEntity(
                        "year",
                        resolved,
                        f"{phrase}\u2192{resolved}",
                        metadata={"implicit": True, "normalized_from": phrase,
                                  "anchor_year": anchor_year, "timex_type": "DATE",
                                  "timex_value": str(resolved)},
                    )
                )

        for phrase, delta in self.RELATIVE_QUARTER_MAP.items():
            if phrase in q:
                aq = anchor_quarter or 4
                res_year, res_q = self._advance_quarter(anchor_year, aq, delta)
                entities.append(
                    TemporalEntity(
                        "quarter",
                        (res_year, res_q),
                        f"{phrase}\u2192Q{res_q} {res_year}",
                        metadata={"implicit": True, "normalized_from": phrase,
                                  "anchor_year": anchor_year, "timex_type": "DATE",
                                  "timex_value": f"{res_year}-Q{res_q}"},
                    )
                )

        if "prior period" in q or "previous period" in q or "the period before" in q:
            entities.append(
                TemporalEntity(
                    "relative_period",
                    "prior_period",
                    "prior period",
                    metadata={"implicit": True, "anchor_year": anchor_year,
                              "timex_type": "DATE"},
                )
            )

        if "same period last year" in q or "comparable quarter" in q or "year-ago quarter" in q or "same quarter last year" in q:
            aq = anchor_quarter or 4
            entities.append(
                TemporalEntity(
                    "quarter",
                    (anchor_year - 1, aq),
                    f"same period last year\u2192Q{aq} {anchor_year - 1}",
                    metadata={"implicit": True, "normalized_from": "same period last year",
                              "anchor_year": anchor_year, "timex_type": "DATE",
                              "timex_value": f"{anchor_year - 1}-Q{aq}"},
                )
            )

        m = re.search(r"(\d+)\s+years?\s+ago", q)
        if m:
            k = int(m.group(1))
            resolved = anchor_year - k
            entities.append(
                TemporalEntity(
                    "year",
                    resolved,
                    f"{k} years ago\u2192{resolved}",
                    metadata={"implicit": True, "normalized_from": m.group(0),
                              "anchor_year": anchor_year, "timex_type": "DATE",
                              "timex_value": str(resolved)},
                )
            )

        m = re.search(r"(\d+)\s+quarters?\s+ago", q)
        if m:
            k = int(m.group(1))
            aq = anchor_quarter or 4
            res_year, res_q = self._advance_quarter(anchor_year, aq, -k)
            entities.append(
                TemporalEntity(
                    "quarter",
                    (res_year, res_q),
                    f"{k} quarters ago\u2192Q{res_q} {res_year}",
                    metadata={"implicit": True, "normalized_from": m.group(0),
                              "anchor_year": anchor_year, "timex_type": "DATE",
                              "timex_value": f"{res_year}-Q{res_q}"},
                )
            )

        m = re.search(r"(?:over|during|in)\s+the\s+(?:past|last|previous)\s+(\d+)\s+years?", q)
        if m:
            k = int(m.group(1))
            start_year = anchor_year - k + 1
            entities.append(
                TemporalEntity(
                    "range",
                    (start_year, anchor_year),
                    f"past {k} years\u2192{start_year}-{anchor_year}",
                    metadata={"implicit": True, "normalized_from": m.group(0),
                              "anchor_year": anchor_year, "timex_type": "DURATION",
                              "timex_value": f"P{k}Y",
                              "range_start": start_year, "range_end": anchor_year},
                )
            )

        m = re.search(r"(?:over|during|in)\s+the\s+(?:past|last|previous)\s+(\d+)\s+quarters?", q)
        if m:
            k = int(m.group(1))
            aq = anchor_quarter or 4
            start_y, start_q = self._advance_quarter(anchor_year, aq, -(k - 1))
            entities.append(
                TemporalEntity(
                    "range",
                    ((start_y, start_q), (anchor_year, aq)),
                    f"past {k} quarters\u2192Q{start_q} {start_y}-Q{aq} {anchor_year}",
                    metadata={"implicit": True, "normalized_from": m.group(0),
                              "anchor_year": anchor_year, "timex_type": "DURATION",
                              "timex_value": f"P{k}Q"},
                )
            )

        if "year-to-date" in q or "ytd" in q:
            entities.append(
                TemporalEntity(
                    "range",
                    (anchor_year, "ytd"),
                    f"YTD {anchor_year}",
                    metadata={"implicit": True, "normalized_from": "year-to-date",
                              "anchor_year": anchor_year, "timex_type": "DURATION"},
                )
            )

        for phrase, (q_start, q_end) in self.HALF_YEAR_MAP.items():
            if phrase in q:
                m_year = re.search(rf"{re.escape(phrase)}\s+(?:of\s+)?(\d{{4}})", q)
                yr = int(m_year.group(1)) if m_year else anchor_year
                entities.append(
                    TemporalEntity(
                        "range",
                        ((yr, q_start), (yr, q_end)),
                        f"{phrase} {yr}\u2192Q{q_start}-Q{q_end} {yr}",
                        metadata={"implicit": True, "normalized_from": phrase,
                                  "anchor_year": anchor_year, "timex_type": "DATE",
                                  "timex_value": f"{yr}-H{1 if q_start == 1 else 2}"},
                    )
                )

        for phrase, qnum in self.ORDINAL_QUARTER.items():
            if phrase in q:
                m_year = re.search(rf"{re.escape(phrase)}\s+(?:of\s+)?(\d{{4}})", q)
                yr = int(m_year.group(1)) if m_year else anchor_year
                entities.append(
                    TemporalEntity(
                        "quarter",
                        (yr, qnum),
                        f"{phrase}\u2192Q{qnum} {yr}",
                        metadata={"implicit": True, "normalized_from": phrase,
                                  "anchor_year": anchor_year, "timex_type": "DATE",
                                  "timex_value": f"{yr}-Q{qnum}"},
                    )
                )

        m_fy = re.search(r"\bfy\s*(\d{2,4})\b", q)
        if m_fy:
            fy = int(m_fy.group(1))
            if fy < 100:
                fy += 2000 if fy < 50 else 1900
            entities.append(
                TemporalEntity(
                    "fiscal_year",
                    fy,
                    f"FY{fy}",
                    metadata={"implicit": True, "normalized_from": m_fy.group(0),
                              "anchor_year": anchor_year, "timex_type": "DATE",
                              "timex_value": str(fy),
                              "fiscal_year_end_month": self.fiscal_year_end_month},
                )
            )

        q_merger = re.search(r"(?:since|following|after)\s+the\s+(?:merger|acquisition|restructuring|ipo|spin-?off)(?:\s+in\s+q([1-4]))?", q)
        if q_merger:
            qtr = int(q_merger.group(1)) if q_merger.group(1) else (known_quarters[-1][1] if known_quarters else 1)
            event_name = re.search(r"the\s+(\w+)", q_merger.group(0)).group(1)
            entities.append(
                TemporalEntity(
                    "range",
                    ((anchor_year, qtr), "present"),
                    f"since {event_name} Q{qtr} {anchor_year}",
                    metadata={"implicit": True, "normalized_from": q_merger.group(0),
                              "anchor_year": anchor_year, "timex_type": "DURATION",
                              "event_type": event_name},
                )
            )

        return entities


class TemporalGraph:
    """Graph structure for temporal reasoning over financial events."""

    def __init__(self):
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.adjacency: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
            self.nodes: Dict[str, Dict] = {}
        self.entities: Dict[str, TemporalEntity] = {}
        self.stn_constraints: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def add_entity(self, entity: TemporalEntity):
        """Add a temporal entity as a node."""
        node_id = f"{entity.entity_type}_{entity.value}"
        self.entities[node_id] = entity
        if HAS_NETWORKX:
            self.graph.add_node(node_id, **entity.metadata, label=entity.label)
        else:
            self.nodes[node_id] = {**entity.metadata, "label": entity.label}

    def add_relation(
        self, source_id: str, target_id: str, relation: str, metadata: Dict = None
    ):
        """Add a temporal relation between entities."""
        meta = metadata or {}
        if HAS_NETWORKX:
            self.graph.add_edge(source_id, target_id, relation=relation, **meta)
        else:
            self.adjacency[source_id].append((target_id, {"relation": relation, **meta}))

    def add_constraint(
        self,
        source_id: str,
        target_id: str,
        lower_bound: float = 0.0,
        upper_bound: float = float("inf"),
        relation: str = "constraint",
    ):
        """Add STN-like temporal constraint: target - source in [lower, upper]."""
        self.stn_constraints[(source_id, target_id)] = (lower_bound, upper_bound)
        self.add_relation(
            source_id,
            target_id,
            relation,
            metadata={"lower_bound": lower_bound, "upper_bound": upper_bound},
        )

    def propagate_constraints(self) -> Dict[str, Any]:
        """Simple constraint propagation over qualitative BEFORE links."""
        before_pairs = set()
        for (src, tgt), _ in self.stn_constraints.items():
            before_pairs.add((src, tgt))

        changed = True
        while changed:
            changed = False
            current = list(before_pairs)
            for a, b in current:
                for c, d in current:
                    if b == c and (a, d) not in before_pairs:
                        before_pairs.add((a, d))
                        changed = True

        inferred = []
        for a, b in sorted(before_pairs):
            if (a, b) not in self.stn_constraints:
                inferred.append((a, b))
                self.add_relation(a, b, "inferred_before", metadata={"inferred": True})

        return {"consistent": True, "inferred_before_edges": inferred, "num_constraints": len(self.stn_constraints)}

    def get_temporal_path(self, start_id: str, end_id: str) -> List[str]:
        """Find temporal path between two entities."""
        if HAS_NETWORKX:
            try:
                return nx.shortest_path(self.graph, start_id, end_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
        else:
            # BFS fallback
            visited = set()
            queue = [(start_id, [start_id])]
            while queue:
                node, path = queue.pop(0)
                if node == end_id:
                    return path
                if node in visited:
                    continue
                visited.add(node)
                for neighbor, _ in self.adjacency.get(node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
            return []

    def get_trend(self, metric: str, entity_ids: List[str]) -> Dict[str, Any]:
        """Analyze trend across temporal entities for a given metric."""
        values = []
        for eid in sorted(entity_ids):
            entity = self.entities.get(eid)
            if entity and metric in entity.metadata:
                val = entity.metadata[metric]
                if isinstance(val, (int, float)):
                    values.append((entity.value, val))

        if len(values) < 2:
            return {"trend": "insufficient_data", "values": values}

        # Simple trend analysis
        changes = [values[i][1] - values[i-1][1] for i in range(1, len(values))]
        avg_change = sum(changes) / len(changes) if changes else 0

        mean_value = sum(v for _, v in values) / len(values)
        magnitude = max(abs(mean_value), 1e-9)
        rel_changes = [abs(c) / magnitude for c in changes]
        stability_threshold = 0.02

        if all(rc < stability_threshold for rc in rel_changes):
            trend = "stable"
        elif all(c > 0 for c in changes):
            trend = "increasing"
        elif all(c < 0 for c in changes):
            trend = "decreasing"
        elif abs(avg_change) / magnitude < stability_threshold:
            trend = "stable"
        else:
            trend = "fluctuating"

        return {
            "trend": trend,
            "values": values,
            "changes": changes,
            "average_change": avg_change,
            "total_change": values[-1][1] - values[0][1] if values else 0,
        }


class TemporalReasoner:
    """Temporal reasoning for financial question answering.

    Capabilities:
    1. Extract temporal entities from questions and context
    2. Build temporal graphs connecting financial events
    3. Detect trends across time periods
    4. Resolve implicit temporal references
    5. Multi-hop temporal reasoning
    """

    # Temporal patterns for extraction
    QUARTER_PATTERN = re.compile(r"Q([1-4])\s*(\d{4})?", re.IGNORECASE)
    YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
    FISCAL_PATTERN = re.compile(
        r"fiscal\s*(year|quarter)?\s*(\d{4})?", re.IGNORECASE
    )
    RELATIVE_PATTERNS = {
        "previous": -1,
        "prior": -1,
        "preceding": -1,
        "last": -1,
        "next": 1,
        "following": 1,
        "subsequent": 1,
        "current": 0,
        "this": 0,
    }

    TEMPORAL_KEYWORDS = [
        "year-over-year", "yoy", "quarter-over-quarter", "qoq",
        "month-over-month", "mom", "annually", "quarterly", "monthly",
        "growth", "decline", "increase", "decrease", "trend",
        "compared to", "versus", "from", "to", "between", "during",
        "before", "after", "since", "until",
    ]

    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.normalizer = TemporalExpressionNormalizer()

    def _infer_anchor_year(
        self,
        question: str,
        table: List[List[str]],
        context: str = "",
    ) -> Optional[int]:
        """Infer a document anchor year for deictic temporal resolution."""
        years = []
        years.extend(int(y) for y in self.YEAR_PATTERN.findall(question or ""))
        years.extend(int(y) for y in self.YEAR_PATTERN.findall(context or ""))
        if table and table[0]:
            for col in table[0]:
                years.extend(int(y) for y in self.YEAR_PATTERN.findall(str(col)))
        return max(years) if years else None

    def _resolve_implicit_temporal_entities(
        self,
        text: str,
        anchor_year: Optional[int],
    ) -> List[TemporalEntity]:
        """Resolve implicit/deictic expressions like 'last year', 'prior quarter'."""
        if not text or anchor_year is None:
            return []

        known_quarters = []
        for m in self.QUARTER_PATTERN.finditer(text):
            qtr = int(m.group(1))
            year = int(m.group(2)) if m.group(2) else anchor_year
            if year:
                known_quarters.append((year, qtr))

        anchor_quarter = None
        if known_quarters:
            anchor_quarter = max(known_quarters, key=lambda x: (x[0], x[1]))[1]

        return self.normalizer.normalize(text, anchor_year, known_quarters, anchor_quarter)

    def extract_event_temporal_relations(
        self,
        text: str,
    ) -> List[Tuple[str, str, str]]:
        """Extract event-event temporal relations with typed labels."""
        relations: List[Tuple[str, str, str]] = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
        for s in sentences:
            lower = s.lower()
            if " before " in lower:
                left, right = re.split(r"\bbefore\b", s, maxsplit=1, flags=re.I)
                relations.append((left.strip(), right.strip(), "before"))
            if " after " in lower:
                left, right = re.split(r"\bafter\b", s, maxsplit=1, flags=re.I)
                relations.append((left.strip(), right.strip(), "after"))
            if " during " in lower:
                left, right = re.split(r"\bduring\b", s, maxsplit=1, flags=re.I)
                relations.append((left.strip(), right.strip(), "during"))
            if re.search(r"\b(coincided with|overlap(?:ped)? with)\b", lower):
                parts = re.split(r"\bcoincided with\b|\boverlap(?:ped)? with\b", s, maxsplit=1, flags=re.I)
                if len(parts) == 2:
                    relations.append((parts[0].strip(), parts[1].strip(), "overlap"))
        return relations

    def extract_temporal_entities(
        self, text: str
    ) -> List[TemporalEntity]:
        """Extract temporal entities from text."""
        entities = []

        # Extract years
        for match in self.YEAR_PATTERN.finditer(text):
            year = int(match.group(1))
            entities.append(TemporalEntity("year", year, str(year)))

        # Extract quarters
        for match in self.QUARTER_PATTERN.finditer(text):
            quarter = int(match.group(1))
            year = int(match.group(2)) if match.group(2) else None
            label = f"Q{quarter}" + (f" {year}" if year else "")
            val = (year, quarter) if year else quarter
            entities.append(TemporalEntity("quarter", val, label))

        # Extract fiscal year references
        for match in self.FISCAL_PATTERN.finditer(text):
            year_str = match.group(2)
            if year_str:
                entities.append(
                    TemporalEntity("fiscal_year", int(year_str), f"FY{year_str}")
                )

        return entities

    def detect_temporal_relations(
        self, entities: List[TemporalEntity]
    ) -> List[Tuple[TemporalEntity, TemporalEntity, str]]:
        """Detect temporal ordering relations between entities."""
        relations = []

        # Sort entities by value for ordering
        year_entities = sorted(
            [e for e in entities if e.entity_type == "year"],
            key=lambda e: e.value,
        )

        for i in range(len(year_entities) - 1):
            relations.append(
                (year_entities[i], year_entities[i + 1], "precedes")
            )

        # Quarter ordering
        quarter_entities = sorted(
            [e for e in entities if e.entity_type == "quarter"],
            key=lambda e: (e.value if isinstance(e.value, tuple) else (0, e.value)),
        )

        for i in range(len(quarter_entities) - 1):
            relations.append(
                (quarter_entities[i], quarter_entities[i + 1], "precedes")
            )

        return relations

    def build_temporal_graph(
        self,
        question: str,
        table: List[List[str]],
        context: str = "",
    ) -> TemporalGraph:
        """Build a temporal graph from question, table, and context.

        Creates nodes for each time period found and connects them
        with temporal ordering edges. Also attaches financial metric
        values from the table.
        """
        graph = TemporalGraph()

        # Extract temporal entities from question and context
        all_text = question + " " + context
        entities = self.extract_temporal_entities(all_text)
        anchor_year = self._infer_anchor_year(question, table, context)
        entities.extend(self._resolve_implicit_temporal_entities(all_text, anchor_year))

        # Also extract years from table header
        if table and table[0]:
            header = table[0]
            for col_idx, col_name in enumerate(header):
                col_years = self.YEAR_PATTERN.findall(str(col_name))
                for y in col_years:
                    year = int(y)
                    entity = TemporalEntity("year", year, str(year))
                    entities.append(entity)

        # Deduplicate
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.entity_type, str(e.value))
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
                graph.add_entity(e)

        # Add temporal relations
        relations = self.detect_temporal_relations(unique_entities)
        for src, tgt, rel in relations:
            src_id = f"{src.entity_type}_{src.value}"
            tgt_id = f"{tgt.entity_type}_{tgt.value}"
            graph.add_relation(src_id, tgt_id, rel)
            if rel == "precedes":
                graph.add_constraint(src_id, tgt_id, 0.0, float("inf"), relation="before")

        # Event-event temporal edges (BEFORE/AFTER/DURING/OVERLAP)
        for ev1, ev2, rel in self.extract_event_temporal_relations(all_text):
            e1 = TemporalEntity("event", re.sub(r"\s+", "_", ev1.lower())[:80], ev1, metadata={"event_text": ev1})
            e2 = TemporalEntity("event", re.sub(r"\s+", "_", ev2.lower())[:80], ev2, metadata={"event_text": ev2})
            for e in (e1, e2):
                eid = f"{e.entity_type}_{e.value}"
                if eid not in graph.entities:
                    graph.add_entity(e)
            src_id = f"event_{e1.value}"
            tgt_id = f"event_{e2.value}"
            graph.add_relation(src_id, tgt_id, rel)
            if rel == "before":
                graph.add_constraint(src_id, tgt_id, 0.0, float("inf"), relation="before")
            elif rel == "after":
                graph.add_constraint(tgt_id, src_id, 0.0, float("inf"), relation="before")

        # Attach table values to temporal nodes
        if table and len(table) > 1:
            header = table[0]
            for row in table[1:]:
                metric_name = str(row[0]).strip() if row else ""
                for col_idx, col_name in enumerate(header[1:], 1):
                    col_years = self.YEAR_PATTERN.findall(str(col_name))
                    for y in col_years:
                        year = int(y)
                        node_id = f"year_{year}"
                        if node_id in graph.entities:
                            val = parse_financial_number(str(row[col_idx]) if col_idx < len(row) else "")
                            if val is not None:
                                graph.entities[node_id].metadata[metric_name] = val

        return graph

    DEICTIC_YEAR_PHRASES = list(TemporalExpressionNormalizer.RELATIVE_YEAR_MAP.keys())
    DEICTIC_QUARTER_PHRASES = list(TemporalExpressionNormalizer.RELATIVE_QUARTER_MAP.keys())
    DURATION_PATTERNS = [
        r"(?:over|during|in)\s+the\s+(?:past|last|previous)\s+\d+\s+(?:years?|quarters?)",
        r"\d+\s+(?:years?|quarters?)\s+ago",
    ]

    def detect_temporal_question_type(self, question: str) -> Dict[str, Any]:
        """Classify the temporal reasoning type needed for a question."""
        q_lower = question.lower()

        types = {
            "comparison": False,
            "trend": False,
            "point_lookup": False,
            "relative": False,
            "range": False,
            "extreme": False,
            "deictic": False,
            "duration": False,
        }

        if any(kw in q_lower for kw in ["compared to", "versus", "vs", "relative to", "from", "change"]):
            types["comparison"] = True

        if any(kw in q_lower for kw in ["trend", "growth", "over time", "year-over-year", "consistently"]):
            types["trend"] = True

        years = self.YEAR_PATTERN.findall(q_lower)
        if len(years) == 1:
            types["point_lookup"] = True

        if any(kw in q_lower for kw in self.RELATIVE_PATTERNS.keys()):
            types["relative"] = True

        if any(phrase in q_lower for phrase in self.DEICTIC_YEAR_PHRASES + self.DEICTIC_QUARTER_PHRASES):
            types["deictic"] = True
            types["relative"] = True

        if any(kw in q_lower for kw in ["same period last year", "comparable quarter",
                                         "year-ago quarter", "same quarter last year"]):
            types["deictic"] = True
            types["comparison"] = True

        if any(re.search(p, q_lower) for p in self.DURATION_PATTERNS):
            types["duration"] = True
            types["range"] = True

        if any(kw in q_lower for kw in ["between", "from", "during", "throughout"]):
            if len(years) >= 2:
                types["range"] = True

        if any(kw in q_lower for kw in ["year-to-date", "ytd", "first half", "second half", "h1", "h2"]):
            types["range"] = True

        if any(kw in q_lower for kw in ["highest", "lowest", "maximum", "minimum", "most", "least", "best", "worst"]):
            types["extreme"] = True

        return types

    def reason(
        self,
        question: str,
        table: List[List[str]],
        context: str = "",
    ) -> Dict[str, Any]:
        """Perform temporal reasoning for a financial question.

        Returns:
            Dict with temporal analysis results including:
            - temporal_entities: extracted time references
            - temporal_type: classification of temporal reasoning needed
            - temporal_graph: the constructed temporal graph info
            - temporal_context: enriched context with temporal information
            - trend_analysis: trend results if applicable
        """
        result = {
            "question": question,
            "temporal_entities": [],
            "implicit_temporal_entities": [],
            "temporal_type": {},
            "temporal_context": "",
            "trend_analysis": None,
            "temporal_ordering": [],
            "anchor_year": None,
            "event_temporal_relations": [],
            "constraint_propagation": {},
        }

        # Extract entities
        anchor_year = self._infer_anchor_year(question, table, context)
        entities = self.extract_temporal_entities(question + " " + context)
        implicit_entities = self._resolve_implicit_temporal_entities(
            question + " " + context,
            anchor_year,
        )
        entities.extend(implicit_entities)
        result["temporal_entities"] = [
            {"type": e.entity_type, "value": e.value, "label": e.label}
            for e in entities
        ]
        result["implicit_temporal_entities"] = [
            {"type": e.entity_type, "value": e.value, "label": e.label}
            for e in implicit_entities
        ]
        result["anchor_year"] = anchor_year

        # Classify temporal type
        result["temporal_type"] = self.detect_temporal_question_type(question)

        # Build temporal graph
        graph = self.build_temporal_graph(question, table, context)
        result["event_temporal_relations"] = self.extract_event_temporal_relations(question + " " + context)
        result["constraint_propagation"] = graph.propagate_constraints()

        # Extract temporal ordering
        year_entities = sorted(
            [e for e in entities if e.entity_type == "year"],
            key=lambda e: e.value,
        )
        result["temporal_ordering"] = [e.value for e in year_entities]

        # Perform trend analysis when temporal data is available
        # Trigger on: explicit trend/comparison keywords, or any question with 2+ years
        has_temporal_signal = (
            result["temporal_type"].get("trend")
            or result["temporal_type"].get("comparison")
            or result["temporal_type"].get("range")
            or len(result["temporal_ordering"]) >= 2
        )
        if has_temporal_signal and table and len(table) > 1:
            year_ids = [f"year_{y}" for y in result["temporal_ordering"]]
            # Try trend for the first data row (most relevant metric)
            best_trend = None
            for row_idx in range(1, min(len(table), 6)):
                metric_name = str(table[row_idx][0]).strip() if table[row_idx] else ""
                trend = graph.get_trend(metric_name, year_ids)
                if trend and trend.get("trend") != "insufficient_data":
                    best_trend = trend
                    break
            if best_trend is None:
                # Fallback: try with all metrics
                for entity_id, entity in graph.entities.items():
                    if entity.metadata:
                        for metric in entity.metadata:
                            trend = graph.get_trend(metric, year_ids)
                            if trend and trend.get("trend") != "insufficient_data":
                                best_trend = trend
                                break
                    if best_trend:
                        break
            result["trend_analysis"] = best_trend

        # Build enriched temporal context
        temporal_ctx_parts = []
        if result["temporal_ordering"]:
            temporal_ctx_parts.append(
                f"Time periods referenced: {', '.join(str(y) for y in result['temporal_ordering'])}"
            )
        if result["implicit_temporal_entities"]:
            temporal_ctx_parts.append(
                "Implicit time refs: "
                + ", ".join(ent["label"] for ent in result["implicit_temporal_entities"])
            )
        if result["event_temporal_relations"]:
            formatted = [f"{r[2].upper()}({r[0]} -> {r[1]})" for r in result["event_temporal_relations"][:3]]
            temporal_ctx_parts.append("Event relations: " + "; ".join(formatted))
        if result["trend_analysis"]:
            ta = result["trend_analysis"]
            temporal_ctx_parts.append(f"Trend: {ta['trend']}")
            if ta.get("total_change"):
                temporal_ctx_parts.append(
                    f"Total change: {ta['total_change']:.2f}"
                )

        result["temporal_context"] = " | ".join(temporal_ctx_parts)

        return result
