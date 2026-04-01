"""Temporal Reasoning Module for financial QA.

Handles time-based reasoning including:
- Temporal ordering of financial events
- Trend detection across reporting periods
- Implicit temporal reference resolution
- Temporal graph construction for multi-hop reasoning
"""

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


class TemporalGraph:
    """Graph structure for temporal reasoning over financial events."""

    def __init__(self):
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.adjacency: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
            self.nodes: Dict[str, Dict] = {}
        self.entities: Dict[str, TemporalEntity] = {}

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

        if all(c > 0 for c in changes):
            trend = "increasing"
        elif all(c < 0 for c in changes):
            trend = "decreasing"
        elif abs(avg_change) < 1e-6:
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

    def detect_temporal_question_type(self, question: str) -> Dict[str, Any]:
        """Classify the temporal reasoning type needed for a question."""
        q_lower = question.lower()

        types = {
            "comparison": False,  # Compare values across time
            "trend": False,       # Detect trend over multiple periods
            "point_lookup": False, # Find value at specific time
            "relative": False,    # Relative temporal reference
            "range": False,       # Query over time range
            "extreme": False,     # Find max/min over time
        }

        # Comparison patterns
        if any(kw in q_lower for kw in ["compared to", "versus", "vs", "relative to", "from", "change"]):
            types["comparison"] = True

        # Trend patterns
        if any(kw in q_lower for kw in ["trend", "growth", "over time", "year-over-year", "consistently"]):
            types["trend"] = True

        # Point lookup
        years = self.YEAR_PATTERN.findall(q_lower)
        if len(years) == 1:
            types["point_lookup"] = True

        # Relative reference
        if any(kw in q_lower for kw in self.RELATIVE_PATTERNS.keys()):
            types["relative"] = True

        # Range
        if any(kw in q_lower for kw in ["between", "from", "during", "throughout"]):
            if len(years) >= 2:
                types["range"] = True

        # Extreme
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
            "temporal_type": {},
            "temporal_context": "",
            "trend_analysis": None,
            "temporal_ordering": [],
        }

        # Extract entities
        entities = self.extract_temporal_entities(question + " " + context)
        result["temporal_entities"] = [
            {"type": e.entity_type, "value": e.value, "label": e.label}
            for e in entities
        ]

        # Classify temporal type
        result["temporal_type"] = self.detect_temporal_question_type(question)

        # Build temporal graph
        graph = self.build_temporal_graph(question, table, context)

        # Extract temporal ordering
        year_entities = sorted(
            [e for e in entities if e.entity_type == "year"],
            key=lambda e: e.value,
        )
        result["temporal_ordering"] = [e.value for e in year_entities]

        # If trend analysis is needed, analyze trends from table
        if result["temporal_type"].get("trend") or result["temporal_type"].get("comparison"):
            if table and len(table) > 1:
                metric_name = str(table[1][0]).strip() if table[1] else ""
                year_ids = [f"year_{y}" for y in result["temporal_ordering"]]
                trend = graph.get_trend(metric_name, year_ids)
                result["trend_analysis"] = trend

        # Build enriched temporal context
        temporal_ctx_parts = []
        if result["temporal_ordering"]:
            temporal_ctx_parts.append(
                f"Time periods referenced: {', '.join(str(y) for y in result['temporal_ordering'])}"
            )
        if result["trend_analysis"]:
            ta = result["trend_analysis"]
            temporal_ctx_parts.append(f"Trend: {ta['trend']}")
            if ta.get("total_change"):
                temporal_ctx_parts.append(
                    f"Total change: {ta['total_change']:.2f}"
                )

        result["temporal_context"] = " | ".join(temporal_ctx_parts)

        return result
