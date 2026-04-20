"""Temporal Reasoning Module for financial QA.

Handles time-based reasoning including:
- Temporal ordering of financial events
- Trend detection across reporting periods
- Implicit temporal reference resolution
- Temporal graph construction for multi-hop reasoning
- Event-event temporal relation extraction (Allen's interval algebra)
- Financial event classification and temporal linking
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..utils.financial_utils import extract_years_from_text, parse_financial_number


class AllenRelation(Enum):
    """Allen's interval algebra relations (Allen, 1983).

    Defines 13 possible temporal relations between two intervals:
    7 basic + 6 inverses, plus EQUAL.
    """
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    DURING = "during"
    CONTAINS = "contains"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUAL = "equal"

    @classmethod
    def inverse(cls, rel: "AllenRelation") -> "AllenRelation":
        _inv = {
            cls.BEFORE: cls.AFTER, cls.AFTER: cls.BEFORE,
            cls.MEETS: cls.MET_BY, cls.MET_BY: cls.MEETS,
            cls.OVERLAPS: cls.OVERLAPPED_BY, cls.OVERLAPPED_BY: cls.OVERLAPS,
            cls.STARTS: cls.STARTED_BY, cls.STARTED_BY: cls.STARTS,
            cls.DURING: cls.CONTAINS, cls.CONTAINS: cls.DURING,
            cls.FINISHES: cls.FINISHED_BY, cls.FINISHED_BY: cls.FINISHES,
            cls.EQUAL: cls.EQUAL,
        }
        return _inv[rel]


class FinancialEventType(Enum):
    EARNINGS = "earnings"
    REVENUE = "revenue"
    ACQUISITION = "acquisition"
    MERGER = "merger"
    DIVESTITURE = "divestiture"
    RESTRUCTURING = "restructuring"
    IPO = "ipo"
    DIVIDEND = "dividend"
    BUYBACK = "buyback"
    DEBT_ISSUANCE = "debt_issuance"
    GUIDANCE = "guidance"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    COST_ACTION = "cost_action"
    IMPAIRMENT = "impairment"
    OTHER = "other"


@dataclass
class FinancialEvent:
    """A financial event with temporal grounding and type classification."""
    text: str
    event_type: FinancialEventType = FinancialEventType.OTHER
    temporal_anchor: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[int] = None
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        slug = re.sub(r"\s+", "_", self.text.lower().strip()[:60])
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        return f"event_{slug}"


@dataclass
class EventTemporalRelation:
    """A temporal relation between two financial events."""
    event1: FinancialEvent
    event2: FinancialEvent
    relation: AllenRelation
    confidence: float = 0.5
    evidence: str = ""
    signal_word: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event1": self.event1.text,
            "event2": self.event2.text,
            "event1_type": self.event1.event_type.value,
            "event2_type": self.event2.event_type.value,
            "relation": self.relation.value,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "signal_word": self.signal_word,
        }

    def inverse(self) -> "EventTemporalRelation":
        return EventTemporalRelation(
            event1=self.event2,
            event2=self.event1,
            relation=AllenRelation.inverse(self.relation),
            confidence=self.confidence,
            evidence=self.evidence,
            signal_word=self.signal_word,
        )


class FinancialEventClassifier:
    """Classify financial events by type using keyword matching."""

    _PATTERNS: List[Tuple[FinancialEventType, re.Pattern]] = [
        (FinancialEventType.ACQUISITION, re.compile(
            r"\b(acqui(?:red?|sition)|bought|purchased|takeover)\b", re.I)),
        (FinancialEventType.MERGER, re.compile(
            r"\b(merg(?:er|ed|ing)|consolidat(?:ed|ion))\b", re.I)),
        (FinancialEventType.DIVESTITURE, re.compile(
            r"\b(divest(?:ed|iture|ing)|sold\s+(?:its|the)|spin-?off|spun off|dispos(?:ed|al|ition))\b", re.I)),
        (FinancialEventType.RESTRUCTURING, re.compile(
            r"\b(restructur(?:ed|ing)|reorganiz(?:ed|ation)|transformation|turnaround)\b", re.I)),
        (FinancialEventType.IPO, re.compile(
            r"\b(ipo|initial public offering|went public|public listing)\b", re.I)),
        (FinancialEventType.DIVIDEND, re.compile(
            r"\b(dividend|distribut(?:ed|ion)|payout|special dividend)\b", re.I)),
        (FinancialEventType.BUYBACK, re.compile(
            r"\b(buyback|repurchas(?:ed?|ing)|share repurchase|stock repurchase)\b", re.I)),
        (FinancialEventType.DEBT_ISSUANCE, re.compile(
            r"\b(debt issuance|bond offering|borrowed|refinanc(?:ed|ing)|credit facility)\b", re.I)),
        (FinancialEventType.GUIDANCE, re.compile(
            r"\b(guidance|outlook|forecast|project(?:ed|ion)|expect(?:ed|ation))\b", re.I)),
        (FinancialEventType.REGULATORY, re.compile(
            r"\b(regulat(?:ory|ion)|compliance|approv(?:ed|al)|sanction|fine|penalt(?:y|ies)|litigation|settlement)\b", re.I)),
        (FinancialEventType.PRODUCT_LAUNCH, re.compile(
            r"\b(launch(?:ed)?|introduc(?:ed|tion)|new product|roll(?:ed)?\s*out|debut(?:ed)?)\b", re.I)),
        (FinancialEventType.COST_ACTION, re.compile(
            r"\b(cost(?:\s+|-)?cut(?:ting)?|layoff|headcount reduction|efficiency|savings program|workforce reduction)\b", re.I)),
        (FinancialEventType.IMPAIRMENT, re.compile(
            r"\b(impairment|write(?:\s*|-)?(?:down|off)|goodwill charge)\b", re.I)),
        (FinancialEventType.EARNINGS, re.compile(
            r"\b(earnings|profit(?:ability)?|income|loss(?:es)?|beat|miss(?:ed)?)\b", re.I)),
        (FinancialEventType.REVENUE, re.compile(
            r"\b(revenue|sales|top(?:\s*|-)?line)\b", re.I)),
    ]

    @classmethod
    def classify(cls, text: str) -> FinancialEventType:
        for event_type, pattern in cls._PATTERNS:
            if pattern.search(text):
                return event_type
        return FinancialEventType.OTHER


class EventTemporalRelationExtractor:
    """Extract event-event temporal relations from financial text.

    Implements Allen's interval algebra with financial-domain signal words.
    Handles 13 temporal relations between event pairs, temporal anchoring
    via year/quarter mentions, and transitive closure over relation chains.

    Reference: Allen (1983), TimeML (Pustejovsky et al., 2003).
    """

    SIGNAL_PATTERNS: List[Tuple[str, AllenRelation, str]] = [
        (r"\bbefore\b", AllenRelation.BEFORE, "before"),
        (r"\bprior to\b", AllenRelation.BEFORE, "prior to"),
        (r"\bahead of\b", AllenRelation.BEFORE, "ahead of"),
        (r"\bpreceding\b", AllenRelation.BEFORE, "preceding"),
        (r"\bleading up to\b", AllenRelation.BEFORE, "leading up to"),
        (r"\bin advance of\b", AllenRelation.BEFORE, "in advance of"),

        (r"\bafter\b", AllenRelation.AFTER, "after"),
        (r"\bfollowing\b", AllenRelation.AFTER, "following"),
        (r"\bsubsequent(?:ly)? to\b", AllenRelation.AFTER, "subsequent to"),
        (r"\bin the wake of\b", AllenRelation.AFTER, "in the wake of"),
        (r"\bpost\b", AllenRelation.AFTER, "post"),

        (r"\bduring\b", AllenRelation.DURING, "during"),
        (r"\bthroughout\b", AllenRelation.DURING, "throughout"),
        (r"\bin the course of\b", AllenRelation.DURING, "in the course of"),
        (r"\bamid(?:st)?\b", AllenRelation.DURING, "amid"),
        (r"\bwhile\b", AllenRelation.DURING, "while"),

        (r"\bsimultaneously\b", AllenRelation.EQUAL, "simultaneously"),
        (r"\bat the same time\b", AllenRelation.EQUAL, "at the same time"),
        (r"\bcoincid(?:ed|ing) with\b", AllenRelation.EQUAL, "coincided with"),
        (r"\balongside\b", AllenRelation.EQUAL, "alongside"),
        (r"\bconcurrently\b", AllenRelation.EQUAL, "concurrently"),

        (r"\boverlap(?:ped|ping)?\s+with\b", AllenRelation.OVERLAPS, "overlapped with"),
        (r"\bpartially during\b", AllenRelation.OVERLAPS, "partially during"),

        (r"\bimmediately (?:before|prior to)\b", AllenRelation.MEETS, "immediately before"),
        (r"\bjust before\b", AllenRelation.MEETS, "just before"),
        (r"\bon the eve of\b", AllenRelation.MEETS, "on the eve of"),

        (r"\bimmediately after\b", AllenRelation.MET_BY, "immediately after"),
        (r"\bjust after\b", AllenRelation.MET_BY, "just after"),
        (r"\bright after\b", AllenRelation.MET_BY, "right after"),

        (r"\bsince\b", AllenRelation.AFTER, "since"),
        (r"\buntil\b", AllenRelation.BEFORE, "until"),
        (r"\bwhen\b", AllenRelation.EQUAL, "when"),
        (r"\bas\s+(?:soon\s+as)\b", AllenRelation.MET_BY, "as soon as"),
    ]

    SEQUENCE_PATTERNS = [
        re.compile(
            r"(.{10,80}?)\s*(?:,\s*)?(?:and\s+)?then\s+(.{10,80})", re.I),
        re.compile(
            r"first[,\s]+(.{10,60}?)[,;]\s*(?:and\s+)?then\s+(.{10,60})", re.I),
        re.compile(
            r"(.{10,80}?)\s*,\s*followed by\s+(.{10,80})", re.I),
        re.compile(
            r"(.{10,80}?)\s*(?:,\s*)?which\s+(?:was\s+)?followed by\s+(.{10,80})", re.I),
        re.compile(
            r"(.{10,80}?)\s*,\s*(?:and\s+)?subsequently\s+(.{10,80})", re.I),
        re.compile(
            r"(.{10,80}?)\s*,\s*(?:and\s+)?later\s+(.{10,80})", re.I),
    ]

    YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
    QUARTER_RE = re.compile(r"\bQ([1-4])(?:\s+(19\d{2}|20\d{2}))?\b", re.I)

    def __init__(self):
        self._compiled_signals = [
            (re.compile(pat, re.I), rel, word)
            for pat, rel, word in self.SIGNAL_PATTERNS
        ]

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

    def _clean_event_span(self, span: str) -> str:
        span = re.sub(r"\s+", " ", span.strip().rstrip(".,;:"))
        span = re.sub(r"^(the|a|an|and|or|but|that|this)\s+", "", span, flags=re.I)
        return span.strip()

    def _extract_temporal_anchor(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        year = None
        quarter = None
        ym = self.YEAR_RE.search(text)
        if ym:
            year = int(ym.group(1))
        qm = self.QUARTER_RE.search(text)
        if qm:
            quarter = int(qm.group(1))
            if qm.group(2):
                year = int(qm.group(2))
        return year, quarter

    def extract_relations(self, text: str) -> List[EventTemporalRelation]:
        """Extract event-event temporal relations from text.

        1. Signal-word-based extraction (before, after, during, etc.)
        2. Sequence patterns (first...then, followed by, subsequently)
        3. Temporal anchoring via year/quarter mentions
        """
        relations: List[EventTemporalRelation] = []
        seen: Set[Tuple[str, str, str]] = set()

        for sentence in self._split_sentences(text):
            for pat, allen_rel, signal_word in self._compiled_signals:
                m = pat.search(sentence)
                if not m:
                    continue

                pos = m.start()
                left = self._clean_event_span(sentence[:pos])
                right = self._clean_event_span(sentence[m.end():])

                if len(left) < 5 and len(right) >= 10 and "," in right:
                    parts = right.split(",", 1)
                    event_ref = self._clean_event_span(parts[0])
                    event_consequence = self._clean_event_span(parts[1])
                    if len(event_ref) >= 3 and len(event_consequence) >= 5:
                        left = event_consequence
                        right = event_ref

                if len(left) < 5 or len(right) < 3:
                    continue

                key = (left.lower()[:40], right.lower()[:40], allen_rel.value)
                if key in seen:
                    continue
                seen.add(key)

                year1, q1 = self._extract_temporal_anchor(left)
                year2, q2 = self._extract_temporal_anchor(right)

                ev1 = FinancialEvent(
                    text=left,
                    event_type=FinancialEventClassifier.classify(left),
                    year=year1,
                    quarter=q1,
                    confidence=0.7,
                )
                ev2 = FinancialEvent(
                    text=right,
                    event_type=FinancialEventClassifier.classify(right),
                    year=year2,
                    quarter=q2,
                    confidence=0.7,
                )

                conf = self._estimate_confidence(signal_word, left, right, allen_rel)

                relations.append(EventTemporalRelation(
                    event1=ev1,
                    event2=ev2,
                    relation=allen_rel,
                    confidence=conf,
                    evidence=sentence,
                    signal_word=signal_word,
                ))
                break

            for seq_pat in self.SEQUENCE_PATTERNS:
                sm = seq_pat.search(sentence)
                if not sm:
                    continue
                ev1_text = self._clean_event_span(sm.group(1))
                ev2_text = self._clean_event_span(sm.group(2))
                if len(ev1_text) < 5 or len(ev2_text) < 5:
                    continue

                key = (ev1_text.lower()[:40], ev2_text.lower()[:40], "before")
                if key in seen:
                    continue
                seen.add(key)

                y1, q1 = self._extract_temporal_anchor(ev1_text)
                y2, q2 = self._extract_temporal_anchor(ev2_text)

                ev1 = FinancialEvent(
                    text=ev1_text,
                    event_type=FinancialEventClassifier.classify(ev1_text),
                    year=y1, quarter=q1, confidence=0.65,
                )
                ev2 = FinancialEvent(
                    text=ev2_text,
                    event_type=FinancialEventClassifier.classify(ev2_text),
                    year=y2, quarter=q2, confidence=0.65,
                )
                relations.append(EventTemporalRelation(
                    event1=ev1, event2=ev2,
                    relation=AllenRelation.BEFORE,
                    confidence=0.65,
                    evidence=sentence,
                    signal_word="sequence",
                ))
                break

        return relations

    def _estimate_confidence(
        self, signal: str, ev1: str, ev2: str, rel: AllenRelation
    ) -> float:
        conf = 0.6
        strong_signals = {"before", "after", "prior to", "following", "subsequently",
                          "during", "simultaneously", "immediately before", "immediately after"}
        if signal in strong_signals:
            conf += 0.15

        has_anchor_1 = bool(self.YEAR_RE.search(ev1) or self.QUARTER_RE.search(ev1))
        has_anchor_2 = bool(self.YEAR_RE.search(ev2) or self.QUARTER_RE.search(ev2))
        if has_anchor_1 and has_anchor_2:
            conf += 0.15
        elif has_anchor_1 or has_anchor_2:
            conf += 0.05

        finance_kw = ["revenue", "earnings", "acquisition", "restructuring", "merger",
                       "dividend", "guidance", "launch", "margin", "growth"]
        combined = (ev1 + " " + ev2).lower()
        relevance = sum(1 for k in finance_kw if k in combined)
        conf += min(0.1, relevance * 0.02)

        return min(1.0, conf)

    def infer_from_temporal_anchors(
        self, relations: List[EventTemporalRelation]
    ) -> List[EventTemporalRelation]:
        """Infer additional relations from temporal anchoring.

        If event A is anchored to 2020 and event B to 2022, infer A BEFORE B.
        """
        inferred: List[EventTemporalRelation] = []
        anchored = [(r.event1, r) for r in relations if r.event1.year]
        anchored += [(r.event2, r) for r in relations if r.event2.year]

        events_by_time: Dict[str, List[FinancialEvent]] = defaultdict(list)
        seen_events: Dict[str, FinancialEvent] = {}
        for ev, _ in anchored:
            key = ev.event_id
            if key not in seen_events:
                seen_events[key] = ev
                time_key = f"{ev.year or 0}_{ev.quarter or 0}"
                events_by_time[time_key].append(ev)

        sorted_keys = sorted(events_by_time.keys())
        for i in range(len(sorted_keys)):
            for j in range(i + 1, len(sorted_keys)):
                t1, t2 = sorted_keys[i], sorted_keys[j]
                y1, q1 = t1.split("_")
                y2, q2 = t2.split("_")
                if int(y1) > int(y2) or (int(y1) == int(y2) and int(q1) >= int(q2)):
                    continue

                for ev_a in events_by_time[t1]:
                    for ev_b in events_by_time[t2]:
                        if ev_a.event_id == ev_b.event_id:
                            continue
                        inferred.append(EventTemporalRelation(
                            event1=ev_a, event2=ev_b,
                            relation=AllenRelation.BEFORE,
                            confidence=0.75,
                            evidence=f"Inferred: {ev_a.text} ({ev_a.year}) before {ev_b.text} ({ev_b.year})",
                            signal_word="temporal_anchor",
                            metadata={"inferred": True},
                        ))

        return inferred

    def compute_transitive_closure(
        self, relations: List[EventTemporalRelation]
    ) -> List[EventTemporalRelation]:
        """Compute transitive closure of BEFORE relations.

        If A BEFORE B and B BEFORE C, infer A BEFORE C.
        """
        before_graph: Dict[str, Set[str]] = defaultdict(set)
        event_map: Dict[str, FinancialEvent] = {}

        for r in relations:
            if r.relation == AllenRelation.BEFORE:
                e1_id = r.event1.event_id
                e2_id = r.event2.event_id
                before_graph[e1_id].add(e2_id)
                event_map[e1_id] = r.event1
                event_map[e2_id] = r.event2

        existing = set()
        for r in relations:
            existing.add((r.event1.event_id, r.event2.event_id, r.relation.value))

        changed = True
        while changed:
            changed = False
            for a in list(before_graph.keys()):
                for b in list(before_graph.get(a, [])):
                    for c in list(before_graph.get(b, [])):
                        if c not in before_graph[a]:
                            before_graph[a].add(c)
                            changed = True

        inferred: List[EventTemporalRelation] = []
        for a, successors in before_graph.items():
            for c in successors:
                key = (a, c, AllenRelation.BEFORE.value)
                if key not in existing and a in event_map and c in event_map:
                    existing.add(key)
                    inferred.append(EventTemporalRelation(
                        event1=event_map[a],
                        event2=event_map[c],
                        relation=AllenRelation.BEFORE,
                        confidence=0.55,
                        evidence=f"Transitive: {event_map[a].text} before {event_map[c].text}",
                        signal_word="transitive",
                        metadata={"inferred": True, "inference_type": "transitive_closure"},
                    ))

        return inferred

    def build_event_timeline(
        self, relations: List[EventTemporalRelation]
    ) -> List[Dict[str, Any]]:
        """Build a chronologically ordered timeline from relations."""
        before_graph: Dict[str, Set[str]] = defaultdict(set)
        event_map: Dict[str, FinancialEvent] = {}

        for r in relations:
            event_map[r.event1.event_id] = r.event1
            event_map[r.event2.event_id] = r.event2
            if r.relation in (AllenRelation.BEFORE, AllenRelation.MEETS):
                before_graph[r.event1.event_id].add(r.event2.event_id)
            elif r.relation in (AllenRelation.AFTER, AllenRelation.MET_BY):
                before_graph[r.event2.event_id].add(r.event1.event_id)

        in_degree = defaultdict(int)
        for node in event_map:
            in_degree.setdefault(node, 0)
        for src, targets in before_graph.items():
            for tgt in targets:
                in_degree[tgt] += 1

        queue = sorted(n for n in event_map if in_degree[n] == 0)
        timeline = []
        visited = set()
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            ev = event_map[node]
            timeline.append({
                "event_id": node,
                "text": ev.text,
                "event_type": ev.event_type.value,
                "year": ev.year,
                "quarter": ev.quarter,
                "order": len(timeline),
            })
            for nxt in sorted(before_graph.get(node, [])):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        for node in event_map:
            if node not in visited:
                ev = event_map[node]
                timeline.append({
                    "event_id": node,
                    "text": ev.text,
                    "event_type": ev.event_type.value,
                    "year": ev.year,
                    "quarter": ev.quarter,
                    "order": len(timeline),
                })

        return timeline


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
        self.event_extractor = EventTemporalRelationExtractor()

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
    ) -> List[EventTemporalRelation]:
        """Extract event-event temporal relations using Allen's interval algebra.

        Returns rich EventTemporalRelation objects with typed events,
        confidence scores, and Allen relation labels.
        """
        base = self.event_extractor.extract_relations(text)
        anchor_inferred = self.event_extractor.infer_from_temporal_anchors(base)
        transitive = self.event_extractor.compute_transitive_closure(base + anchor_inferred)
        return base + anchor_inferred + transitive

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

        # Event-event temporal edges with Allen's interval algebra
        event_relations = self.extract_event_temporal_relations(all_text)
        for etr in event_relations:
            e1 = TemporalEntity(
                "event", etr.event1.event_id.replace("event_", ""),
                etr.event1.text,
                metadata={
                    "event_text": etr.event1.text,
                    "event_type": etr.event1.event_type.value,
                    "year": etr.event1.year,
                    "quarter": etr.event1.quarter,
                },
            )
            e2 = TemporalEntity(
                "event", etr.event2.event_id.replace("event_", ""),
                etr.event2.text,
                metadata={
                    "event_text": etr.event2.text,
                    "event_type": etr.event2.event_type.value,
                    "year": etr.event2.year,
                    "quarter": etr.event2.quarter,
                },
            )
            for e in (e1, e2):
                eid = f"event_{e.value}"
                if eid not in graph.entities:
                    graph.add_entity(e)
            src_id = f"event_{e1.value}"
            tgt_id = f"event_{e2.value}"
            rel_str = etr.relation.value
            graph.add_relation(src_id, tgt_id, rel_str, metadata={
                "confidence": etr.confidence,
                "signal_word": etr.signal_word,
                "allen_relation": rel_str,
            })
            if etr.relation in (AllenRelation.BEFORE, AllenRelation.MEETS):
                graph.add_constraint(src_id, tgt_id, 0.0, float("inf"), relation="before")
            elif etr.relation in (AllenRelation.AFTER, AllenRelation.MET_BY):
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
            "event_timeline": [],
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
        all_text = question + " " + context
        event_relations = self.extract_event_temporal_relations(all_text)
        result["event_temporal_relations"] = [r.to_dict() for r in event_relations]
        result["event_timeline"] = self.event_extractor.build_event_timeline(event_relations)
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
            formatted = [
                f"{r['relation'].upper()}({r['event1']} -> {r['event2']})"
                for r in result["event_temporal_relations"][:3]
            ]
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
