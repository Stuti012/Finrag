"""Temporal Reasoning Module for financial QA.

Handles time-based reasoning including:
- Temporal ordering of financial events
- Trend detection across reporting periods
- Implicit temporal reference resolution
- Temporal graph construction for multi-hop reasoning
- Event-event temporal relation extraction (Allen's interval algebra)
- Financial event classification and temporal linking
- Temporal constraint propagation with Allen composition and STN tightening
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


class TemporalConstraintPropagator:
    """Temporal constraint propagation via Allen composition and STN tightening.

    Implements:
    - Allen relation composition table (Allen, 1983) for multi-hop inference
    - Bellman-Ford-style bound tightening on the Simple Temporal Network
    - Consistency checking with negative-cycle detection
    - Allen-to-endpoint constraint mapping for all 13 relations
    - Multi-hop Allen relation inference through chains of events

    Reference: Allen (1983), Dechter et al. (1991) STN algorithms.
    """

    _B = "before"
    _A = "after"
    _M = "meets"
    _MI = "met_by"
    _O = "overlaps"
    _OI = "overlapped_by"
    _S = "starts"
    _SI = "started_by"
    _D = "during"
    _C = "contains"
    _F = "finishes"
    _FI = "finished_by"
    _E = "equal"

    ALLEN_COMPOSITION: Dict[Tuple[str, str], Set[str]] = {}

    @classmethod
    def _init_composition_table(cls):
        """Build the Allen composition table (subset covering key financial relations).

        Full 13×13 table has 169 entries. We populate the most useful
        compositions for financial event reasoning: chains involving BEFORE,
        AFTER, MEETS, DURING, EQUAL, OVERLAPS, STARTS, FINISHES.
        """
        if cls.ALLEN_COMPOSITION:
            return

        B, A, M, MI = cls._B, cls._A, cls._M, cls._MI
        O, OI, S, SI = cls._O, cls._OI, cls._S, cls._SI
        D, C, F, FI = cls._D, cls._C, cls._F, cls._FI
        E = cls._E
        ALL = {B, A, M, MI, O, OI, S, SI, D, C, F, FI, E}

        t = cls.ALLEN_COMPOSITION
        t[(B, B)] = {B}
        t[(B, M)] = {B}
        t[(B, O)] = {B}
        t[(B, S)] = {B}
        t[(B, D)] = {B, M, O, S, D}
        t[(B, F)] = {B, M, O, S, D}
        t[(B, FI)] = {B}
        t[(B, E)] = {B}
        t[(B, A)] = ALL
        t[(B, MI)] = {B, M, O, S, D}
        t[(B, OI)] = {B, M, O, S, D}
        t[(B, SI)] = {B}
        t[(B, C)] = {B}

        t[(A, A)] = {A}
        t[(A, MI)] = {A}
        t[(A, OI)] = {A}
        t[(A, F)] = {A}
        t[(A, D)] = {A, MI, OI, F, D}
        t[(A, S)] = {A, MI, OI, F, D}
        t[(A, SI)] = {A}
        t[(A, E)] = {A}
        t[(A, B)] = ALL
        t[(A, M)] = {A, MI, OI, F, D}
        t[(A, O)] = {A, MI, OI, F, D}
        t[(A, FI)] = {A}
        t[(A, C)] = {A}

        t[(M, B)] = {B}
        t[(M, M)] = {B}
        t[(M, O)] = {B}
        t[(M, S)] = {M}
        t[(M, D)] = {O, S, D}
        t[(M, F)] = {O, S, D}
        t[(M, FI)] = {M}
        t[(M, E)] = {M}
        t[(M, A)] = {A, MI, OI, F, D}
        t[(M, MI)] = {S, SI, E}
        t[(M, OI)] = {O, S, D}
        t[(M, SI)] = {M}
        t[(M, C)] = {B}

        t[(MI, A)] = {A}
        t[(MI, MI)] = {A}
        t[(MI, OI)] = {A}
        t[(MI, F)] = {MI}
        t[(MI, D)] = {OI, F, D}
        t[(MI, S)] = {OI, F, D}
        t[(MI, SI)] = {MI}
        t[(MI, E)] = {MI}
        t[(MI, B)] = {B, M, O, S, D}
        t[(MI, M)] = {F, FI, E}
        t[(MI, O)] = {OI, F, D}
        t[(MI, FI)] = {MI}
        t[(MI, C)] = {A}

        t[(E, B)] = {B}
        t[(E, A)] = {A}
        t[(E, M)] = {M}
        t[(E, MI)] = {MI}
        t[(E, O)] = {O}
        t[(E, OI)] = {OI}
        t[(E, S)] = {S}
        t[(E, SI)] = {SI}
        t[(E, D)] = {D}
        t[(E, C)] = {C}
        t[(E, F)] = {F}
        t[(E, FI)] = {FI}
        t[(E, E)] = {E}

        t[(D, B)] = {B}
        t[(D, A)] = {A}
        t[(D, M)] = {B}
        t[(D, MI)] = {A}
        t[(D, O)] = {B, M, O, S, D}
        t[(D, OI)] = {A, MI, OI, F, D}
        t[(D, S)] = {D}
        t[(D, SI)] = {B, M, O, S, D}
        t[(D, D)] = {D}
        t[(D, C)] = ALL
        t[(D, F)] = {D}
        t[(D, FI)] = {A, MI, OI, F, D}
        t[(D, E)] = {D}

        t[(C, B)] = {B, M, O, FI, C}
        t[(C, A)] = {A, MI, OI, SI, C}
        t[(C, M)] = {O, FI, C}
        t[(C, MI)] = {OI, SI, C}
        t[(C, O)] = {O, FI, C}
        t[(C, OI)] = {OI, SI, C}
        t[(C, S)] = {O, FI, C}
        t[(C, SI)] = {C}
        t[(C, D)] = ALL
        t[(C, C)] = {C}
        t[(C, F)] = {OI, SI, C}
        t[(C, FI)] = {C}
        t[(C, E)] = {C}

        t[(O, B)] = {B}
        t[(O, A)] = {A, MI, OI, F, D}
        t[(O, M)] = {B}
        t[(O, MI)] = {OI, F, D}
        t[(O, O)] = {B, M, O}
        t[(O, OI)] = {O, OI, S, SI, D, C, F, FI, E}
        t[(O, S)] = {O}
        t[(O, SI)] = {O, FI, C}
        t[(O, D)] = {O, S, D}
        t[(O, C)] = {B, M, O, FI, C}
        t[(O, F)] = {O, S, D}
        t[(O, FI)] = {B, M, O}
        t[(O, E)] = {O}

        t[(S, B)] = {B}
        t[(S, A)] = {A}
        t[(S, M)] = {B}
        t[(S, MI)] = {MI}
        t[(S, O)] = {B, M, O}
        t[(S, OI)] = {OI, F, D}
        t[(S, S)] = {S}
        t[(S, SI)] = {S, SI, E}
        t[(S, D)] = {D}
        t[(S, C)] = {B, M, O, FI, C}
        t[(S, F)] = {D}
        t[(S, FI)] = {B, M, O}
        t[(S, E)] = {S}

        t[(F, B)] = {B}
        t[(F, A)] = {A}
        t[(F, M)] = {M}
        t[(F, MI)] = {A}
        t[(F, O)] = {B, M, O}
        t[(F, OI)] = {A, MI, OI}
        t[(F, S)] = {D}
        t[(F, SI)] = {A, MI, OI}
        t[(F, D)] = {D}
        t[(F, C)] = {A, MI, OI, SI, C}
        t[(F, F)] = {F}
        t[(F, FI)] = {F, FI, E}
        t[(F, E)] = {F}

        for r in [B, A, M, MI, O, OI, S, SI, D, C, F, FI, E]:
            for s in [B, A, M, MI, O, OI, S, SI, D, C, F, FI, E]:
                if (r, s) not in t:
                    t[(r, s)] = ALL

    @classmethod
    def compose(cls, rel1: str, rel2: str) -> Set[str]:
        """Compose two Allen relations: given A rel1 B and B rel2 C, return possible A ? C."""
        cls._init_composition_table()
        return set(cls.ALLEN_COMPOSITION.get((rel1, rel2), set()))

    @staticmethod
    def allen_to_precedence(rel: str) -> Optional[int]:
        """Map Allen relation to precedence direction.

        Returns +1 if source precedes target, -1 if target precedes source,
        0 if concurrent/overlapping, None if ambiguous.
        """
        _map = {
            "before": 1, "meets": 1, "overlaps": 1, "starts": 0,
            "during": 0, "finishes": 0, "equal": 0,
            "after": -1, "met_by": -1, "overlapped_by": -1,
            "started_by": 0, "contains": 0, "finished_by": 0,
        }
        return _map.get(rel)

    @staticmethod
    def allen_to_stn_bounds(rel: str) -> Tuple[float, float]:
        """Map Allen relation to STN constraint bounds [lower, upper].

        Bounds represent: target_start - source_start ∈ [lower, upper].
        For events without known durations, we use qualitative bounds.
        """
        if rel in ("before", "meets"):
            return (0.0, float("inf"))
        elif rel in ("after", "met_by"):
            return (float("-inf"), 0.0)
        elif rel == "equal":
            return (0.0, 0.0)
        elif rel in ("starts", "started_by"):
            return (0.0, 0.0)
        elif rel in ("overlaps",):
            return (0.0, float("inf"))
        elif rel in ("overlapped_by",):
            return (float("-inf"), 0.0)
        elif rel in ("during",):
            return (0.0, float("inf"))
        elif rel in ("contains",):
            return (float("-inf"), 0.0)
        elif rel in ("finishes",):
            return (float("-inf"), float("inf"))
        elif rel in ("finished_by",):
            return (float("-inf"), float("inf"))
        return (float("-inf"), float("inf"))

    def __init__(self):
        self.allen_edges: List[Tuple[str, str, str, float]] = []
        self.stn_edges: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.node_ids: Set[str] = set()

    def add_allen_constraint(
        self, source: str, target: str, relation: str, confidence: float = 1.0
    ):
        self.allen_edges.append((source, target, relation, confidence))
        self.node_ids.add(source)
        self.node_ids.add(target)
        lo, hi = self.allen_to_stn_bounds(relation)
        self._add_stn_edge(source, target, lo, hi)

    def _add_stn_edge(self, src: str, tgt: str, lo: float, hi: float):
        """Add or tighten an STN edge."""
        key = (src, tgt)
        if key in self.stn_edges:
            old_lo, old_hi = self.stn_edges[key]
            lo = max(lo, old_lo)
            hi = min(hi, old_hi)
        self.stn_edges[key] = (lo, hi)

    def tighten_bounds(self) -> Tuple[bool, Dict[Tuple[str, str], Tuple[float, float]]]:
        """Bellman-Ford-style bound tightening on STN edges.

        Propagates: if A→B has bounds [l1, u1] and B→C has [l2, u2],
        then A→C should be within [l1+l2, u1+u2].
        Tightens existing bounds and detects inconsistencies.

        Returns (consistent, tightened_bounds).
        """
        nodes = sorted(self.node_ids)
        if not nodes:
            return True, {}

        dist_lo: Dict[Tuple[str, str], float] = {}
        dist_hi: Dict[Tuple[str, str], float] = {}

        for (src, tgt), (lo, hi) in self.stn_edges.items():
            dist_lo[(src, tgt)] = lo
            dist_hi[(src, tgt)] = hi

        changed = True
        iterations = 0
        max_iter = len(nodes) + 1
        while changed and iterations < max_iter:
            changed = False
            iterations += 1
            for (ab_s, ab_t), ab_lo in list(dist_lo.items()):
                ab_hi = dist_hi.get((ab_s, ab_t), float("inf"))
                for (bc_s, bc_t), bc_lo in list(dist_lo.items()):
                    if ab_t != bc_s:
                        continue
                    bc_hi = dist_hi.get((bc_s, bc_t), float("inf"))
                    new_lo = ab_lo + bc_lo if ab_lo != float("-inf") and bc_lo != float("-inf") else float("-inf")
                    new_hi = ab_hi + bc_hi if ab_hi != float("inf") and bc_hi != float("inf") else float("inf")

                    key = (ab_s, bc_t)
                    old_lo = dist_lo.get(key, float("-inf"))
                    old_hi = dist_hi.get(key, float("inf"))

                    tight_lo = max(old_lo, new_lo) if new_lo != float("-inf") else old_lo
                    tight_hi = min(old_hi, new_hi) if new_hi != float("inf") else old_hi

                    if tight_lo > old_lo or tight_hi < old_hi:
                        dist_lo[key] = tight_lo
                        dist_hi[key] = tight_hi
                        changed = True

        consistent = True
        for key in dist_lo:
            lo_val = dist_lo[key]
            hi_val = dist_hi.get(key, float("inf"))
            if lo_val > hi_val:
                consistent = False
                break
            if key[0] == key[1] and lo_val > 0:
                consistent = False
                break

        tightened = {}
        for key in dist_lo:
            tightened[key] = (dist_lo[key], dist_hi.get(key, float("inf")))

        return consistent, tightened

    def infer_allen_relations(self, max_hops: int = 3) -> List[Dict[str, Any]]:
        """Infer new Allen relations via composition table.

        If A rel1 B and B rel2 C, compose(rel1, rel2) gives the possible
        relations between A and C. When the result is a single relation
        or a small set, we emit high-confidence inferences.
        """
        self._init_composition_table()
        edge_map: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
        for src, tgt, rel, conf in self.allen_edges:
            edge_map[(src, tgt)].append((rel, conf))

        inferred: List[Dict[str, Any]] = []
        existing_pairs = {(s, t) for s, t, _, _ in self.allen_edges}

        for hop in range(max_hops - 1):
            new_edges = []
            for (ab_s, ab_t), ab_rels in list(edge_map.items()):
                for (bc_s, bc_t), bc_rels in list(edge_map.items()):
                    if ab_t != bc_s or ab_s == bc_t:
                        continue
                    if (ab_s, bc_t) in existing_pairs:
                        continue

                    for ab_rel, ab_conf in ab_rels:
                        for bc_rel, bc_conf in bc_rels:
                            composed = self.compose(ab_rel, bc_rel)
                            if not composed or len(composed) > 5:
                                continue

                            conf = ab_conf * bc_conf * 0.85
                            if len(composed) == 1:
                                result_rel = next(iter(composed))
                                inferred.append({
                                    "source": ab_s,
                                    "target": bc_t,
                                    "relation": result_rel,
                                    "confidence": round(conf, 4),
                                    "inference_type": "allen_composition",
                                    "via": ab_t,
                                    "chain": f"{ab_rel} ∘ {bc_rel} = {result_rel}",
                                    "hop": hop + 2,
                                })
                                new_edges.append((ab_s, bc_t, result_rel, conf))
                                existing_pairs.add((ab_s, bc_t))
                            elif len(composed) <= 3:
                                best = sorted(composed)[0]
                                inferred.append({
                                    "source": ab_s,
                                    "target": bc_t,
                                    "relation": best,
                                    "possible_relations": sorted(composed),
                                    "confidence": round(conf * 0.7, 4),
                                    "inference_type": "allen_composition_ambiguous",
                                    "via": ab_t,
                                    "chain": f"{ab_rel} ∘ {bc_rel} = {{{','.join(sorted(composed))}}}",
                                    "hop": hop + 2,
                                })
                                new_edges.append((ab_s, bc_t, best, conf * 0.7))
                                existing_pairs.add((ab_s, bc_t))

            for s, t, r, c in new_edges:
                edge_map[(s, t)].append((r, c))

        return inferred

    def check_consistency(self) -> Dict[str, Any]:
        """Check temporal consistency of all constraints.

        Detects:
        - Bound contradictions (lower > upper)
        - Positive self-loops (A must be before itself)
        - Conflicting Allen relations (A before B and B before A without intermediary)
        """
        consistent, bounds = self.tighten_bounds()
        violations = []

        for (src, tgt), (lo, hi) in bounds.items():
            if lo > hi:
                violations.append({
                    "type": "bound_contradiction",
                    "source": src,
                    "target": tgt,
                    "lower": lo,
                    "upper": hi,
                    "explanation": f"Impossible constraint: {src} to {tgt} requires [{lo}, {hi}]",
                })
            if src == tgt and lo > 0:
                violations.append({
                    "type": "positive_self_loop",
                    "node": src,
                    "lower": lo,
                    "explanation": f"{src} must precede itself (cycle detected)",
                })

        before_set = set()
        after_set = set()
        for src, tgt, rel, _ in self.allen_edges:
            if rel in ("before", "meets"):
                before_set.add((src, tgt))
            elif rel in ("after", "met_by"):
                after_set.add((src, tgt))

        for s, t in before_set:
            if (t, s) in before_set or (s, t) in after_set:
                violations.append({
                    "type": "conflicting_precedence",
                    "source": s,
                    "target": t,
                    "explanation": f"Conflicting: {s} both before and after {t}",
                })

        if not violations:
            reachable: Dict[str, Set[str]] = defaultdict(set)
            for s, t in before_set:
                reachable[s].add(t)
            changed_tc = True
            while changed_tc:
                changed_tc = False
                for s in list(reachable):
                    for t in list(reachable[s]):
                        for u in list(reachable.get(t, set())):
                            if u not in reachable[s]:
                                reachable[s].add(u)
                                changed_tc = True
            for s in reachable:
                if s in reachable[s]:
                    violations.append({
                        "type": "before_cycle",
                        "node": s,
                        "explanation": f"{s} transitively before itself (cycle in strict ordering)",
                    })
                    break

        return {
            "consistent": consistent and len(violations) == 0,
            "num_violations": len(violations),
            "violations": violations[:10],
            "num_constraints": len(self.stn_edges),
            "num_allen_edges": len(self.allen_edges),
        }

    def propagate(self) -> Dict[str, Any]:
        """Full constraint propagation: tighten bounds + infer Allen relations + check consistency."""
        inferred_relations = self.infer_allen_relations()
        consistent, tightened = self.tighten_bounds()
        consistency = self.check_consistency()

        precedence_order = []
        if consistent:
            before_graph: Dict[str, Set[str]] = defaultdict(set)
            for (src, tgt), (lo, _) in tightened.items():
                if lo > 0 or (lo == 0 and tightened.get((tgt, src), (0, 0))[1] < 0):
                    before_graph[src].add(tgt)
            for s, t, rel, _ in self.allen_edges:
                if rel in ("before", "meets"):
                    before_graph[s].add(t)

            in_deg = defaultdict(int)
            for node in self.node_ids:
                in_deg.setdefault(node, 0)
            for src, targets in before_graph.items():
                for t in targets:
                    in_deg[t] += 1

            queue = sorted(n for n in self.node_ids if in_deg[n] == 0)
            visited = set()
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                precedence_order.append(node)
                for nxt in sorted(before_graph.get(node, [])):
                    in_deg[nxt] -= 1
                    if in_deg[nxt] == 0:
                        queue.append(nxt)
            for node in self.node_ids:
                if node not in visited:
                    precedence_order.append(node)

        return {
            "consistent": consistency["consistent"],
            "num_constraints": len(self.stn_edges),
            "num_allen_edges": len(self.allen_edges),
            "num_nodes": len(self.node_ids),
            "inferred_relations": inferred_relations,
            "num_inferred": len(inferred_relations),
            "tightened_bounds": {
                f"{s}->{t}": {"lower": round(lo, 4) if lo != float("-inf") else "-inf",
                               "upper": round(hi, 4) if hi != float("inf") else "inf"}
                for (s, t), (lo, hi) in sorted(tightened.items())[:20]
            },
            "consistency_check": consistency,
            "precedence_order": precedence_order,
        }


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
        self.constraint_propagator = TemporalConstraintPropagator()

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
        allen_relation: str = None,
        confidence: float = 1.0,
    ):
        """Add STN-like temporal constraint: target - source in [lower, upper].

        If allen_relation is provided, also feeds it into the constraint propagator
        for Allen composition inference and Bellman-Ford bound tightening.
        """
        self.stn_constraints[(source_id, target_id)] = (lower_bound, upper_bound)
        self.add_relation(
            source_id,
            target_id,
            relation,
            metadata={"lower_bound": lower_bound, "upper_bound": upper_bound},
        )
        if allen_relation:
            self.constraint_propagator.add_allen_constraint(
                source_id, target_id, allen_relation, confidence
            )

    def propagate_constraints(self) -> Dict[str, Any]:
        """Full constraint propagation via Allen composition + Bellman-Ford STN tightening.

        Delegates to TemporalConstraintPropagator which performs:
        1. Multi-hop Allen relation inference via composition table
        2. Bellman-Ford-style STN bound tightening
        3. Consistency checking (cycle detection, bound contradictions)
        4. Topological ordering of events
        """
        if not self.constraint_propagator.allen_edges:
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

            return {
                "consistent": True,
                "inferred_before_edges": inferred,
                "num_constraints": len(self.stn_constraints),
                "num_allen_edges": 0,
                "num_inferred": len(inferred),
                "inferred_relations": [],
                "tightened_bounds": {},
                "precedence_order": [],
                "consistency_check": {"consistent": True, "num_violations": 0, "violations": []},
            }

        result = self.constraint_propagator.propagate()

        for inf in result.get("inferred_relations", []):
            src, tgt = inf["source"], inf["target"]
            rel = inf["relation"]
            self.add_relation(src, tgt, f"inferred_{rel}", metadata={
                "inferred": True,
                "confidence": inf.get("confidence", 0.5),
                "inference_type": inf.get("inference_type", "allen_composition"),
            })

        inferred_before = []
        for inf in result.get("inferred_relations", []):
            if inf["relation"] in ("before", "meets"):
                inferred_before.append((inf["source"], inf["target"]))

        result["inferred_before_edges"] = inferred_before
        result["num_constraints"] = len(self.stn_constraints)
        return result

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
            allen_str = etr.relation.value
            lo, hi = TemporalConstraintPropagator.allen_to_stn_bounds(allen_str)
            if etr.relation in (AllenRelation.BEFORE, AllenRelation.MEETS):
                graph.add_constraint(
                    src_id, tgt_id, lo, hi, relation="before",
                    allen_relation=allen_str, confidence=etr.confidence,
                )
            elif etr.relation in (AllenRelation.AFTER, AllenRelation.MET_BY):
                inv_lo, inv_hi = TemporalConstraintPropagator.allen_to_stn_bounds(
                    AllenRelation.inverse(etr.relation).value
                )
                graph.add_constraint(
                    tgt_id, src_id, inv_lo, inv_hi, relation="before",
                    allen_relation=AllenRelation.inverse(etr.relation).value,
                    confidence=etr.confidence,
                )
            elif etr.relation == AllenRelation.EQUAL:
                graph.add_constraint(
                    src_id, tgt_id, lo, hi, relation="equal",
                    allen_relation=allen_str, confidence=etr.confidence,
                )
            elif etr.relation in (AllenRelation.OVERLAPS, AllenRelation.OVERLAPPED_BY):
                graph.add_constraint(
                    src_id, tgt_id, lo, hi, relation=allen_str,
                    allen_relation=allen_str, confidence=etr.confidence,
                )
            elif etr.relation in (AllenRelation.STARTS, AllenRelation.STARTED_BY):
                graph.add_constraint(
                    src_id, tgt_id, lo, hi, relation=allen_str,
                    allen_relation=allen_str, confidence=etr.confidence,
                )
            elif etr.relation in (AllenRelation.DURING, AllenRelation.CONTAINS):
                graph.add_constraint(
                    src_id, tgt_id, lo, hi, relation=allen_str,
                    allen_relation=allen_str, confidence=etr.confidence,
                )
            elif etr.relation in (AllenRelation.FINISHES, AllenRelation.FINISHED_BY):
                graph.add_constraint(
                    src_id, tgt_id, lo, hi, relation=allen_str,
                    allen_relation=allen_str, confidence=etr.confidence,
                )

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
        cp = result.get("constraint_propagation", {})
        if cp.get("num_inferred", 0) > 0:
            temporal_ctx_parts.append(
                f"Constraint propagation: {cp['num_inferred']} inferred relations"
            )
        if cp.get("precedence_order"):
            temporal_ctx_parts.append(
                f"Precedence order: {' -> '.join(cp['precedence_order'][:8])}"
            )
        if not cp.get("consistent", True):
            temporal_ctx_parts.append("WARNING: temporal inconsistency detected")

        result["temporal_context"] = " | ".join(temporal_ctx_parts)

        return result
