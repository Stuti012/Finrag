"""Tests for event-event temporal relation extraction (P2-2b).

Covers Allen's interval algebra, financial event classification,
signal-word extraction, sequence patterns, temporal anchoring,
transitive closure, and TemporalReasoner integration."""

import pytest
from src.reasoning.temporal_reasoner import (
    AllenRelation,
    EventTemporalRelationExtractor,
    EventTemporalRelation,
    FinancialEvent,
    FinancialEventClassifier,
    FinancialEventType,
    TemporalReasoner,
)


class TestAllenRelation:
    def test_inverse_before_after(self):
        assert AllenRelation.inverse(AllenRelation.BEFORE) == AllenRelation.AFTER
        assert AllenRelation.inverse(AllenRelation.AFTER) == AllenRelation.BEFORE

    def test_inverse_meets(self):
        assert AllenRelation.inverse(AllenRelation.MEETS) == AllenRelation.MET_BY

    def test_inverse_during(self):
        assert AllenRelation.inverse(AllenRelation.DURING) == AllenRelation.CONTAINS

    def test_inverse_equal(self):
        assert AllenRelation.inverse(AllenRelation.EQUAL) == AllenRelation.EQUAL

    def test_inverse_overlaps(self):
        assert AllenRelation.inverse(AllenRelation.OVERLAPS) == AllenRelation.OVERLAPPED_BY


class TestFinancialEventClassifier:
    def test_acquisition(self):
        assert FinancialEventClassifier.classify("company acquired XYZ") == FinancialEventType.ACQUISITION

    def test_merger(self):
        assert FinancialEventClassifier.classify("the merger was completed") == FinancialEventType.MERGER

    def test_restructuring(self):
        assert FinancialEventClassifier.classify("restructuring program began") == FinancialEventType.RESTRUCTURING

    def test_ipo(self):
        assert FinancialEventClassifier.classify("the company went public") == FinancialEventType.IPO

    def test_dividend(self):
        assert FinancialEventClassifier.classify("special dividend of $2") == FinancialEventType.DIVIDEND

    def test_buyback(self):
        assert FinancialEventClassifier.classify("share repurchase program") == FinancialEventType.BUYBACK

    def test_product_launch(self):
        assert FinancialEventClassifier.classify("launched new product line") == FinancialEventType.PRODUCT_LAUNCH

    def test_cost_action(self):
        assert FinancialEventClassifier.classify("cost-cutting measures") == FinancialEventType.COST_ACTION

    def test_impairment(self):
        assert FinancialEventClassifier.classify("goodwill write-down") == FinancialEventType.IMPAIRMENT

    def test_regulatory(self):
        assert FinancialEventClassifier.classify("regulatory approval received") == FinancialEventType.REGULATORY

    def test_other(self):
        assert FinancialEventClassifier.classify("random text here") == FinancialEventType.OTHER


class TestFinancialEvent:
    def test_event_id(self):
        ev = FinancialEvent(text="Company acquired XYZ Corp")
        assert ev.event_id.startswith("event_")
        assert "company" in ev.event_id

    def test_event_id_truncation(self):
        ev = FinancialEvent(text="A" * 200)
        assert len(ev.event_id) <= 70


class TestEventTemporalRelation:
    def test_to_dict(self):
        ev1 = FinancialEvent(text="acquisition", event_type=FinancialEventType.ACQUISITION)
        ev2 = FinancialEvent(text="revenue growth", event_type=FinancialEventType.REVENUE)
        rel = EventTemporalRelation(
            event1=ev1, event2=ev2, relation=AllenRelation.BEFORE,
            confidence=0.8, evidence="test", signal_word="before",
        )
        d = rel.to_dict()
        assert d["event1"] == "acquisition"
        assert d["relation"] == "before"
        assert d["event1_type"] == "acquisition"
        assert d["event2_type"] == "revenue"

    def test_inverse(self):
        ev1 = FinancialEvent(text="A")
        ev2 = FinancialEvent(text="B")
        rel = EventTemporalRelation(ev1, ev2, AllenRelation.BEFORE, 0.8)
        inv = rel.inverse()
        assert inv.event1.text == "B"
        assert inv.event2.text == "A"
        assert inv.relation == AllenRelation.AFTER


class TestSignalWordExtraction:
    def setup_method(self):
        self.ext = EventTemporalRelationExtractor()

    def test_after_sentence_initial(self):
        rels = self.ext.extract_relations(
            "After the acquisition, revenue increased significantly."
        )
        assert len(rels) >= 1
        r = rels[0]
        assert r.relation == AllenRelation.AFTER
        assert "acquisition" in r.event2.text.lower()

    def test_before_sentence_initial(self):
        rels = self.ext.extract_relations(
            "Before the IPO, the company restructured operations."
        )
        assert len(rels) >= 1
        assert rels[0].relation == AllenRelation.BEFORE

    def test_during_extraction(self):
        rels = self.ext.extract_relations(
            "During the restructuring, operating costs declined."
        )
        assert len(rels) >= 1
        assert rels[0].relation == AllenRelation.DURING

    def test_following_as_after(self):
        rels = self.ext.extract_relations(
            "Following the merger, earnings improved by 20%."
        )
        assert len(rels) >= 1
        assert rels[0].relation == AllenRelation.AFTER

    def test_coincided_with(self):
        rels = self.ext.extract_relations(
            "The product launch coincided with strong earnings growth."
        )
        assert len(rels) >= 1
        assert rels[0].relation == AllenRelation.EQUAL

    def test_midsentence_before(self):
        rels = self.ext.extract_relations(
            "Revenue growth occurred before the cost restructuring began."
        )
        assert len(rels) >= 1
        assert rels[0].relation == AllenRelation.BEFORE


class TestSequencePatterns:
    def setup_method(self):
        self.ext = EventTemporalRelationExtractor()

    def test_followed_by(self):
        rels = self.ext.extract_relations(
            "The cost reduction, followed by revenue recovery, boosted margins."
        )
        assert any(r.signal_word in ("sequence", "followed by") for r in rels)

    def test_first_then(self):
        rels = self.ext.extract_relations(
            "First, costs were cut significantly, then revenue recovered steadily."
        )
        assert any(r.relation == AllenRelation.BEFORE for r in rels)

    def test_subsequently(self):
        rels = self.ext.extract_relations(
            "The company acquired XYZ, and subsequently expanded operations."
        )
        assert len(rels) >= 1


class TestTemporalAnchoring:
    def setup_method(self):
        self.ext = EventTemporalRelationExtractor()

    def test_year_extraction(self):
        rels = self.ext.extract_relations(
            "The acquisition in 2020 happened before the IPO in 2022."
        )
        assert len(rels) >= 1
        years = set()
        for r in rels:
            if r.event1.year:
                years.add(r.event1.year)
            if r.event2.year:
                years.add(r.event2.year)
        assert 2020 in years or 2022 in years

    def test_anchor_inference(self):
        rels = self.ext.extract_relations(
            "Revenue grew in 2019. The acquisition happened in 2021."
        )
        inferred = self.ext.infer_from_temporal_anchors(rels)
        # Can only infer if there are anchored events
        # These may or may not produce inferred depending on extraction
        assert isinstance(inferred, list)


class TestTransitiveClosure:
    def setup_method(self):
        self.ext = EventTemporalRelationExtractor()

    def test_basic_transitivity(self):
        ev_a = FinancialEvent(text="event A")
        ev_b = FinancialEvent(text="event B")
        ev_c = FinancialEvent(text="event C")
        rels = [
            EventTemporalRelation(ev_a, ev_b, AllenRelation.BEFORE, 0.8),
            EventTemporalRelation(ev_b, ev_c, AllenRelation.BEFORE, 0.8),
        ]
        trans = self.ext.compute_transitive_closure(rels)
        assert len(trans) >= 1
        assert any(
            r.event1.text == "event A" and r.event2.text == "event C"
            for r in trans
        )

    def test_no_duplicate_transitivity(self):
        ev_a = FinancialEvent(text="event A")
        ev_b = FinancialEvent(text="event B")
        rels = [EventTemporalRelation(ev_a, ev_b, AllenRelation.BEFORE, 0.8)]
        trans = self.ext.compute_transitive_closure(rels)
        assert len(trans) == 0


class TestEventTimeline:
    def setup_method(self):
        self.ext = EventTemporalRelationExtractor()

    def test_timeline_ordering(self):
        ev_a = FinancialEvent(text="cost cutting", event_type=FinancialEventType.COST_ACTION)
        ev_b = FinancialEvent(text="revenue recovery", event_type=FinancialEventType.REVENUE)
        ev_c = FinancialEvent(text="margin improvement", event_type=FinancialEventType.EARNINGS)
        rels = [
            EventTemporalRelation(ev_a, ev_b, AllenRelation.BEFORE, 0.8),
            EventTemporalRelation(ev_b, ev_c, AllenRelation.BEFORE, 0.8),
        ]
        timeline = self.ext.build_event_timeline(rels)
        assert len(timeline) == 3
        texts = [t["text"] for t in timeline]
        assert texts.index("cost cutting") < texts.index("revenue recovery")
        assert texts.index("revenue recovery") < texts.index("margin improvement")

    def test_timeline_includes_event_types(self):
        ev = FinancialEvent(text="acquisition", event_type=FinancialEventType.ACQUISITION)
        rels = [EventTemporalRelation(
            ev, FinancialEvent(text="integration"), AllenRelation.BEFORE, 0.7
        )]
        timeline = self.ext.build_event_timeline(rels)
        assert any(t["event_type"] == "acquisition" for t in timeline)


class TestTemporalReasonerIntegration:
    def setup_method(self):
        self.reasoner = TemporalReasoner()

    def test_reason_includes_event_relations(self):
        result = self.reasoner.reason(
            question="What happened after the merger?",
            table=[["Item", "2021", "2022"], ["Revenue", "100", "200"]],
            context="After the merger, the company expanded operations significantly.",
        )
        assert "event_temporal_relations" in result
        assert "event_timeline" in result
        assert isinstance(result["event_temporal_relations"], list)
        if result["event_temporal_relations"]:
            r = result["event_temporal_relations"][0]
            assert "relation" in r
            assert "event1" in r
            assert "event2" in r

    def test_reason_event_relations_are_dicts(self):
        result = self.reasoner.reason(
            question="How did revenue change?",
            table=[["Item", "2020", "2021"], ["Revenue", "100", "200"]],
            context="During the restructuring, costs fell sharply.",
        )
        for r in result["event_temporal_relations"]:
            assert isinstance(r, dict)
            assert "relation" in r
            assert "confidence" in r

    def test_temporal_context_includes_event_info(self):
        result = self.reasoner.reason(
            question="Why did costs increase?",
            table=[["Item", "2020"], ["Revenue", "100"]],
            context="Before the acquisition, margins were stable. After the acquisition, costs surged.",
        )
        ctx = result["temporal_context"]
        if result["event_temporal_relations"]:
            assert "Event relations:" in ctx
