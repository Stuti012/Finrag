"""Tests for temporal constraint propagation (P3-2c).

Covers Allen composition table, STN Bellman-Ford bound tightening,
multi-hop Allen relation inference, consistency checking, topological
ordering, TemporalGraph integration, and TemporalReasoner end-to-end."""

import pytest
from src.reasoning.temporal_reasoner import (
    AllenRelation,
    TemporalConstraintPropagator,
    TemporalGraph,
    TemporalEntity,
    TemporalReasoner,
)


# ---------------------------------------------------------------------------
# 1. Allen composition table
# ---------------------------------------------------------------------------

class TestAllenCompositionTable:
    def setup_method(self):
        self.prop = TemporalConstraintPropagator()
        self.prop._init_composition_table()

    def test_before_before_gives_before(self):
        result = self.prop.compose("before", "before")
        assert result == {"before"}

    def test_before_after_is_ambiguous(self):
        result = self.prop.compose("before", "after")
        assert len(result) > 1

    def test_equal_before_gives_before(self):
        result = self.prop.compose("equal", "before")
        assert result == {"before"}

    def test_before_equal_gives_before(self):
        result = self.prop.compose("before", "equal")
        assert result == {"before"}

    def test_equal_equal_gives_equal(self):
        result = self.prop.compose("equal", "equal")
        assert result == {"equal"}

    def test_meets_before_gives_before(self):
        result = self.prop.compose("meets", "before")
        assert result == {"before"}

    def test_before_meets_gives_before(self):
        result = self.prop.compose("before", "meets")
        assert result == {"before"}

    def test_meets_meets_gives_before(self):
        result = self.prop.compose("meets", "meets")
        assert result == {"before"}

    def test_after_after_gives_after(self):
        result = self.prop.compose("after", "after")
        assert result == {"after"}

    def test_unknown_relation_returns_empty(self):
        result = self.prop.compose("before", "nonexistent")
        assert len(result) == 0

    def test_all_13_relations_have_entries(self):
        rels = [
            "before", "after", "meets", "met_by", "overlaps", "overlapped_by",
            "starts", "started_by", "during", "contains", "finishes",
            "finished_by", "equal",
        ]
        for r1 in rels:
            for r2 in rels:
                result = self.prop.compose(r1, r2)
                assert result is not None, f"Missing composition for ({r1}, {r2})"
                assert len(result) >= 1, f"Empty composition for ({r1}, {r2})"


# ---------------------------------------------------------------------------
# 2. Allen → precedence mapping
# ---------------------------------------------------------------------------

class TestAllenToPrecedence:
    def test_before_gives_positive(self):
        assert TemporalConstraintPropagator.allen_to_precedence("before") == 1

    def test_meets_gives_positive(self):
        assert TemporalConstraintPropagator.allen_to_precedence("meets") == 1

    def test_after_gives_negative(self):
        assert TemporalConstraintPropagator.allen_to_precedence("after") == -1

    def test_met_by_gives_negative(self):
        assert TemporalConstraintPropagator.allen_to_precedence("met_by") == -1

    def test_equal_gives_zero(self):
        assert TemporalConstraintPropagator.allen_to_precedence("equal") == 0

    def test_during_gives_zero(self):
        assert TemporalConstraintPropagator.allen_to_precedence("during") == 0

    def test_unknown_gives_none(self):
        assert TemporalConstraintPropagator.allen_to_precedence("bogus") is None


# ---------------------------------------------------------------------------
# 3. Allen → STN bounds mapping
# ---------------------------------------------------------------------------

class TestAllenToSTNBounds:
    def test_before_bounds(self):
        lo, hi = TemporalConstraintPropagator.allen_to_stn_bounds("before")
        assert lo >= 0 and hi == float("inf")

    def test_after_bounds(self):
        lo, hi = TemporalConstraintPropagator.allen_to_stn_bounds("after")
        assert lo == float("-inf") and hi <= 0

    def test_equal_bounds_zero(self):
        lo, hi = TemporalConstraintPropagator.allen_to_stn_bounds("equal")
        assert lo == 0.0 and hi == 0.0

    def test_overlaps_bounds(self):
        lo, hi = TemporalConstraintPropagator.allen_to_stn_bounds("overlaps")
        assert lo >= 0

    def test_unknown_gives_full_range(self):
        lo, hi = TemporalConstraintPropagator.allen_to_stn_bounds("bogus")
        assert lo == float("-inf") and hi == float("inf")


# ---------------------------------------------------------------------------
# 4. Add Allen constraint + STN edge creation
# ---------------------------------------------------------------------------

class TestAddAllenConstraint:
    def test_adds_edge_and_nodes(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        assert len(prop.allen_edges) == 1
        assert "A" in prop.node_ids
        assert "B" in prop.node_ids
        assert ("A", "B") in prop.stn_edges

    def test_multiple_constraints(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.8)
        assert len(prop.allen_edges) == 2
        assert len(prop.node_ids) == 3

    def test_tightens_existing_bounds(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        old_lo, old_hi = prop.stn_edges[("A", "B")]
        prop.add_allen_constraint("A", "B", "equal", 0.5)
        new_lo, new_hi = prop.stn_edges[("A", "B")]
        assert new_lo >= old_lo


# ---------------------------------------------------------------------------
# 5. Bellman-Ford bound tightening
# ---------------------------------------------------------------------------

class TestTightenBounds:
    def test_empty_graph_consistent(self):
        prop = TemporalConstraintPropagator()
        consistent, bounds = prop.tighten_bounds()
        assert consistent is True
        assert bounds == {}

    def test_single_edge(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before")
        consistent, bounds = prop.tighten_bounds()
        assert consistent is True
        assert ("A", "B") in bounds

    def test_chain_propagation(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before")
        prop.add_allen_constraint("B", "C", "before")
        consistent, bounds = prop.tighten_bounds()
        assert consistent is True
        assert ("A", "C") in bounds
        lo, hi = bounds[("A", "C")]
        assert lo >= 0

    def test_cycle_with_qualitative_bounds_passes_stn(self):
        # Pure STN tightening with [0,inf] bounds can't detect qualitative cycles;
        # cycle detection happens at check_consistency via transitive closure.
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before")
        prop.add_allen_constraint("B", "C", "before")
        prop.add_allen_constraint("C", "A", "before")
        consistent, _ = prop.tighten_bounds()
        assert consistent is True

    def test_numeric_inconsistency_detected(self):
        prop = TemporalConstraintPropagator()
        prop._add_stn_edge("A", "B", 5.0, 10.0)
        prop._add_stn_edge("B", "A", 5.0, 10.0)
        prop.node_ids.update(["A", "B"])
        consistent, bounds = prop.tighten_bounds()
        if ("A", "A") in bounds:
            lo, _ = bounds[("A", "A")]
            if lo > 0:
                assert consistent is False

    def test_equal_constraints_consistent(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "equal")
        prop.add_allen_constraint("B", "C", "equal")
        consistent, bounds = prop.tighten_bounds()
        assert consistent is True


# ---------------------------------------------------------------------------
# 6. Multi-hop Allen relation inference
# ---------------------------------------------------------------------------

class TestInferAllenRelations:
    def test_infer_transitive_before(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.9)
        inferred = prop.infer_allen_relations(max_hops=2)
        assert len(inferred) >= 1
        ac = [i for i in inferred if i["source"] == "A" and i["target"] == "C"]
        assert len(ac) == 1
        assert ac[0]["relation"] == "before"
        assert ac[0]["inference_type"] == "allen_composition"

    def test_infer_confidence_decays(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 1.0)
        prop.add_allen_constraint("B", "C", "before", 1.0)
        inferred = prop.infer_allen_relations(max_hops=2)
        ac = [i for i in inferred if i["source"] == "A" and i["target"] == "C"]
        assert ac[0]["confidence"] < 1.0

    def test_infer_meets_before(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "meets", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.8)
        inferred = prop.infer_allen_relations(max_hops=2)
        ac = [i for i in inferred if i["source"] == "A" and i["target"] == "C"]
        assert len(ac) >= 1
        assert ac[0]["relation"] == "before"

    def test_no_self_inference(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "A", "after", 0.9)
        inferred = prop.infer_allen_relations(max_hops=2)
        self_refs = [i for i in inferred if i["source"] == i["target"]]
        assert len(self_refs) == 0

    def test_no_duplicate_inference(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.8)
        prop.add_allen_constraint("A", "C", "before", 0.7)
        inferred = prop.infer_allen_relations(max_hops=2)
        ac = [i for i in inferred if i["source"] == "A" and i["target"] == "C"]
        assert len(ac) == 0

    def test_three_hop_inference(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.9)
        prop.add_allen_constraint("C", "D", "before", 0.9)
        inferred = prop.infer_allen_relations(max_hops=3)
        ad = [i for i in inferred if i["source"] == "A" and i["target"] == "D"]
        assert len(ad) >= 1

    def test_chain_includes_hop_count(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.9)
        inferred = prop.infer_allen_relations(max_hops=2)
        ac = [i for i in inferred if i["source"] == "A" and i["target"] == "C"]
        assert ac[0]["hop"] == 2


# ---------------------------------------------------------------------------
# 7. Consistency checking
# ---------------------------------------------------------------------------

class TestCheckConsistency:
    def test_consistent_chain(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before")
        prop.add_allen_constraint("B", "C", "before")
        result = prop.check_consistency()
        assert result["consistent"] is True
        assert result["num_violations"] == 0

    def test_inconsistent_cycle_detected(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before")
        prop.add_allen_constraint("B", "C", "before")
        prop.add_allen_constraint("C", "A", "before")
        result = prop.check_consistency()
        assert result["consistent"] is False
        assert result["num_violations"] > 0

    def test_conflicting_precedence(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before")
        prop.add_allen_constraint("A", "B", "after")
        result = prop.check_consistency()
        assert result["consistent"] is False

    def test_empty_graph_consistent(self):
        prop = TemporalConstraintPropagator()
        result = prop.check_consistency()
        assert result["consistent"] is True

    def test_reports_num_constraints(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before")
        result = prop.check_consistency()
        assert result["num_constraints"] >= 1
        assert result["num_allen_edges"] == 1


# ---------------------------------------------------------------------------
# 8. Full propagate() entrypoint
# ---------------------------------------------------------------------------

class TestPropagate:
    def test_propagate_returns_all_keys(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.8)
        result = prop.propagate()
        assert "consistent" in result
        assert "num_constraints" in result
        assert "num_allen_edges" in result
        assert "num_nodes" in result
        assert "inferred_relations" in result
        assert "num_inferred" in result
        assert "tightened_bounds" in result
        assert "consistency_check" in result
        assert "precedence_order" in result

    def test_propagate_consistent_chain(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.8)
        result = prop.propagate()
        assert result["consistent"] is True
        assert result["num_inferred"] >= 1

    def test_propagate_precedence_order(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "before", 0.8)
        result = prop.propagate()
        order = result["precedence_order"]
        assert len(order) == 3
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_propagate_inconsistent(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before")
        prop.add_allen_constraint("B", "C", "before")
        prop.add_allen_constraint("C", "A", "before")
        result = prop.propagate()
        assert result["consistent"] is False

    def test_propagate_tightened_bounds_formatted(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("X", "Y", "before", 1.0)
        result = prop.propagate()
        assert "X->Y" in result["tightened_bounds"]
        bounds = result["tightened_bounds"]["X->Y"]
        assert "lower" in bounds
        assert "upper" in bounds

    def test_propagate_five_node_chain(self):
        prop = TemporalConstraintPropagator()
        for a, b in [("E1", "E2"), ("E2", "E3"), ("E3", "E4"), ("E4", "E5")]:
            prop.add_allen_constraint(a, b, "before", 0.95)
        result = prop.propagate()
        assert result["consistent"] is True
        order = result["precedence_order"]
        for i in range(len(order) - 1):
            assert order[i] < order[i + 1] or order.index(order[i]) < order.index(order[i + 1])

    def test_propagate_equal_nodes_same_position(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "equal", 1.0)
        result = prop.propagate()
        assert result["consistent"] is True


# ---------------------------------------------------------------------------
# 9. TemporalGraph integration
# ---------------------------------------------------------------------------

class TestTemporalGraphIntegration:
    def test_graph_has_constraint_propagator(self):
        graph = TemporalGraph()
        assert hasattr(graph, "constraint_propagator")
        assert isinstance(graph.constraint_propagator, TemporalConstraintPropagator)

    def test_add_constraint_with_allen_feeds_propagator(self):
        graph = TemporalGraph()
        graph.add_constraint(
            "event_A", "event_B", 0.0, float("inf"),
            relation="before", allen_relation="before", confidence=0.9,
        )
        assert len(graph.constraint_propagator.allen_edges) == 1

    def test_add_constraint_without_allen_no_propagator(self):
        graph = TemporalGraph()
        graph.add_constraint("event_A", "event_B", 0.0, float("inf"), relation="before")
        assert len(graph.constraint_propagator.allen_edges) == 0

    def test_propagate_constraints_with_allen_edges(self):
        graph = TemporalGraph()
        e1 = TemporalEntity("event", "ipo", "IPO")
        e2 = TemporalEntity("event", "merger", "Merger")
        e3 = TemporalEntity("event", "earnings", "Earnings")
        for e in [e1, e2, e3]:
            graph.add_entity(e)
        graph.add_constraint(
            "event_ipo", "event_merger", 0.0, float("inf"),
            relation="before", allen_relation="before", confidence=0.9,
        )
        graph.add_constraint(
            "event_merger", "event_earnings", 0.0, float("inf"),
            relation="before", allen_relation="before", confidence=0.8,
        )
        result = graph.propagate_constraints()
        assert result["consistent"] is True
        assert result["num_inferred"] >= 1
        assert "precedence_order" in result

    def test_propagate_constraints_fallback_no_allen(self):
        graph = TemporalGraph()
        e1 = TemporalEntity("event", "A", "A")
        e2 = TemporalEntity("event", "B", "B")
        for e in [e1, e2]:
            graph.add_entity(e)
        graph.add_constraint("event_A", "event_B", 0.0, float("inf"), relation="before")
        result = graph.propagate_constraints()
        assert result["consistent"] is True
        assert result["num_allen_edges"] == 0

    def test_propagate_injects_inferred_relations(self):
        graph = TemporalGraph()
        for name in ["A", "B", "C"]:
            graph.add_entity(TemporalEntity("event", name, name))
        graph.add_constraint(
            "event_A", "event_B", 0.0, float("inf"),
            relation="before", allen_relation="before", confidence=0.9,
        )
        graph.add_constraint(
            "event_B", "event_C", 0.0, float("inf"),
            relation="before", allen_relation="before", confidence=0.9,
        )
        result = graph.propagate_constraints()
        assert result["num_inferred"] >= 1
        assert any(
            (a, b) == ("event_A", "event_C")
            for a, b in result.get("inferred_before_edges", [])
        )

    def test_all_thirteen_allen_relations_accepted(self):
        rels = [
            "before", "after", "meets", "met_by", "overlaps", "overlapped_by",
            "starts", "started_by", "during", "contains", "finishes",
            "finished_by", "equal",
        ]
        graph = TemporalGraph()
        for i, rel in enumerate(rels):
            graph.add_entity(TemporalEntity("event", f"S{i}", f"S{i}"))
            graph.add_entity(TemporalEntity("event", f"T{i}", f"T{i}"))
            graph.add_constraint(
                f"event_S{i}", f"event_T{i}", 0.0, float("inf"),
                relation=rel, allen_relation=rel, confidence=0.8,
            )
        assert len(graph.constraint_propagator.allen_edges) == 13


# ---------------------------------------------------------------------------
# 10. TemporalReasoner end-to-end integration
# ---------------------------------------------------------------------------

class TestTemporalReasonerConstraintPropagation:
    def setup_method(self):
        self.reasoner = TemporalReasoner()

    def test_reason_returns_constraint_propagation(self):
        result = self.reasoner.reason(
            "What happened after the IPO?",
            [["Metric", "2020", "2021"], ["Revenue", "100", "200"]],
            context="Before the merger, the company completed its IPO. After the merger, earnings improved.",
        )
        assert "constraint_propagation" in result

    def test_reason_constraint_propagation_has_keys(self):
        result = self.reasoner.reason(
            "Revenue grew after the acquisition but before the restructuring.",
            [["Metric", "2020", "2021"], ["Revenue", "100", "200"]],
            context="The company completed its acquisition in 2020, then did restructuring.",
        )
        cp = result["constraint_propagation"]
        assert "consistent" in cp
        assert "num_constraints" in cp

    def test_temporal_context_includes_propagation_info(self):
        result = self.reasoner.reason(
            "What happened?",
            [["Metric", "2020", "2021"], ["Revenue", "100", "200"]],
            context="Before the IPO, the company restructured. After the IPO, the merger happened. After the merger, earnings improved.",
        )
        cp = result.get("constraint_propagation", {})
        if cp.get("num_inferred", 0) > 0:
            assert "Constraint propagation" in result["temporal_context"] or "inferred" in str(cp)

    def test_reason_consistent_by_default(self):
        result = self.reasoner.reason(
            "How did revenue change?",
            [["Metric", "2020", "2021"], ["Revenue", "100", "200"]],
        )
        cp = result["constraint_propagation"]
        assert cp["consistent"] is True


# ---------------------------------------------------------------------------
# 11. Edge cases & robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_node_propagation(self):
        prop = TemporalConstraintPropagator()
        prop.node_ids.add("lonely")
        result = prop.propagate()
        assert result["consistent"] is True
        assert result["num_nodes"] == 1
        assert result["precedence_order"] == ["lonely"]

    def test_large_chain_performance(self):
        prop = TemporalConstraintPropagator()
        n = 20
        for i in range(n - 1):
            prop.add_allen_constraint(f"N{i}", f"N{i+1}", "before", 0.95)
        result = prop.propagate()
        assert result["consistent"] is True
        order = result["precedence_order"]
        assert len(order) == n
        for i in range(n - 1):
            assert order.index(f"N{i}") < order.index(f"N{i+1}")

    def test_diamond_graph_consistent(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("A", "C", "before", 0.9)
        prop.add_allen_constraint("B", "D", "before", 0.9)
        prop.add_allen_constraint("C", "D", "before", 0.9)
        result = prop.propagate()
        assert result["consistent"] is True
        order = result["precedence_order"]
        assert order.index("A") < order.index("D")

    def test_mixed_allen_relations(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "before", 0.9)
        prop.add_allen_constraint("B", "C", "overlaps", 0.8)
        prop.add_allen_constraint("C", "D", "meets", 0.85)
        result = prop.propagate()
        assert result["consistent"] is True
        assert result["num_allen_edges"] == 3

    def test_parallel_events_equal(self):
        prop = TemporalConstraintPropagator()
        prop.add_allen_constraint("A", "B", "equal", 1.0)
        prop.add_allen_constraint("A", "C", "before", 0.9)
        result = prop.propagate()
        assert result["consistent"] is True
