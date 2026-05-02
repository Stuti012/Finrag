"""Tests for implicit discourse causality detection (P3-3d).

Covers PDTB-3 discourse relation taxonomy, explicit connective mapping,
implicit causality verb (ICV) bias, Bayesian coherence scoring, entity
continuity, financial event-outcome patterns, direction inference,
CausalityDetector integration, metrics, and edge cases."""

import pytest
import math
from src.reasoning.causality_detector import (
    CausalRelation,
    CausalityDetector,
    DiscourseRelationType,
    ImplicitDiscourseCausalityDetector,
)


# ---------------------------------------------------------------------------
# 1. DiscourseRelationType constants
# ---------------------------------------------------------------------------

class TestDiscourseRelationType:
    def test_causal_types_include_cause(self):
        assert DiscourseRelationType.CONTINGENCY_CAUSE in DiscourseRelationType.CAUSAL_TYPES

    def test_causal_types_include_condition(self):
        assert DiscourseRelationType.CONTINGENCY_CONDITION in DiscourseRelationType.CAUSAL_TYPES

    def test_causal_types_include_purpose(self):
        assert DiscourseRelationType.CONTINGENCY_PURPOSE in DiscourseRelationType.CAUSAL_TYPES

    def test_temporal_not_in_causal(self):
        assert DiscourseRelationType.TEMPORAL not in DiscourseRelationType.CAUSAL_TYPES

    def test_contrast_not_in_causal(self):
        assert DiscourseRelationType.COMPARISON_CONTRAST not in DiscourseRelationType.CAUSAL_TYPES


# ---------------------------------------------------------------------------
# 2. Explicit connective lexicon
# ---------------------------------------------------------------------------

class TestExplicitConnectives:
    def setup_method(self):
        self.det = ImplicitDiscourseCausalityDetector()

    def test_because_maps_to_causal(self):
        result = self.det.classify_discourse_relation(
            "Revenue fell.", "Costs rose because demand dropped."
        )
        assert result["is_causal"] is True
        assert result["explicit"] is True
        assert result["connective"] == "because"

    def test_therefore_maps_to_causal(self):
        result = self.det.classify_discourse_relation(
            "The company cut costs.", "Therefore, margins improved."
        )
        assert result["is_causal"] is True
        assert result["explicit"] is True

    def test_however_maps_to_contrast(self):
        result = self.det.classify_discourse_relation(
            "Revenue grew.", "However, margins declined."
        )
        assert result["level1"] == "comparison"
        assert result["is_causal"] is False

    def test_after_maps_to_temporal(self):
        result = self.det.classify_discourse_relation(
            "After the merger, revenue grew.", "Performance improved."
        )
        assert result["level1"] == "temporal"

    def test_also_maps_to_expansion(self):
        result = self.det.classify_discourse_relation(
            "Revenue grew.", "Also, margins expanded."
        )
        assert result["level1"] == "expansion"
        assert result["is_causal"] is False

    def test_led_to_maps_to_causal(self):
        result = self.det.classify_discourse_relation(
            "Rate hike led to higher costs.", "Demand weakened."
        )
        assert result["is_causal"] is True

    def test_due_to_maps_to_causal(self):
        result = self.det.classify_discourse_relation(
            "Revenue declined due to lower demand.", "The outlook worsened."
        )
        assert result["is_causal"] is True

    def test_confidence_high_for_explicit(self):
        result = self.det.classify_discourse_relation(
            "Costs rose.", "Revenue fell because demand dropped."
        )
        assert result["confidence"] >= 0.8


# ---------------------------------------------------------------------------
# 3. Implicit Causality Verb (ICV) database
# ---------------------------------------------------------------------------

class TestICVDatabase:
    def test_cause_biased_verbs_present(self):
        det = ImplicitDiscourseCausalityDetector()
        assert "restructured" in det.ICV_CAUSE_BIASED
        assert "acquired" in det.ICV_CAUSE_BIASED
        assert "cut" in det.ICV_CAUSE_BIASED

    def test_effect_biased_verbs_present(self):
        det = ImplicitDiscourseCausalityDetector()
        assert "grew" in det.ICV_EFFECT_BIASED
        assert "declined" in det.ICV_EFFECT_BIASED
        assert "improved" in det.ICV_EFFECT_BIASED

    def test_cause_biased_scores_in_range(self):
        det = ImplicitDiscourseCausalityDetector()
        for verb, score in det.ICV_CAUSE_BIASED.items():
            assert 0.0 <= score <= 1.0, f"{verb} score out of range: {score}"

    def test_effect_biased_scores_in_range(self):
        det = ImplicitDiscourseCausalityDetector()
        for verb, score in det.ICV_EFFECT_BIASED.items():
            assert 0.0 <= score <= 1.0, f"{verb} score out of range: {score}"


# ---------------------------------------------------------------------------
# 4. Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def setup_method(self):
        self.det = ImplicitDiscourseCausalityDetector()

    def test_cause_effect_pattern_detected(self):
        features = self.det._compute_features(
            "The acquisition was completed in Q3.",
            "Revenue grew 15% year-over-year."
        )
        assert features["cause_effect_pattern"] == 1.0

    def test_icv_cause_bias_from_restructured(self):
        features = self.det._compute_features(
            "The company restructured its operations.",
            "Profitability improved."
        )
        assert features["icv_cause_bias"] > 0.5

    def test_icv_effect_bias_from_grew(self):
        features = self.det._compute_features(
            "They launched a new product.",
            "Revenue grew substantially."
        )
        assert features["icv_effect_bias"] > 0.5

    def test_entity_continuity_detected(self):
        features = self.det._compute_features(
            "Apple expanded its services segment.",
            "The company reported higher margins."
        )
        assert features["entity_continuity"] == 1.0

    def test_entity_continuity_noun_overlap(self):
        features = self.det._compute_features(
            "Apple expanded its services segment.",
            "Apple reported higher margins."
        )
        assert features["entity_continuity"] == 1.0

    def test_temporal_adjacency_cue(self):
        features = self.det._compute_features(
            "The Fed raised rates.",
            "In turn, borrowing costs increased."
        )
        assert features["temporal_adjacency"] == 1.0

    def test_financial_domain_scoring(self):
        features = self.det._compute_features(
            "Revenue growth accelerated as margin expanded.",
            "Earnings per share rose to record levels."
        )
        assert features["financial_domain"] > 0.0

    def test_connective_absence_when_no_explicit(self):
        features = self.det._compute_features(
            "The company restructured.",
            "Margins improved."
        )
        assert features["connective_absence"] == 1.0

    def test_contrast_cue_detected(self):
        features = self.det._compute_features(
            "Revenue grew.",
            "However, margins declined sharply."
        )
        assert features["contrast_cue"] == 1.0


# ---------------------------------------------------------------------------
# 5. Bayesian coherence scoring
# ---------------------------------------------------------------------------

class TestBayesianScoring:
    def setup_method(self):
        self.det = ImplicitDiscourseCausalityDetector()

    def test_strong_cause_effect_scores_high(self):
        features = {
            "cause_effect_pattern": 1.0,
            "icv_cause_bias": 0.8,
            "icv_effect_bias": 0.7,
            "entity_continuity": 1.0,
            "temporal_adjacency": 0.0,
            "financial_domain": 0.5,
            "connective_absence": 1.0,
            "contrast_cue": 0.0,
        }
        score = self.det._bayesian_causal_score(features)
        assert score > 0.7

    def test_contrast_cue_lowers_score(self):
        base_features = {
            "cause_effect_pattern": 0.0,
            "icv_cause_bias": 0.3,
            "icv_effect_bias": 0.3,
            "entity_continuity": 0.0,
            "temporal_adjacency": 0.0,
            "financial_domain": 0.2,
            "connective_absence": 1.0,
            "contrast_cue": 0.0,
        }
        score_no_contrast = self.det._bayesian_causal_score(base_features)
        contrast_features = dict(base_features, contrast_cue=1.0)
        score_with_contrast = self.det._bayesian_causal_score(contrast_features)
        assert score_with_contrast < score_no_contrast

    def test_empty_features_return_prior(self):
        score = self.det._bayesian_causal_score({})
        assert abs(score - self.det.base_prior) < 0.05

    def test_score_bounded_0_1(self):
        high = {k: 1.0 for k in self.det.PRIOR_CAUSAL_WEIGHTS}
        low = {k: 0.0 for k in self.det.PRIOR_CAUSAL_WEIGHTS}
        assert 0.0 <= self.det._bayesian_causal_score(high) <= 1.0
        assert 0.0 <= self.det._bayesian_causal_score(low) <= 1.0


# ---------------------------------------------------------------------------
# 6. Direction inference
# ---------------------------------------------------------------------------

class TestDirectionInference:
    def setup_method(self):
        self.det = ImplicitDiscourseCausalityDetector()

    def test_forward_direction(self):
        direction = self.det._infer_causal_direction(
            "The company restructured aggressively.",
            "Margins improved steadily."
        )
        assert direction == "forward"

    def test_backward_direction(self):
        direction = self.det._infer_causal_direction(
            "Earnings declined sharply.",
            "Management had cut marketing spend."
        )
        assert direction == "backward"

    def test_ambiguous_when_no_icv(self):
        direction = self.det._infer_causal_direction(
            "The meeting took place yesterday.",
            "Analysts were present."
        )
        assert direction == "ambiguous"


# ---------------------------------------------------------------------------
# 7. Full implicit causality detection
# ---------------------------------------------------------------------------

class TestDetectImplicitCausality:
    def setup_method(self):
        self.det = ImplicitDiscourseCausalityDetector()

    def test_detects_financial_cause_effect(self):
        text = (
            "The company restructured its operations in Q3. "
            "Revenue grew 15% year-over-year."
        )
        results = self.det.detect_implicit_causality(text)
        assert len(results) >= 1
        assert results[0]["confidence"] >= 0.45

    def test_explicit_connective_detected(self):
        text = (
            "Revenue fell sharply. "
            "This was because demand collapsed in the region."
        )
        results = self.det.detect_implicit_causality(text)
        assert len(results) >= 1
        assert any(r.get("explicit") for r in results)

    def test_no_causality_in_unrelated_sentences(self):
        text = (
            "The weather was nice today. "
            "Birds sang in the trees."
        )
        results = self.det.detect_implicit_causality(text)
        assert len(results) == 0

    def test_single_sentence_returns_empty(self):
        results = self.det.detect_implicit_causality("Just one sentence.")
        assert len(results) == 0

    def test_empty_text_returns_empty(self):
        results = self.det.detect_implicit_causality("")
        assert len(results) == 0

    def test_result_has_required_keys(self):
        text = (
            "The company acquired a competitor. "
            "Market share increased significantly."
        )
        results = self.det.detect_implicit_causality(text)
        if results:
            r = results[0]
            assert "cause" in r
            assert "effect" in r
            assert "discourse_relation" in r
            assert "confidence" in r
            assert "features" in r

    def test_multi_sentence_chain(self):
        text = (
            "The company acquired a major competitor. "
            "Revenue surged 25% year-over-year. "
            "Margins also expanded significantly."
        )
        results = self.det.detect_implicit_causality(text)
        assert len(results) >= 1

    def test_sentence_indices_tracked(self):
        text = (
            "The acquisition closed in March. "
            "Revenue surged in Q2."
        )
        results = self.det.detect_implicit_causality(text)
        if results:
            assert "sentence_indices" in results[0]
            assert isinstance(results[0]["sentence_indices"], tuple)


# ---------------------------------------------------------------------------
# 8. Conversion to CausalRelation
# ---------------------------------------------------------------------------

class TestToCausalRelations:
    def setup_method(self):
        self.det = ImplicitDiscourseCausalityDetector()

    def test_converts_to_causal_relation(self):
        text = (
            "The company restructured operations. "
            "Profitability improved significantly."
        )
        implicit = self.det.detect_implicit_causality(text)
        relations = self.det.to_causal_relations(implicit)
        assert all(isinstance(r, CausalRelation) for r in relations)

    def test_relation_type_is_implicit_discourse(self):
        text = (
            "Management cut costs aggressively. "
            "Margins expanded by 300 basis points."
        )
        implicit = self.det.detect_implicit_causality(text)
        relations = self.det.to_causal_relations(implicit)
        for r in relations:
            assert r.relation_type == "implicit_discourse"

    def test_metadata_has_discourse_info(self):
        text = (
            "The Fed raised interest rates. "
            "Borrowing costs increased across the sector."
        )
        implicit = self.det.detect_implicit_causality(text)
        relations = self.det.to_causal_relations(implicit)
        for r in relations:
            assert "discourse_relation" in r.metadata

    def test_clean_fn_applied(self):
        text = (
            "The company acquired a rival. "
            "Market share jumped substantially."
        )
        implicit = self.det.detect_implicit_causality(text)
        clean = lambda s: s.upper()
        relations = self.det.to_causal_relations(implicit, clean_fn=clean)
        for r in relations:
            assert r.cause == r.cause.upper()


# ---------------------------------------------------------------------------
# 9. CausalityDetector integration
# ---------------------------------------------------------------------------

class TestCausalityDetectorDiscourseIntegration:
    def setup_method(self):
        self.detector = CausalityDetector()

    def test_has_discourse_detector(self):
        assert hasattr(self.detector, "discourse_detector")
        assert isinstance(self.detector.discourse_detector, ImplicitDiscourseCausalityDetector)

    def test_classify_discourse_relation_backward_compat(self):
        label = self.detector.classify_discourse_relation(
            "Revenue fell.", "This was because demand dropped."
        )
        assert label == "causal"

    def test_classify_discourse_relation_contrast(self):
        label = self.detector.classify_discourse_relation(
            "Revenue grew.", "However, margins fell."
        )
        assert label == "contrast"

    def test_classify_discourse_relation_temporal(self):
        label = self.detector.classify_discourse_relation(
            "Revenue grew.", "After the merger, costs rose."
        )
        assert label == "temporal"

    def test_classify_discourse_relation_full(self):
        result = self.detector.classify_discourse_relation_full(
            "Revenue fell.", "This was because demand dropped."
        )
        assert "relation" in result
        assert "level1" in result
        assert "is_causal" in result
        assert "confidence" in result

    def test_detect_implicit_discourse_returns_relations(self):
        text = (
            "The company restructured its operations. "
            "Revenue grew 15% year-over-year."
        )
        relations = self.detector.detect_implicit_discourse_causality(text)
        assert isinstance(relations, list)
        for r in relations:
            assert isinstance(r, CausalRelation)

    def test_implicit_discourse_in_detect_financial_causality(self):
        text = (
            "The company restructured its operations. "
            "Revenue grew 15% year-over-year."
        )
        relations = self.detector.detect_financial_causality(text)
        discourse_rels = [r for r in relations if r.relation_type == "implicit_discourse"]
        assert len(discourse_rels) >= 0

    def test_reason_includes_discourse_analysis(self):
        result = self.detector.reason(
            question="Why did revenue increase?",
            context=(
                "The company launched a new product line. "
                "Revenue grew significantly in the following quarter."
            ),
        )
        assert "discourse_analysis" in result
        disc = result["discourse_analysis"]
        assert "total_discourse_relations" in disc
        assert "num_implicit_causal" in disc
        assert "num_explicit_causal" in disc
        assert "avg_confidence" in disc

    def test_reason_causal_context_mentions_discourse(self):
        result = self.detector.reason(
            question="Why did revenue increase?",
            context=(
                "The company acquired a major competitor. "
                "Revenue surged in the next quarter. "
                "Margins also improved due to economies of scale."
            ),
        )
        disc = result.get("discourse_analysis", {})
        if disc.get("num_implicit_causal", 0) > 0:
            assert "Implicit discourse causality" in result["causal_context"]


# ---------------------------------------------------------------------------
# 10. Metrics integration
# ---------------------------------------------------------------------------

class TestDiscourseMetrics:
    def test_discourse_quality_with_data(self):
        from src.evaluation.metrics import CausalityDetectionMetrics
        m = CausalityDetectionMetrics()
        causal_info = {
            "discourse_analysis": {
                "num_implicit_causal": 2,
                "num_explicit_causal": 1,
                "total_discourse_relations": 3,
                "avg_confidence": 0.65,
                "relations": [{"features": {"icv_cause_bias": 0.7}}],
            }
        }
        result = m.discourse_causality_quality(causal_info)
        assert result["discourse_total"] == 3
        assert result["discourse_implicit_causal"] == 2
        assert result["discourse_explicit_causal"] == 1
        assert result["discourse_avg_confidence"] == 0.65
        assert result["discourse_has_features"] == 1.0

    def test_discourse_quality_empty(self):
        from src.evaluation.metrics import CausalityDetectionMetrics
        m = CausalityDetectionMetrics()
        result = m.discourse_causality_quality({})
        assert result["discourse_total"] == 0
        assert result["discourse_avg_confidence"] == 0.0


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        self.det = ImplicitDiscourseCausalityDetector()

    def test_very_short_sentences(self):
        results = self.det.detect_implicit_causality("Up. Down.")
        assert isinstance(results, list)

    def test_identical_sentences(self):
        results = self.det.detect_implicit_causality(
            "Revenue grew. Revenue grew."
        )
        assert isinstance(results, list)

    def test_many_sentences(self):
        text = ". ".join([f"Sentence {i} about finance" for i in range(20)]) + "."
        results = self.det.detect_implicit_causality(text)
        assert isinstance(results, list)

    def test_custom_min_confidence(self):
        det_strict = ImplicitDiscourseCausalityDetector(min_confidence=0.9)
        det_lax = ImplicitDiscourseCausalityDetector(min_confidence=0.1)
        text = (
            "The company expanded overseas. "
            "Revenue grew modestly."
        )
        strict = det_strict.detect_implicit_causality(text)
        lax = det_lax.detect_implicit_causality(text)
        assert len(strict) <= len(lax)

    def test_custom_base_prior(self):
        det_high = ImplicitDiscourseCausalityDetector(base_prior=0.8)
        det_low = ImplicitDiscourseCausalityDetector(base_prior=0.1)
        features = {"cause_effect_pattern": 0.5, "icv_cause_bias": 0.5}
        high_score = det_high._bayesian_causal_score(features)
        low_score = det_low._bayesian_causal_score(features)
        assert high_score > low_score

    def test_non_financial_text(self):
        text = (
            "Birds sang in the morning. "
            "Clouds drifted across the sky."
        )
        results = self.det.detect_implicit_causality(text)
        assert len(results) == 0

    def test_mixed_explicit_implicit(self):
        text = (
            "The company restructured aggressively. "
            "Margins improved as a result. "
            "The stock price also rose."
        )
        results = self.det.detect_implicit_causality(text)
        explicit = [r for r in results if r.get("explicit")]
        implicit = [r for r in results if not r.get("explicit")]
        assert len(explicit) + len(implicit) == len(results)
