"""Learned question classifier with Mixture-of-Experts routing.

Replaces the pattern-based classifier with a trainable linear probe
over normalized lexical + embedding features. Supports online learning
from labeled examples via SGD weight updates.

Reference: Mixture of Experts (Shazeer et al., ICLR 2017) adapted as
module routing for multi-module QA.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib
from importlib.util import find_spec

import numpy as np

HAS_SENTENCE_TRANSFORMERS = find_spec("sentence_transformers") is not None
if HAS_SENTENCE_TRANSFORMERS:
    SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer


class QuestionClassifier:
    """Learned question classifier for financial QA module routing.

    Architecture: linear probe over a normalized feature vector combining
    lexical pattern features (12-dim) and optional embedding similarity
    features (4-dim). Total input dimension = 16.

    The gating network produces a distribution over 4 expert modules
    (numerical, temporal, causal, factual) via a single softmax, plus
    a calibrated temporal-causal joint score.
    """

    ROUTE_ORDER = ("numerical", "temporal", "causal", "factual")
    NUM_FEATURES = 16

    NUMERICAL_PATTERNS = [
        re.compile(p, re.I) for p in [
            r"\b(total|sum|difference|ratio|percentage|average|net|gross|cagr|eps|roe|roa)\b",
            r"\b(calculate|compute|determine|how much|how many|by how much)\b",
            r"\b(increase|decrease|change|grew|declined|rose|fell)\s+by\b",
            r"\b(percent|percentage|fraction|proportion|share|basis points?)\b",
        ]
    ]

    TEMPORAL_PATTERNS = [
        re.compile(p, re.I) for p in [
            r"\b(19\d{2}|20\d{2})\b",
            r"\b(year|quarter|month|fiscal|annual|quarterly|yoy|qoq|trend|historically)\b",
            r"\b(previous|prior|last|next|following|subsequent|current)\s+(year|quarter|period)\b",
            r"\b(before|after|since|until|between|during)\b",
        ]
    ]

    CAUSAL_PATTERNS = [
        re.compile(p, re.I) for p in [
            r"^why\b",
            r"\b(caused|led to|drove|contributed to|resulted in|because|due to|reason|driver|factor)\b",
            r"\b(impact|effect|influence|consequence)\b",
            r"\b(explain|account for)\b",
        ]
    ]

    FACTUAL_PATTERNS = [
        re.compile(p, re.I) for p in [
            r"^(what|which)\s+(is|was|were|are)\s+the\s+(company|segment|name|category|type|item|product|unit)\b",
            r"^(list|name|identify)\b",
            r"\b(company|segment|category|item|product|entity)\s+(name|type|called)\b",
        ]
    ]

    def __init__(
        self,
        use_embedding_router: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        weights_path: Optional[str] = None,
        shared_encoder=None,
    ):
        self.embedder = None
        self.route_prototypes: Dict[str, np.ndarray] = {}

        self._init_weights(weights_path)

        if use_embedding_router:
            if shared_encoder is not None:
                self.embedder = shared_encoder
                self.route_prototypes = self._build_route_prototypes()
            elif HAS_SENTENCE_TRANSFORMERS:
                try:
                    self.embedder = SentenceTransformer(embedding_model)
                    self.route_prototypes = self._build_route_prototypes()
                except Exception:
                    self.embedder = None

    def _init_weights(self, weights_path: Optional[str] = None):
        """Initialize or load the linear probe weights.

        Weight matrix W is (4 routes × 16 features), bias is (4,).
        Initialized with sensible domain priors that can be refined via fit().
        """
        if weights_path and Path(weights_path).exists():
            data = json.loads(Path(weights_path).read_text())
            self.W = np.array(data["W"], dtype=float)
            self.b = np.array(data["b"], dtype=float)
            self._feat_mean = np.array(data.get("feat_mean", [0.0] * self.NUM_FEATURES), dtype=float)
            self._feat_std = np.array(data.get("feat_std", [1.0] * self.NUM_FEATURES), dtype=float)
            return

        self.W = np.zeros((4, self.NUM_FEATURES), dtype=float)
        self.b = np.zeros(4, dtype=float)

        # Lexical feature indices:
        #  0: numerical pattern count (normalized)
        #  1: temporal pattern count (normalized)
        #  2: causal pattern count (normalized)
        #  3: factual pattern count (normalized)
        #  4: contains numbers (0/1)
        #  5: contains temporal markers (0/1)
        #  6: starts with "why" (0/1)
        #  7: year count
        #  8: has percentage keywords (0/1)
        #  9: has causal+temporal overlap (0/1)
        # 10: question length bucket (0-1)
        # 11: has comparison keywords (0/1)
        # Embedding similarity indices (12-15): cos_sim to each route prototype

        # numerical route: responds to numerical patterns, numbers, percentages
        self.W[0] = [1.0, 0.0, -0.3, -0.4,  0.5, 0.0, -0.4, 0.0,  0.7, -0.2, -0.1, 0.1,
                     0.4, 0.0, -0.1, -0.1]
        # temporal route: responds to temporal patterns, years, temporal markers, comparison
        self.W[1] = [0.0, 1.2, 0.0, -0.3,  0.0, 0.6, -0.2, 0.5,  0.0, 0.2, 0.0, 0.4,
                     0.0, 0.4, 0.0, -0.1]
        # causal route: responds to causal patterns, "why", causal+temporal overlap
        self.W[2] = [-0.2, 0.0, 1.2, -0.3,  -0.1, 0.0, 0.9, 0.0,  -0.1, 0.3, 0.0, 0.0,
                     -0.1, 0.0, 0.4, -0.1]
        # factual route: responds to factual patterns, inversely to others
        self.W[3] = [-0.4, -0.3, -0.3, 1.2,  -0.2, -0.2, -0.4, -0.2,  -0.3, -0.2, 0.1, -0.2,
                     -0.1, -0.1, -0.1, 0.4]

        self._feat_mean = np.zeros(self.NUM_FEATURES, dtype=float)
        self._feat_std = np.ones(self.NUM_FEATURES, dtype=float)

    def _build_route_prototypes(self) -> Dict[str, np.ndarray]:
        """Build route prototypes from canonical examples for each category."""
        examples = {
            "numerical": [
                "what is the percentage change in revenue from 2020 to 2021",
                "calculate the ratio of operating income to total revenue",
                "how much did expenses increase year over year",
                "what was the total cost of goods sold",
                "compute the eps growth rate",
            ],
            "temporal": [
                "what was the trend in cash flow over the last three years",
                "compare operating margin between 2019 and 2021",
                "what happened after the restructuring in q3",
                "how did revenue change from prior year",
                "what was the growth trajectory over five years",
            ],
            "causal": [
                "why did net income decline this year",
                "what caused gross margin compression",
                "which factors led to lower eps",
                "explain the drop in operating income",
                "what drove the increase in revenue",
            ],
            "factual": [
                "what is the company segment with highest revenue",
                "which category had the largest contribution",
                "name the top business unit by sales",
                "what is the primary revenue source",
                "list the main operating segments",
            ],
        }
        proto = {}
        for label, texts in examples.items():
            emb = self.embedder.encode(texts, show_progress_bar=False)
            proto[label] = np.mean(emb, axis=0)
        return proto

    def _extract_features(self, question: str) -> np.ndarray:
        """Extract a 16-dimensional feature vector from a question.

        Features 0-11 are lexical; features 12-15 are embedding similarities
        (zero if no embedder is available).
        """
        q = question.strip()
        ql = q.lower()

        numerical_count = sum(1 for p in self.NUMERICAL_PATTERNS if p.search(q))
        temporal_count = sum(1 for p in self.TEMPORAL_PATTERNS if p.search(q))
        causal_count = sum(1 for p in self.CAUSAL_PATTERNS if p.search(q))
        factual_count = sum(1 for p in self.FACTUAL_PATTERNS if p.search(q))

        has_numbers = 1.0 if re.search(r"\b\d[\d,.]*\b", q) else 0.0
        has_temporal_markers = 1.0 if re.search(
            r"\b(q[1-4]|fy\d{2,4}|year-over-year|quarter-over-quarter|yoy|qoq)\b", ql
        ) else 0.0
        starts_why = 1.0 if ql.startswith("why") else 0.0
        year_count = len(re.findall(r"\b(19\d{2}|20\d{2})\b", ql))
        has_pct = 1.0 if re.search(r"\b(percent|percentage|%)\b", ql) else 0.0
        has_causal_temporal = 1.0 if (
            re.search(r"\b(why|due to|because|impact|caused)\b", ql)
            and re.search(r"\b(year|quarter|q[1-4]|yoy|qoq|before|after|since)\b", ql)
        ) else 0.0
        length_bucket = min(1.0, len(q.split()) / 20.0)
        has_comparison = 1.0 if re.search(
            r"\b(compared? to|versus|vs|relative to|higher|lower|more|less)\b", ql
        ) else 0.0

        feat = np.array([
            float(numerical_count),
            float(temporal_count),
            float(causal_count),
            float(factual_count),
            has_numbers,
            has_temporal_markers,
            starts_why,
            float(min(year_count, 4)),
            has_pct,
            has_causal_temporal,
            length_bucket,
            has_comparison,
            0.0, 0.0, 0.0, 0.0,
        ], dtype=float)

        if self.embedder and self.route_prototypes:
            q_emb = self.embedder.encode([q], show_progress_bar=False)[0]
            norm_q = np.linalg.norm(q_emb) + 1e-10
            for i, route in enumerate(self.ROUTE_ORDER):
                if route in self.route_prototypes:
                    proto = self.route_prototypes[route]
                    cos = float(np.dot(q_emb, proto) / (norm_q * (np.linalg.norm(proto) + 1e-10)))
                    feat[12 + i] = (cos + 1.0) / 2.0

        return feat

    def _normalize_features(self, feat: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using running statistics."""
        return (feat - self._feat_mean) / (self._feat_std + 1e-8)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / (np.sum(exps) + 1e-10)

    def classify(self, question: str, program: List[str] = None) -> Dict[str, float]:
        """Classify a question into routing scores via learned linear probe.

        Returns a dict with scores for each route plus temporal_causal_joint.
        Scores are proper probabilities from a single softmax (no double-softmax).
        """
        q = (question or "").strip()
        if not q:
            return {r: 0.25 for r in self.ROUTE_ORDER} | {"temporal_causal_joint": 0.0}

        feat = self._extract_features(q)
        norm_feat = self._normalize_features(feat)

        logits = self.W @ norm_feat + self.b
        probs = self._softmax(logits)

        scores = {route: float(probs[i]) for i, route in enumerate(self.ROUTE_ORDER)}

        tc_joint = min(1.0, scores["temporal"] * scores["causal"] * 16.0)
        if feat[9] > 0.5:
            tc_joint = max(tc_joint, 0.6)
        scores["temporal_causal_joint"] = tc_joint

        return scores

    def get_primary_type(self, question: str, program: List[str] = None) -> str:
        scores = self.classify(question, program)
        filtered = {k: v for k, v in scores.items() if k in self.ROUTE_ORDER}
        return max(filtered, key=filtered.get)

    def get_active_modules(
        self, question: str, program: List[str] = None, threshold: float = 0.15
    ) -> List[str]:
        """Return modules whose routing score exceeds the threshold.

        Uses a relative threshold: a module is active if its score is at least
        `threshold` AND at least 50% of the top-scoring module.
        """
        scores = self.classify(question, program)
        top_score = max(scores.get(r, 0) for r in self.ROUTE_ORDER)
        relative_threshold = max(threshold, top_score * 0.5)

        modules = [
            r for r in self.ROUTE_ORDER
            if scores.get(r, 0) >= relative_threshold
        ]

        if scores.get("temporal_causal_joint", 0) >= 0.3:
            for m in ("temporal", "causal"):
                if m not in modules:
                    modules.append(m)
            modules.append("temporal_causal_joint")

        if not modules:
            modules = ["factual"]

        return modules

    def fit(
        self,
        questions: List[str],
        labels: List[str],
        learning_rate: float = 0.01,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """Train the linear probe from labeled (question, primary_type) pairs.

        Uses online SGD with cross-entropy loss. Updates both the weight
        matrix and the feature normalization statistics.

        Returns training metrics.
        """
        label_to_idx = {r: i for i, r in enumerate(self.ROUTE_ORDER)}
        features = np.array([self._extract_features(q) for q in questions])

        self._feat_mean = features.mean(axis=0)
        self._feat_std = features.std(axis=0)
        self._feat_std[self._feat_std < 1e-6] = 1.0

        norm_features = (features - self._feat_mean) / (self._feat_std + 1e-8)
        targets = np.array([label_to_idx.get(l, 3) for l in labels])

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(len(questions))
            for idx in indices:
                x = norm_features[idx]
                y = targets[idx]

                logits = self.W @ x + self.b
                probs = self._softmax(logits)

                loss = -np.log(probs[y] + 1e-10)
                epoch_loss += loss

                grad = probs.copy()
                grad[y] -= 1.0

                self.W -= learning_rate * np.outer(grad, x)
                self.b -= learning_rate * grad

            losses.append(epoch_loss / len(questions))

        preds = []
        for x in norm_features:
            logits = self.W @ x + self.b
            preds.append(np.argmax(logits))
        accuracy = np.mean(np.array(preds) == targets)

        return {
            "final_loss": float(losses[-1]) if losses else 0.0,
            "accuracy": float(accuracy),
            "num_examples": len(questions),
            "epochs": epochs,
        }

    def save_weights(self, path: str):
        """Save learned weights to JSON."""
        data = {
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "feat_mean": self._feat_mean.tolist(),
            "feat_std": self._feat_std.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_weights(self, path: str):
        """Load learned weights from JSON."""
        self._init_weights(path)
