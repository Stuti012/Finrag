"""Enhanced multi-label question classifier with optional embedding router."""

import re
from typing import Dict, List
import importlib
from importlib.util import find_spec
import numpy as np

HAS_SENTENCE_TRANSFORMERS = find_spec("sentence_transformers") is not None
if HAS_SENTENCE_TRANSFORMERS:
    SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer


class QuestionClassifier:
    """Classifies financial QA questions for module routing."""

    NUMERICAL_PATTERNS = [
        r"\b(total|sum|difference|ratio|percentage|average|net|gross|cagr|eps|roe|roa)\b",
        r"\b(calculate|compute|determine|how much|how many|by how much)\b",
        r"\b(increase|decrease|change|grew|declined|rose|fell)\s+by\b",
        r"\b(percent|percentage|fraction|proportion|share|basis points?)\b",
    ]

    TEMPORAL_PATTERNS = [
        r"\b(19\d{2}|20\d{2})\b",
        r"\b(year|quarter|month|fiscal|annual|quarterly|yoy|qoq|trend|historically)\b",
        r"\b(previous|prior|last|next|following|subsequent|current)\s+(year|quarter|period)\b",
        r"\b(before|after|since|until|between|during)\b",
    ]

    CAUSAL_PATTERNS = [
        r"^why\b",
        r"\b(caused|led to|drove|contributed to|resulted in|because|due to|reason|driver|factor)\b",
        r"\b(impact|effect|influence|consequence)\b",
        r"\b(explain|account for)\b",
    ]

    FACTUAL_PATTERNS = [
        r"^(what|which)\s+(is|was|were|are)\b",
        r"^(list|name|identify)\b",
        r"\b(company|segment|category|item)\b",
    ]
    ROUTE_ORDER = ("numerical", "temporal", "causal", "factual")

    def __init__(
        self,
        use_embedding_router: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._numerical_re = [re.compile(p, re.I) for p in self.NUMERICAL_PATTERNS]
        self._temporal_re = [re.compile(p, re.I) for p in self.TEMPORAL_PATTERNS]
        self._causal_re = [re.compile(p, re.I) for p in self.CAUSAL_PATTERNS]
        self._factual_re = [re.compile(p, re.I) for p in self.FACTUAL_PATTERNS]
        self.embedder = None
        self.route_prototypes = {}
        # Linear probe style weights over compact lexical feature vector.
        self.router_weights = {
            "numerical": np.array([1.2, 0.2, -0.4, -0.6, 0.6, -0.3], dtype=float),
            "temporal": np.array([0.1, 1.3, 0.1, -0.5, 0.2, 0.7], dtype=float),
            "causal": np.array([-0.2, 0.2, 1.4, -0.5, -0.1, 0.4], dtype=float),
            "factual": np.array([-0.4, -0.4, -0.3, 1.5, -0.2, -0.3], dtype=float),
        }

        if use_embedding_router and HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                self.route_prototypes = self._build_route_prototypes()
            except Exception:
                self.embedder = None
                self.route_prototypes = {}

    @staticmethod
    def _cap(scores: Dict[str, float]) -> Dict[str, float]:
        return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

    def _build_route_prototypes(self) -> Dict[str, np.ndarray]:
        examples = {
            "numerical": [
                "what is the percentage change in revenue from 2020 to 2021",
                "calculate the ratio of operating income to total revenue",
                "how much did expenses increase year over year",
            ],
            "temporal": [
                "what was the trend in cash flow over the last three years",
                "compare operating margin between 2019 and 2021",
                "what happened after the restructuring in q3",
            ],
            "causal": [
                "why did net income decline this year",
                "what caused gross margin compression",
                "which factors led to lower eps",
            ],
            "factual": [
                "what is the company segment with highest revenue",
                "which category had the largest contribution",
                "name the top business unit by sales",
            ],
        }
        proto = {}
        for label, texts in examples.items():
            emb = self.embedder.encode(texts)
            proto[label] = np.mean(emb, axis=0)
        return proto

    def _embedding_scores(self, question: str) -> Dict[str, float]:
        if not self.embedder or not self.route_prototypes:
            return {}
        q_emb = self.embedder.encode([question])[0]

        def cosine(a, b):
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
            return float(np.dot(a, b) / denom)

        raw = {label: cosine(q_emb, proto) for label, proto in self.route_prototypes.items()}
        # map cosine [-1,1] -> [0,1]
        return {k: max(0.0, min(1.0, (v + 1) / 2.0)) for k, v in raw.items()}

    @staticmethod
    def _softmax(vals: Dict[str, float]) -> Dict[str, float]:
        if not vals:
            return {}
        m = max(vals.values())
        exps = {k: np.exp(v - m) for k, v in vals.items()}
        denom = sum(exps.values()) + 1e-10
        return {k: float(v / denom) for k, v in exps.items()}

    def _feature_vector(self, q: str) -> np.ndarray:
        """Compact feature vector for learned routing."""
        ql = q.lower()
        year_count = len(re.findall(r"\b(19\d{2}|20\d{2})\b", ql))
        wh_why = 1.0 if ql.startswith("why") else 0.0
        numerical_count = sum(1 for p in self._numerical_re if p.search(q))
        temporal_count = sum(1 for p in self._temporal_re if p.search(q))
        causal_count = sum(1 for p in self._causal_re if p.search(q))
        factual_count = sum(1 for p in self._factual_re if p.search(q))
        return np.array(
            [
                float(numerical_count),
                float(temporal_count + min(1, year_count)),
                float(causal_count + wh_why),
                float(factual_count),
                1.0 if re.search(r"\b\d[\d,.]*\b", q) else 0.0,
                1.0 if re.search(r"\b(before|after|since|during|following)\b", ql) else 0.0,
            ],
            dtype=float,
        )

    def _moe_router_scores(self, question: str, emb_scores: Dict[str, float]) -> Dict[str, float]:
        """Mixture-of-experts style route scorer."""
        feat = self._feature_vector(question)
        logits = {route: float(np.dot(w, feat)) for route, w in self.router_weights.items()}
        lexical_router = self._softmax(logits)

        if emb_scores:
            blended = {}
            for route in self.ROUTE_ORDER:
                blended[route] = 0.6 * lexical_router.get(route, 0.0) + 0.4 * emb_scores.get(route, 0.0)
            return self._softmax(blended)
        return lexical_router

    def classify(self, question: str, program: List[str] = None) -> Dict[str, float]:
        q = (question or "").strip()
        ql = q.lower()

        scores = {"numerical": 0.0, "temporal": 0.0, "causal": 0.0, "factual": 0.0, "temporal_causal_joint": 0.0}

        for p in self._numerical_re:
            if p.search(q):
                scores["numerical"] += 0.22
        for p in self._temporal_re:
            if p.search(q):
                scores["temporal"] += 0.22
        for p in self._causal_re:
            if p.search(q):
                scores["causal"] += 0.28
        for p in self._factual_re:
            if p.search(q):
                scores["factual"] += 0.20

        # lexical priors for finqa-like questions
        if re.search(r"\b\d[\d,.]*\b", q):
            scores["numerical"] += 0.12
        if re.search(r"\b(q[1-4]|fy\d{2,4}|year-over-year|quarter-over-quarter)\b", ql):
            scores["temporal"] += 0.15
        if re.search(r"\b(why|due to|because|impact)\b", ql) and re.search(r"\b(year|quarter|q[1-4]|yoy|qoq|before|after)\b", ql):
            scores["temporal_causal_joint"] += 0.65

        # mutual reinforcement
        if scores["causal"] > 0.25 and scores["temporal"] > 0.25:
            scores["temporal_causal_joint"] += 0.25
            scores["causal"] += 0.08
            scores["temporal"] += 0.08

        # if nothing fires, factual fallback
        capped = self._cap(scores)
        emb_scores = self._embedding_scores(q)
        moe_scores = self._moe_router_scores(q, emb_scores)
        if moe_scores:
            for key in self.ROUTE_ORDER:
                capped[key] = max(capped[key], moe_scores.get(key, 0.0))
        if emb_scores:
            # Learned router contribution
            for key in ("numerical", "temporal", "causal", "factual"):
                capped[key] = max(capped[key], 0.35 * emb_scores.get(key, 0.0))

            if emb_scores.get("temporal", 0) > 0.55 and emb_scores.get("causal", 0) > 0.55:
                capped["temporal_causal_joint"] = max(
                    capped["temporal_causal_joint"],
                    min(1.0, 0.5 * (emb_scores["temporal"] + emb_scores["causal"])),
                )

        if max(capped.values()) == 0:
            capped["factual"] = 0.35

        return capped

    def get_primary_type(self, question: str, program: List[str] = None) -> str:
        scores = self.classify(question, program)
        filtered = {k: v for k, v in scores.items() if k != "temporal_causal_joint"}
        return max(filtered, key=filtered.get)

    def get_active_modules(self, question: str, program: List[str] = None, threshold: float = 0.2) -> List[str]:
        scores = self.classify(question, program)
        modules = [k for k in ("numerical", "temporal", "causal", "factual") if scores.get(k, 0.0) >= threshold]
        if scores.get("temporal_causal_joint", 0.0) >= max(0.25, threshold):
            if "temporal" not in modules:
                modules.append("temporal")
            if "causal" not in modules:
                modules.append("causal")
            modules.append("temporal_causal_joint")
        return modules
