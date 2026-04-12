"""Enhanced multi-label question classifier with temporal-causal joint detection."""

import re
from typing import Dict, List


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

    def __init__(self):
        self._numerical_re = [re.compile(p, re.I) for p in self.NUMERICAL_PATTERNS]
        self._temporal_re = [re.compile(p, re.I) for p in self.TEMPORAL_PATTERNS]
        self._causal_re = [re.compile(p, re.I) for p in self.CAUSAL_PATTERNS]
        self._factual_re = [re.compile(p, re.I) for p in self.FACTUAL_PATTERNS]

    @staticmethod
    def _cap(scores: Dict[str, float]) -> Dict[str, float]:
        return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

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
