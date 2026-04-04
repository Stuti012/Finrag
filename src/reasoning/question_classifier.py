"""Question Classification Module for routing to appropriate reasoning modules."""

import re
from typing import Dict, List, Tuple


class QuestionClassifier:
    """Classifies financial questions to determine which reasoning modules to invoke.

    Categories:
    - numerical: requires mathematical computation
    - temporal: involves time-based reasoning
    - causal: involves cause-effect reasoning
    - factual: straightforward fact lookup from table/text
    """

    NUMERICAL_PATTERNS = [
        r"(?:what|how much|how many)\s+(?:is|was|were|are)\s+the\s+(?:total|sum|difference|ratio|percentage|average|net|gross)",
        r"(?:percentage|percent)\s+(?:change|growth|increase|decrease|decline)",
        r"(?:calculate|compute|determine)\s+the",
        r"how much (?:did|does|was|is|were|are)",
        r"what (?:fraction|proportion|share|percent)",
        r"(?:increase|decrease|change|grew|declined|rose|fell)\s+by",
        r"(?:margin|rate|ratio|yield|return|roe|roa|eps)\s",
    ]

    TEMPORAL_PATTERNS = [
        r"\b(?:20\d{2}|19\d{2})\b",
        r"(?:year|quarter|month|fiscal|annual|quarterly|monthly)",
        r"(?:previous|prior|last|next|following|subsequent|current)\s+(?:year|quarter|period|fiscal)",
        r"(?:q[1-4]|first quarter|second quarter|third quarter|fourth quarter)",
        r"(?:from|between|during|since|until)\s+\d{4}",
        r"(?:year-over-year|yoy|quarter-over-quarter|qoq)",
        r"(?:trend|over time|historically|over the past)",
        r"(?:highest|lowest|maximum|minimum|peak|trough)\s+(?:in|during|over|across)",
    ]

    CAUSAL_PATTERNS = [
        r"^why\s+",
        r"what\s+(?:caused|led to|drove|contributed to|resulted in)",
        r"(?:reason|factor|driver)\s+(?:for|behind|of)",
        r"(?:due to|because of|as a result of|attributed to)",
        r"(?:explain|account for)\s+(?:the|why|how)",
        r"(?:impact|effect|consequence)\s+(?:of|on)",
        r"how\s+did\s+.+\s+(?:affect|impact|influence)",
    ]

    FACTUAL_PATTERNS = [
        r"^(?:what|which)\s+(?:is|was|were|are)\s+(?:the|a)\s+\w+\s*\?",
        r"^(?:list|name|identify)\s+",
        r"(?:what|which)\s+(?:company|item|category|segment)",
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._numerical_re = [re.compile(p, re.IGNORECASE) for p in self.NUMERICAL_PATTERNS]
        self._temporal_re = [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS]
        self._causal_re = [re.compile(p, re.IGNORECASE) for p in self.CAUSAL_PATTERNS]
        self._factual_re = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]

    def classify(self, question: str, program: List[str] = None) -> Dict[str, float]:
        """Classify a question into reasoning types with confidence scores.

        Returns dict mapping each type to a confidence score [0, 1].
        Note: Classification is based solely on question text, not gold programs.
        """
        scores = {
            "numerical": 0.0,
            "temporal": 0.0,
            "causal": 0.0,
            "factual": 0.0,
        }

        q = question.strip()

        # Pattern matching scores
        for pattern in self._numerical_re:
            if pattern.search(q):
                scores["numerical"] += 0.25

        for pattern in self._temporal_re:
            if pattern.search(q):
                scores["temporal"] += 0.2

        for pattern in self._causal_re:
            if pattern.search(q):
                scores["causal"] += 0.3

        for pattern in self._factual_re:
            if pattern.search(q):
                scores["factual"] += 0.25

        # Heuristic: most financial questions involve numbers
        # Give a small baseline boost to numerical if question mentions numbers/amounts
        if re.search(r"\b\d[\d,.]*\b", q) or re.search(r"(?:what|how much|how many|calculate)", q, re.I):
            scores["numerical"] = max(scores["numerical"], 0.2)

        # Cap scores at 1.0
        scores = {k: min(1.0, v) for k, v in scores.items()}

        # Ensure at least one type is selected
        if max(scores.values()) == 0:
            scores["factual"] = 0.3

        return scores

    def get_primary_type(self, question: str, program: List[str] = None) -> str:
        """Get the primary reasoning type for a question."""
        scores = self.classify(question)
        return max(scores, key=scores.get)

    def get_active_modules(
        self, question: str, program: List[str] = None, threshold: float = 0.2
    ) -> List[str]:
        """Get list of reasoning modules that should be activated."""
        scores = self.classify(question)
        return [k for k, v in scores.items() if v >= threshold]
