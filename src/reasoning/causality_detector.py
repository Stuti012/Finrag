"""Causality Detection Module for financial QA.

Identifies cause-effect relationships in financial narratives to answer
"why" questions and trace causal chains in financial events.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CausalRelation:
    """Represents a detected causal relationship."""

    def __init__(
        self,
        cause: str,
        effect: str,
        confidence: float,
        evidence: str = "",
        relation_type: str = "direct",
    ):
        self.cause = cause
        self.effect = effect
        self.confidence = confidence
        self.evidence = evidence
        self.relation_type = relation_type  # direct, indirect, contributing

    def __repr__(self):
        return f"CausalRelation({self.cause} -> {self.effect}, conf={self.confidence:.2f})"

    def to_dict(self) -> Dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "relation_type": self.relation_type,
        }


class CausalGraph:
    """Graph structure for causal relationships in financial narratives."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[CausalRelation] = []
        self.adjacency: Dict[str, List[CausalRelation]] = defaultdict(list)

    def add_node(self, node_id: str, metadata: Dict = None):
        self.nodes[node_id] = metadata or {}

    def add_relation(self, relation: CausalRelation):
        cause_id = relation.cause.lower().strip()
        effect_id = relation.effect.lower().strip()

        if cause_id not in self.nodes:
            self.add_node(cause_id, {"text": relation.cause})
        if effect_id not in self.nodes:
            self.add_node(effect_id, {"text": relation.effect})

        self.edges.append(relation)
        self.adjacency[cause_id].append(relation)

    def get_causes(self, effect: str) -> List[CausalRelation]:
        """Find all causes leading to a given effect."""
        effect_lower = effect.lower().strip()
        causes = []
        for edge in self.edges:
            if effect_lower in edge.effect.lower():
                causes.append(edge)
        return sorted(causes, key=lambda r: r.confidence, reverse=True)

    def get_effects(self, cause: str) -> List[CausalRelation]:
        """Find all effects of a given cause."""
        cause_lower = cause.lower().strip()
        return self.adjacency.get(cause_lower, [])

    def get_causal_chain(self, start: str, max_depth: int = 3) -> List[List[CausalRelation]]:
        """Trace causal chains from a starting event."""
        chains = []
        self._dfs_chains(start.lower().strip(), [], set(), max_depth, chains)
        return chains

    def _dfs_chains(
        self, current: str, path: List[CausalRelation],
        visited: set, max_depth: int, chains: List
    ):
        if len(path) >= max_depth:
            if path:
                chains.append(list(path))
            return

        effects = self.adjacency.get(current, [])
        if not effects and path:
            chains.append(list(path))
            return

        for relation in effects:
            next_node = relation.effect.lower().strip()
            if next_node not in visited:
                visited.add(next_node)
                path.append(relation)
                self._dfs_chains(next_node, path, visited, max_depth, chains)
                path.pop()
                visited.discard(next_node)


class CausalityDetector:
    """Detects causal relationships in financial narratives.

    Uses pattern-based extraction with optional neural scoring.

    Capabilities:
    1. Causal span extraction from financial text
    2. Causal graph construction
    3. Multi-hop causal reasoning
    4. Confidence scoring for causal relations
    """

    # Explicit causal connectors in financial text
    CAUSAL_PATTERNS = [
        # "X due to Y" -> cause=Y, effect=X
        (r"(.+?)\s+(?:due to|because of|as a result of|owing to|attributed to)\s+(.+)",
         "effect_first"),
        # "X caused Y" -> cause=X, effect=Y
        (r"(.+?)\s+(?:caused|led to|resulted in|contributed to|drove|triggered)\s+(.+)",
         "cause_first"),
        # "Because X, Y" -> cause=X, effect=Y
        (r"(?:because|since|as)\s+(.+?),\s*(.+)",
         "cause_first"),
        # "X, which led to Y"
        (r"(.+?),\s*which\s+(?:led to|caused|resulted in|drove)\s+(.+)",
         "cause_first"),
        # "X was driven by Y"
        (r"(.+?)\s+(?:was|were)\s+(?:driven by|caused by|impacted by|affected by)\s+(.+)",
         "effect_first"),
        # "following X, Y happened"
        (r"(?:following|after)\s+(.+?),\s*(.+)",
         "cause_first"),
    ]

    # Financial causal keywords
    FINANCIAL_CAUSAL_INDICATORS = {
        "revenue_increase": [
            "higher sales", "market expansion", "price increases",
            "new product launch", "increased demand", "acquisition",
        ],
        "revenue_decrease": [
            "lower demand", "competitive pressure", "market downturn",
            "loss of customers", "product discontinuation", "divestiture",
        ],
        "cost_increase": [
            "supply chain disruption", "inflation", "regulatory compliance",
            "expansion costs", "higher wages", "raw material prices",
        ],
        "cost_decrease": [
            "cost optimization", "efficiency improvements", "automation",
            "restructuring", "economies of scale", "renegotiated contracts",
        ],
        "profit_increase": [
            "revenue growth", "cost reduction", "margin improvement",
            "operational efficiency", "favorable exchange rates",
        ],
        "profit_decrease": [
            "revenue decline", "cost overruns", "impairment charges",
            "write-downs", "unfavorable exchange rates", "litigation costs",
        ],
    }

    def __init__(self, confidence_threshold: float = 0.5, max_causal_hops: int = 2):
        self.confidence_threshold = confidence_threshold
        self.max_causal_hops = max_causal_hops

    def extract_causal_spans(self, text: str) -> List[CausalRelation]:
        """Extract causal relationships from a text passage using patterns."""
        relations = []

        for sentence in self._split_sentences(text):
            sentence = sentence.strip()
            if not sentence:
                continue

            for pattern, direction in self.CAUSAL_PATTERNS:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    group1 = match.group(1).strip()
                    group2 = match.group(2).strip()

                    if direction == "cause_first":
                        cause, effect = group1, group2
                    else:
                        cause, effect = group2, group1

                    # Clean up spans
                    cause = self._clean_span(cause)
                    effect = self._clean_span(effect)

                    if cause and effect and len(cause) > 3 and len(effect) > 3:
                        confidence = self._compute_confidence(cause, effect, sentence)
                        relations.append(CausalRelation(
                            cause=cause,
                            effect=effect,
                            confidence=confidence,
                            evidence=sentence,
                            relation_type="direct",
                        ))
                    break  # Take first matching pattern per sentence

        return relations

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]

    def _clean_span(self, span: str) -> str:
        """Clean a causal span."""
        span = span.strip().rstrip(".,;:")
        # Remove leading articles and conjunctions
        span = re.sub(r"^(the|a|an|and|or|but|that|this|which)\s+", "", span, flags=re.IGNORECASE)
        return span.strip()

    def _compute_confidence(self, cause: str, effect: str, evidence: str) -> float:
        """Compute confidence score for a causal relation."""
        score = 0.5  # Base score

        # Boost for financial-specific causal terms
        financial_terms = [
            "revenue", "income", "profit", "loss", "cost", "expense",
            "margin", "growth", "decline", "increase", "decrease",
            "operating", "net", "gross", "sales", "earnings",
        ]
        cause_lower = cause.lower()
        effect_lower = effect.lower()

        cause_financial = sum(1 for t in financial_terms if t in cause_lower)
        effect_financial = sum(1 for t in financial_terms if t in effect_lower)

        score += min(0.2, (cause_financial + effect_financial) * 0.05)

        # Boost for explicit causal connectors
        strong_connectors = ["due to", "because", "caused by", "resulted in", "led to"]
        if any(c in evidence.lower() for c in strong_connectors):
            score += 0.15

        # Penalize very short or very long spans
        for span in [cause, effect]:
            words = span.split()
            if len(words) < 2:
                score -= 0.1
            elif len(words) > 20:
                score -= 0.05

        return max(0.0, min(1.0, score))

    def detect_financial_causality(
        self, text: str, question: str = ""
    ) -> List[CausalRelation]:
        """Detect financial-specific causal relations.

        Uses domain knowledge about common financial cause-effect patterns.
        """
        relations = self.extract_causal_spans(text)

        # Also check for implicit financial causality patterns
        text_lower = text.lower()
        q_lower = question.lower()

        for effect_type, causes in self.FINANCIAL_CAUSAL_INDICATORS.items():
            effect_keyword = effect_type.replace("_", " ")
            if effect_keyword in text_lower or effect_keyword in q_lower:
                for cause_phrase in causes:
                    if cause_phrase in text_lower:
                        # Check if this is already captured
                        already_found = any(
                            cause_phrase in r.cause.lower() for r in relations
                        )
                        if not already_found:
                            relations.append(CausalRelation(
                                cause=cause_phrase,
                                effect=effect_keyword,
                                confidence=0.6,
                                evidence=text[:200],
                                relation_type="implicit",
                            ))

        return [r for r in relations if r.confidence >= self.confidence_threshold]

    def build_causal_graph(
        self, texts: List[str], question: str = ""
    ) -> CausalGraph:
        """Build a causal graph from multiple text passages."""
        graph = CausalGraph()

        for text in texts:
            relations = self.detect_financial_causality(text, question)
            for rel in relations:
                graph.add_relation(rel)

        return graph

    def detect_is_causal_question(self, question: str) -> bool:
        """Determine if a question requires causal reasoning."""
        q_lower = question.lower()
        causal_indicators = [
            "why", "what caused", "what led to", "reason for",
            "due to", "because", "explain", "factor", "driver",
            "impact of", "effect of", "consequence", "result of",
            "how did .* affect", "how did .* impact",
        ]
        return any(re.search(ind, q_lower) for ind in causal_indicators)

    def reason(
        self,
        question: str,
        context: str,
        table: List[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform causal reasoning for a financial question.

        Returns:
            Dict with:
            - is_causal: whether question requires causal reasoning
            - causal_relations: extracted cause-effect pairs
            - causal_graph_info: graph structure summary
            - causal_context: enriched context with causal information
            - causal_chains: multi-hop causal chains if found
        """
        result = {
            "question": question,
            "is_causal": self.detect_is_causal_question(question),
            "causal_relations": [],
            "causal_graph_info": {},
            "causal_context": "",
            "causal_chains": [],
        }

        # Extract causal relations from context
        relations = self.detect_financial_causality(context, question)
        result["causal_relations"] = [r.to_dict() for r in relations]

        # Build causal graph
        graph = self.build_causal_graph([context], question)
        result["causal_graph_info"] = {
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
            "nodes": list(graph.nodes.keys())[:10],
        }

        # Find causal chains
        for rel in relations:
            chains = graph.get_causal_chain(rel.cause, self.max_causal_hops)
            for chain in chains:
                result["causal_chains"].append([r.to_dict() for r in chain])

        # Build enriched causal context
        causal_parts = []
        if relations:
            causal_parts.append("Detected causal relationships:")
            for i, rel in enumerate(relations[:5]):
                causal_parts.append(
                    f"  {i+1}. {rel.cause} -> {rel.effect} "
                    f"(confidence: {rel.confidence:.2f}, type: {rel.relation_type})"
                )

        result["causal_context"] = "\n".join(causal_parts)

        return result
