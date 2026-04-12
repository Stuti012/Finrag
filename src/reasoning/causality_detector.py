"""Research-grade causal reasoning for financial QA.

Implements:
- Rich causal pattern extraction (20+ patterns)
- Causal knowledge graph with confidence-aware edges
- Multi-hop chain reasoning with confidence propagation
- Temporal-causal integration hooks
- Counterfactual generation framework
- Causal strength estimation from lexical, structural, and temporal signals
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CausalRelation:
    """Represents a directed cause-effect relation."""

    cause: str
    effect: str
    confidence: float
    evidence: str = ""
    relation_type: str = "direct"
    mechanism: str = ""
    lag_hint: Optional[str] = None
    polarity: str = "neutral"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "evidence": self.evidence,
            "relation_type": self.relation_type,
            "mechanism": self.mechanism,
            "lag_hint": self.lag_hint,
            "polarity": self.polarity,
            "metadata": self.metadata,
        }


class CausalGraph:
    """Confidence-aware causal graph with chain search."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[CausalRelation] = []
        self.outgoing: Dict[str, List[CausalRelation]] = defaultdict(list)
        self.incoming: Dict[str, List[CausalRelation]] = defaultdict(list)

    @staticmethod
    def _nid(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def add_relation(self, relation: CausalRelation):
        cause_id = self._nid(relation.cause)
        effect_id = self._nid(relation.effect)
        self.nodes.setdefault(cause_id, {"text": relation.cause})
        self.nodes.setdefault(effect_id, {"text": relation.effect})
        self.edges.append(relation)
        self.outgoing[cause_id].append(relation)
        self.incoming[effect_id].append(relation)

    def find_chains(
        self,
        start: str,
        max_depth: int = 3,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return confidence-aware chains from a start concept."""
        start_id = self._nid(start)
        chains: List[Dict[str, Any]] = []

        def dfs(current_id: str, path: List[CausalRelation], visited: Set[str], conf: float):
            if len(path) >= max_depth:
                if path:
                    chains.append({
                        "chain": [p.to_dict() for p in path],
                        "propagated_confidence": conf,
                        "length": len(path),
                    })
                return

            next_edges = self.outgoing.get(current_id, [])
            if not next_edges and path:
                chains.append({
                    "chain": [p.to_dict() for p in path],
                    "propagated_confidence": conf,
                    "length": len(path),
                })
                return

            for edge in next_edges:
                if edge.confidence < min_confidence:
                    continue
                nxt = self._nid(edge.effect)
                if nxt in visited:
                    continue

                # Confidence propagation with mild path-length decay.
                decay = 0.92
                propagated = conf * edge.confidence * (decay ** len(path))

                path.append(edge)
                visited.add(nxt)
                dfs(nxt, path, visited, propagated)
                visited.remove(nxt)
                path.pop()

        dfs(start_id, [], {start_id}, 1.0)
        return sorted(chains, key=lambda x: x["propagated_confidence"], reverse=True)


class CausalityDetector:
    """Research-level causality detector with temporal-causal fusion."""

    # 20+ causal templates. tuple(pattern, direction, mechanism)
    CAUSAL_PATTERNS: List[Tuple[str, str, str]] = [
        (r"(.+?)\s+(?:due to|because of|as a result of|owing to|attributed to)\s+(.+)", "effect_first", "attribution"),
        (r"(.+?)\s+(?:caused|led to|resulted in|contributed to|triggered|drove)\s+(.+)", "cause_first", "direct"),
        (r"(?:because|since|as)\s+(.+?),\s*(.+)", "cause_first", "premise"),
        (r"(.+?),\s*which\s+(?:led to|caused|resulted in|drove)\s+(.+)", "cause_first", "relative_clause"),
        (r"(.+?)\s+(?:was|were)\s+(?:driven by|caused by|impacted by|affected by|pressured by)\s+(.+)", "effect_first", "passive"),
        (r"(?:following|after)\s+(.+?),\s*(.+)", "cause_first", "temporal_trigger"),
        (r"(.+?)\s+(?:therefore|thus|hence),\s*(.+)", "cause_first", "logical"),
        (r"(.+?)\s+(?:which in turn|thereby)\s+(?:caused|led to|resulted in)\s+(.+)", "cause_first", "mediated"),
        (r"(.+?)\s+(?:amid|under)\s+(.+?),\s*(.+)", "cause_middle", "contextual"),
        (r"(.+?)\s+(?:accelerated|slowed|weakened|boosted)\s+(.+)", "cause_first", "modulation"),
        (r"(.+?)\s+(?:offset|mitigated|buffered)\s+(.+)", "cause_first", "mitigation"),
        (r"(.+?)\s+(?:stemming from|arising from|originating from)\s+(.+)", "effect_first", "origin"),
        (r"(.+?)\s+(?:in response to)\s+(.+)", "effect_first", "response"),
        (r"(.+?)\s+(?:corresponded with|coincided with)\s+(.+)", "cause_first", "association"),
        (r"(.+?)\s+(?:put pressure on|lifted)\s+(.+)", "cause_first", "pressure"),
        (r"(.+?)\s+(?:supporting|hurting)\s+(.+)", "cause_first", "impact"),
        (r"(.+?)\s+(?:enabled|allowed|helped)\s+(.+)", "cause_first", "enablement"),
        (r"(.+?)\s+(?:prevented|limited|constrained)\s+(.+)", "cause_first", "constraint"),
        (r"if\s+(.+?),\s*(?:then\s+)?(.+)", "cause_first", "conditional"),
        (r"(.+?)\s+(?:transmitted to|spilled over to)\s+(.+)", "cause_first", "spillover"),
        (r"(.+?)\s+(?:raising|reducing)\s+(.+)", "cause_first", "directional"),
        (r"(.+?)\s+(?:as|while)\s+(.+?)\s+(?:increased|decreased),\s*(.+)", "cause_middle", "co_movement"),
    ]

    TEMPORAL_LAG_PATTERNS = [
        r"with a lag of\s+(\d+\s+(?:day|days|week|weeks|month|months|quarter|quarters|year|years))",
        r"(\d+\s+(?:day|days|week|weeks|month|months|quarter|quarters|year|years))\s+later",
        r"in the following\s+(quarter|year|month)",
        r"subsequently",
        r"thereafter",
    ]

    FINANCIAL_CAUSAL_PRIORS = {
        "rate hike": ["loan demand decline", "net interest margin expansion", "valuation compression"],
        "inflation": ["input cost increase", "margin pressure", "pricing actions"],
        "supply chain disruption": ["revenue shortfall", "working capital increase", "cost inflation"],
        "fx headwind": ["revenue decline", "earnings volatility"],
        "share buyback": ["eps increase", "share count reduction"],
        "capex increase": ["depreciation increase", "future capacity growth"],
    }

    POSITIVE_WORDS = {"increase", "improve", "growth", "expand", "boost", "higher", "gain", "strong"}
    NEGATIVE_WORDS = {"decrease", "decline", "drop", "weak", "lower", "loss", "pressure", "fall"}

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        max_causal_hops: int = 3,
        chain_min_confidence: float = 0.2,
        enable_counterfactuals: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_causal_hops = max_causal_hops
        self.chain_min_confidence = chain_min_confidence
        self.enable_counterfactuals = enable_counterfactuals

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

    def _clean_span(self, span: str) -> str:
        span = re.sub(r"\s+", " ", span.strip().rstrip(".,;:"))
        span = re.sub(r"^(the|a|an|and|or|but|that|this|which|to)\s+", "", span, flags=re.I)
        return span.strip()

    def _estimate_polarity(self, text: str) -> str:
        t = text.lower()
        pos = sum(1 for w in self.POSITIVE_WORDS if w in t)
        neg = sum(1 for w in self.NEGATIVE_WORDS if w in t)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    def _extract_lag_hint(self, evidence: str) -> Optional[str]:
        for pat in self.TEMPORAL_LAG_PATTERNS:
            m = re.search(pat, evidence, flags=re.I)
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return None

    def _causal_strength(self, cause: str, effect: str, evidence: str, mechanism: str) -> float:
        """Estimate causal strength in [0,1] from blended signals."""
        score = 0.35
        ev_lower = evidence.lower()

        strong_cues = ["because", "due to", "led to", "resulted in", "caused", "therefore"]
        weak_cues = ["coincided", "associated", "amid"]

        score += min(0.25, sum(0.05 for c in strong_cues if c in ev_lower))
        score -= min(0.12, sum(0.04 for c in weak_cues if c in ev_lower))

        # Domain relevance boost.
        finance_tokens = ["revenue", "cost", "margin", "earnings", "cash", "debt", "demand", "price", "guidance"]
        relevance = sum(1 for t in finance_tokens if t in (cause + " " + effect).lower())
        score += min(0.2, relevance * 0.03)

        if mechanism in {"direct", "attribution", "logical", "conditional"}:
            score += 0.08
        if self._extract_lag_hint(evidence):
            score += 0.05

        # Penalize noisy spans.
        for span in (cause, effect):
            wc = len(span.split())
            if wc < 2:
                score -= 0.08
            elif wc > 28:
                score -= 0.06

        return float(max(0.0, min(1.0, score)))

    def extract_causal_spans(self, text: str) -> List[CausalRelation]:
        relations: List[CausalRelation] = []

        for sentence in self._split_sentences(text):
            for pattern, direction, mechanism in self.CAUSAL_PATTERNS:
                match = re.search(pattern, sentence, flags=re.I)
                if not match:
                    continue

                groups = [g.strip() for g in match.groups() if g and g.strip()]
                if len(groups) < 2:
                    continue

                if direction == "cause_first":
                    cause, effect = groups[0], groups[1]
                elif direction == "effect_first":
                    cause, effect = groups[1], groups[0]
                else:  # cause_middle style, use middle as cause and final as effect
                    cause, effect = groups[1], groups[-1]

                cause = self._clean_span(cause)
                effect = self._clean_span(effect)
                if not cause or not effect:
                    continue

                strength = self._causal_strength(cause, effect, sentence, mechanism)
                relation = CausalRelation(
                    cause=cause,
                    effect=effect,
                    confidence=strength,
                    evidence=sentence,
                    relation_type="direct" if mechanism != "association" else "associative",
                    mechanism=mechanism,
                    lag_hint=self._extract_lag_hint(sentence),
                    polarity=self._estimate_polarity(f"{cause} {effect}"),
                )
                relations.append(relation)
                break

        return relations

    def detect_financial_causality(self, text: str, question: str = "") -> List[CausalRelation]:
        relations = self.extract_causal_spans(text)
        corpus = f"{text} {question}".lower()

        for prior_cause, prior_effects in self.FINANCIAL_CAUSAL_PRIORS.items():
            if prior_cause in corpus:
                for eff in prior_effects:
                    if eff in corpus:
                        relations.append(
                            CausalRelation(
                                cause=prior_cause,
                                effect=eff,
                                confidence=0.58,
                                evidence=f"Prior matched in context: {prior_cause} -> {eff}",
                                relation_type="implicit",
                                mechanism="domain_prior",
                                polarity=self._estimate_polarity(eff),
                            )
                        )

        dedup: Dict[Tuple[str, str], CausalRelation] = {}
        for rel in relations:
            key = (rel.cause.lower(), rel.effect.lower())
            if key not in dedup or rel.confidence > dedup[key].confidence:
                dedup[key] = rel

        return [r for r in dedup.values() if r.confidence >= self.confidence_threshold]

    def build_causal_graph(self, texts: List[str], question: str = "") -> CausalGraph:
        graph = CausalGraph()
        for text in texts:
            for rel in self.detect_financial_causality(text, question):
                graph.add_relation(rel)
        return graph

    def detect_is_causal_question(self, question: str) -> bool:
        return bool(re.search(
            r"\b(why|what caused|what led to|reason|driver|factor|due to|because|impact|effect|consequence|influence)\b",
            question.lower(),
        ))

    def _counterfactuals(self, relation: CausalRelation) -> Dict[str, str]:
        if not self.enable_counterfactuals:
            return {}
        return {
            "counterfactual_question": f"If {relation.cause} had not occurred, how would {relation.effect} likely change?",
            "expected_direction": "opposite" if relation.polarity != "neutral" else "uncertain",
            "confidence": f"{relation.confidence:.2f}",
        }

    def reason(
        self,
        question: str,
        context: str,
        table: List[List[str]] = None,
        temporal_signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Temporal-causal joint reasoning entrypoint."""
        is_causal = self.detect_is_causal_question(question)
        relations = self.detect_financial_causality(context, question)
        graph = self.build_causal_graph([context], question)

        chains: List[Dict[str, Any]] = []
        for rel in relations[:5]:
            chains.extend(
                graph.find_chains(
                    rel.cause,
                    max_depth=self.max_causal_hops,
                    min_confidence=self.chain_min_confidence,
                )
            )

        temporal_overlap = 0
        temporal_entities = set()
        if temporal_signals:
            for entity in temporal_signals.get("entities", []):
                temporal_entities.add(str(entity).lower())
            for rel in relations:
                span = f"{rel.cause} {rel.effect}".lower()
                if any(t in span for t in temporal_entities):
                    temporal_overlap += 1

        relation_dicts = [r.to_dict() for r in relations]
        avg_strength = sum(r["confidence"] for r in relation_dicts) / len(relation_dicts) if relation_dicts else 0.0

        counterfactuals = [self._counterfactuals(r) for r in relations[:3]]

        causal_context_lines = []
        if relation_dicts:
            causal_context_lines.append("Detected financial causal structure:")
            for i, rel in enumerate(sorted(relation_dicts, key=lambda x: x["confidence"], reverse=True)[:5], 1):
                lag = f", lag={rel['lag_hint']}" if rel.get("lag_hint") else ""
                causal_context_lines.append(
                    f"  {i}. {rel['cause']} -> {rel['effect']} (conf={rel['confidence']:.2f}, mech={rel['mechanism']}{lag})"
                )

        return {
            "question": question,
            "is_causal": is_causal,
            "causal_relations": relation_dicts,
            "causal_graph_info": {
                "num_nodes": len(graph.nodes),
                "num_edges": len(graph.edges),
                "density": (len(graph.edges) / max(1, len(graph.nodes) * (len(graph.nodes) - 1))),
            },
            "causal_chains": chains[:10],
            "causal_strength": avg_strength,
            "temporal_causal_overlap": temporal_overlap,
            "counterfactuals": counterfactuals,
            "causal_context": "\n".join(causal_context_lines),
        }
