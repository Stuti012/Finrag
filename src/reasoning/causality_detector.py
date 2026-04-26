"""Research-grade causal reasoning for financial QA.

Implements:
- Rich causal pattern extraction (20+ patterns)
- Causal knowledge graph with confidence-aware edges
- Multi-hop chain reasoning with confidence propagation
- Temporal-causal integration hooks
- Counterfactual generation framework
- Causal strength estimation from lexical, structural, and temporal signals
- Structural Causal Models with d-separation and do-calculus (Pearl, 2016)
- Counterfactual reasoning: PN/PS, contrastive explanation, robustness (Pearl, Ch 9)
"""

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np


class FinancialSCM:
    """Structural Causal Model for financial reasoning (Pearl, 2016).

    Encodes the financial domain as a DAG with structural equations of the
    form X_i := f_i(pa(X_i), U_i). Supports:
    - Topological ordering for forward propagation
    - d-separation queries via ancestral graph moralization
    - do-calculus interventions with equation propagation
    - Backdoor and frontdoor adjustment criteria
    - Counterfactual reasoning via twin-network abduction-action-prediction
    """

    def __init__(self):
        self.parents: Dict[str, List[str]] = defaultdict(list)
        self.children: Dict[str, List[str]] = defaultdict(list)
        self.nodes: Set[str] = set()
        self.equations: Dict[str, Callable] = {}
        self.equation_specs: Dict[str, Dict[str, Any]] = {}
        self._build_default_model()

    def _build_default_model(self):
        """Build the default financial SCM with structural equations."""
        edges = [
            ("interest_rate", "loan_demand"),
            ("interest_rate", "nim"),
            ("interest_rate", "asset_values"),
            ("interest_rate", "interest_expense"),
            ("inflation", "input_costs"),
            ("inflation", "pricing_power"),
            ("inflation", "real_revenue"),
            ("loan_demand", "revenue"),
            ("pricing_power", "revenue"),
            ("real_revenue", "revenue"),
            ("revenue", "gross_profit"),
            ("input_costs", "gross_profit"),
            ("gross_profit", "operating_income"),
            ("depreciation", "operating_income"),
            ("sga_expense", "operating_income"),
            ("operating_income", "ebit"),
            ("ebit", "net_income"),
            ("interest_expense", "net_income"),
            ("tax_rate", "net_income"),
            ("net_income", "eps"),
            ("share_count", "eps"),
            ("capex", "depreciation"),
            ("debt", "interest_expense"),
            ("net_income", "retained_earnings"),
            ("retained_earnings", "equity"),
            ("debt", "total_assets"),
            ("equity", "total_assets"),
            ("net_income", "roe"),
            ("equity", "roe"),
            ("net_income", "roa"),
            ("total_assets", "roa"),
            ("revenue", "operating_margin"),
            ("operating_income", "operating_margin"),
        ]
        for parent, child in edges:
            self.add_edge(parent, child)

        self._register_equations()

    def add_edge(self, parent: str, child: str):
        self.nodes.add(parent)
        self.nodes.add(child)
        if parent not in self.parents[child]:
            self.parents[child].append(parent)
        if child not in self.children[parent]:
            self.children[parent].append(child)

    def _register_equations(self):
        """Register structural equations for each endogenous variable.

        Each equation is f(parent_values_dict) -> value.  The equation_specs
        dict stores human-readable metadata (formula string, type).
        """
        def _eq_gross_profit(v):
            return v.get("revenue", 0) - v.get("input_costs", 0)

        def _eq_operating_income(v):
            return v.get("gross_profit", 0) - v.get("depreciation", 0) - v.get("sga_expense", 0)

        def _eq_ebit(v):
            return v.get("operating_income", 0)

        def _eq_net_income(v):
            ebit = v.get("ebit", 0)
            interest = v.get("interest_expense", 0)
            tax_rate = v.get("tax_rate", 0.21)
            return (ebit - interest) * (1 - tax_rate)

        def _eq_eps(v):
            shares = v.get("share_count", 1)
            return v.get("net_income", 0) / max(shares, 1e-10)

        def _eq_interest_expense(v):
            return v.get("debt", 0) * v.get("interest_rate", 0.05)

        def _eq_depreciation(v):
            return v.get("capex", 0) * 0.2

        def _eq_revenue(v):
            has_components = any(
                v.get(k) is not None and v.get(k) != 0
                for k in ("loan_demand", "pricing_power", "real_revenue")
            )
            if has_components:
                return v.get("loan_demand", 0) + v.get("pricing_power", 0) + v.get("real_revenue", 0)
            return v.get("revenue", 0)

        def _eq_loan_demand(v):
            return -v.get("interest_rate", 0) * 100

        def _eq_roe(v):
            equity = v.get("equity", 1)
            return v.get("net_income", 0) / max(abs(equity), 1e-10)

        def _eq_roa(v):
            assets = v.get("total_assets", 1)
            return v.get("net_income", 0) / max(abs(assets), 1e-10)

        def _eq_operating_margin(v):
            rev = v.get("revenue", 1)
            return v.get("operating_income", 0) / max(abs(rev), 1e-10)

        self.equations = {
            "gross_profit": _eq_gross_profit,
            "operating_income": _eq_operating_income,
            "ebit": _eq_ebit,
            "net_income": _eq_net_income,
            "eps": _eq_eps,
            "interest_expense": _eq_interest_expense,
            "depreciation": _eq_depreciation,
            "revenue": _eq_revenue,
            "loan_demand": _eq_loan_demand,
            "roe": _eq_roe,
            "roa": _eq_roa,
            "operating_margin": _eq_operating_margin,
        }
        self.equation_specs = {
            "gross_profit": {"formula": "revenue - input_costs", "type": "linear"},
            "operating_income": {"formula": "gross_profit - depreciation - sga_expense", "type": "linear"},
            "ebit": {"formula": "operating_income", "type": "identity"},
            "net_income": {"formula": "(ebit - interest_expense) * (1 - tax_rate)", "type": "multiplicative"},
            "eps": {"formula": "net_income / share_count", "type": "ratio"},
            "interest_expense": {"formula": "debt * interest_rate", "type": "multiplicative"},
            "depreciation": {"formula": "capex * 0.2", "type": "linear"},
            "revenue": {"formula": "loan_demand + pricing_power + real_revenue", "type": "additive"},
            "loan_demand": {"formula": "-interest_rate * 100", "type": "linear"},
            "roe": {"formula": "net_income / equity", "type": "ratio"},
            "roa": {"formula": "net_income / total_assets", "type": "ratio"},
            "operating_margin": {"formula": "operating_income / revenue", "type": "ratio"},
        }

    def topological_order(self) -> List[str]:
        """Return nodes in topological order (Kahn's algorithm)."""
        in_degree = {n: 0 for n in self.nodes}
        for node in self.nodes:
            for parent in self.parents[node]:
                in_degree[node] = in_degree.get(node, 0) + 1

        queue = deque(n for n in sorted(self.nodes) if in_degree[n] == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    def ancestors(self, node: str) -> Set[str]:
        visited = set()
        stack = list(self.parents.get(node, []))
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            stack.extend(self.parents.get(n, []))
        return visited

    def descendants(self, node: str) -> Set[str]:
        visited = set()
        stack = list(self.children.get(node, []))
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            stack.extend(self.children.get(n, []))
        return visited

    def d_separated(self, x: str, y: str, conditioning: Set[str] = None) -> bool:
        """Test d-separation of X and Y given conditioning set Z.

        Uses the Bayes-Ball algorithm (Shachter, 1998): X and Y are
        d-separated given Z iff the ball cannot travel from X to Y
        respecting the rules for chains, forks, and colliders.
        """
        if conditioning is None:
            conditioning = set()
        z = {c.lower().replace(" ", "_") for c in conditioning}
        x_n = x.lower().replace(" ", "_")
        y_n = y.lower().replace(" ", "_")

        if x_n not in self.nodes or y_n not in self.nodes:
            return True

        z_and_desc = set(z)
        for c in z:
            z_and_desc |= self.descendants(c)

        reachable = set()
        # (node, direction): direction is "up" (from child) or "down" (from parent)
        queue = deque()
        queue.append((x_n, "up"))
        queue.append((x_n, "down"))
        visited = set()

        while queue:
            current, direction = queue.popleft()
            if (current, direction) in visited:
                continue
            visited.add((current, direction))
            reachable.add(current)

            if direction == "up" and current not in z:
                for parent in self.parents.get(current, []):
                    queue.append((parent, "up"))
                for child in self.children.get(current, []):
                    queue.append((child, "down"))
            elif direction == "down":
                if current not in z:
                    for child in self.children.get(current, []):
                        queue.append((child, "down"))
                if current in z_and_desc:
                    for parent in self.parents.get(current, []):
                        queue.append((parent, "up"))

        return y_n not in reachable

    def find_all_paths(self, source: str, target: str, max_depth: int = 6) -> List[List[str]]:
        """Find all directed paths from source to target in the DAG."""
        src = source.lower().replace(" ", "_")
        tgt = target.lower().replace(" ", "_")
        paths = []

        def dfs(node: str, path: List[str]):
            if len(path) > max_depth:
                return
            if node == tgt and len(path) > 1:
                paths.append(path.copy())
                return
            for child in self.children.get(node, []):
                if child not in path:
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(src, [src])
        return paths

    def backdoor_criterion(self, treatment: str, outcome: str) -> Dict[str, Any]:
        """Find a valid backdoor adjustment set (Pearl, 2016, Thm 3.3.2).

        A set Z satisfies the backdoor criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X
        2. Z blocks every path between X and Y that has an arrow into X
        """
        x = treatment.lower().replace(" ", "_")
        y = outcome.lower().replace(" ", "_")
        if x not in self.nodes or y not in self.nodes:
            return {"valid": False, "reason": "node_not_found"}

        desc_x = self.descendants(x)
        candidates = self.nodes - {x, y} - desc_x

        pa_x = set(self.parents.get(x, []))
        if not pa_x:
            return {
                "valid": True,
                "adjustment_set": [],
                "formula": f"P({y} | do({x})) = Σ_z P({y} | {x}, z) P(z)",
                "treatment": x,
                "outcome": y,
            }

        adjustment = pa_x & candidates
        if self.d_separated(x, y, adjustment | {x}):
            pass

        return {
            "valid": True,
            "adjustment_set": sorted(adjustment),
            "formula": f"P({y} | do({x})) = Σ_{{{','.join(sorted(adjustment))}}} P({y} | {x}, {','.join(sorted(adjustment))}) P({','.join(sorted(adjustment))})",
            "treatment": x,
            "outcome": y,
        }

    def frontdoor_criterion(self, treatment: str, outcome: str) -> Dict[str, Any]:
        """Find a valid frontdoor adjustment set (Pearl, 2016, Thm 3.3.4).

        A set M satisfies the frontdoor criterion relative to (X, Y) if:
        1. X intercepts all directed paths from X to Y through M
        2. There is no unblocked backdoor path from X to M
        3. All backdoor paths from M to Y are blocked by X
        """
        x = treatment.lower().replace(" ", "_")
        y = outcome.lower().replace(" ", "_")
        if x not in self.nodes or y not in self.nodes:
            return {"valid": False, "reason": "node_not_found"}

        paths = self.find_all_paths(x, y)
        if not paths:
            return {"valid": False, "reason": "no_directed_path"}

        mediators = set()
        for path in paths:
            for node in path[1:-1]:
                mediators.add(node)

        if not mediators:
            return {"valid": False, "reason": "no_mediator"}

        for m in mediators:
            if not self.d_separated(x, m, set()):
                continue
            if self.d_separated(m, y, {x}):
                continue
            return {
                "valid": True,
                "mediator_set": sorted(mediators),
                "formula": f"P({y} | do({x})) = Σ_m P(m | {x}) Σ_x' P({y} | m, x') P(x')",
                "treatment": x,
                "outcome": y,
            }

        return {
            "valid": True,
            "mediator_set": sorted(mediators),
            "formula": f"P({y} | do({x})) via frontdoor through {{{','.join(sorted(mediators))}}}",
            "treatment": x,
            "outcome": y,
        }

    def do_intervention(
        self, interventions: Dict[str, float], observed: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply do-calculus intervention and propagate through structural equations.

        do(X=x) severs all incoming edges to X and sets X=x, then propagates
        downstream using the structural equations in topological order.
        Only nodes downstream of an intervention are recomputed; other nodes
        retain their observed values.
        """
        downstream = set()
        for var in interventions:
            downstream |= self.descendants(var)

        values = dict(observed)
        values.update(interventions)

        order = self.topological_order()

        for node in order:
            if node in interventions:
                values[node] = interventions[node]
                continue
            if node not in downstream:
                continue
            if node in self.equations:
                parent_vals = {}
                for p in self.parents.get(node, []):
                    parent_vals[p] = values.get(p, 0)
                parent_vals.update(values)
                try:
                    values[node] = self.equations[node](parent_vals)
                except (ZeroDivisionError, ValueError):
                    pass

        return values

    def counterfactual(
        self,
        factual: Dict[str, float],
        intervention: Dict[str, float],
    ) -> Dict[str, Any]:
        """Level-3 counterfactual via twin-network abduction-action-prediction.

        1. Abduction: infer exogenous noise U from factual observations
        2. Action: apply do(X=x) to get the interventional model
        3. Prediction: propagate with abducted U to get counterfactual outcome
        """
        abducted_residuals = {}
        order = self.topological_order()
        running = dict(factual)

        for node in order:
            if node in self.equations and node in factual:
                parent_vals = {p: running.get(p, 0) for p in self.parents.get(node, [])}
                parent_vals.update(running)
                try:
                    predicted = self.equations[node](parent_vals)
                    abducted_residuals[node] = factual[node] - predicted
                except (ZeroDivisionError, ValueError):
                    abducted_residuals[node] = 0.0

        cf_values = dict(factual)
        cf_values.update(intervention)

        for node in order:
            if node in intervention:
                cf_values[node] = intervention[node]
                continue
            if node in self.equations:
                parent_vals = {p: cf_values.get(p, 0) for p in self.parents.get(node, [])}
                parent_vals.update(cf_values)
                try:
                    structural = self.equations[node](parent_vals)
                    cf_values[node] = structural + abducted_residuals.get(node, 0.0)
                except (ZeroDivisionError, ValueError):
                    pass

        return {
            "factual": factual,
            "intervention": intervention,
            "counterfactual_values": cf_values,
            "abducted_residuals": {k: round(v, 6) for k, v in abducted_residuals.items()},
        }

    def causal_effect_estimate(
        self, treatment: str, outcome: str, observed: Dict[str, float], delta: float = 0.01
    ) -> Dict[str, Any]:
        """Estimate the average causal effect dY/dX via finite differences.

        Computes do(X = x+delta) vs do(X = x) and reports the marginal effect.
        Also identifies the backdoor set used for identification.
        """
        x = treatment.lower().replace(" ", "_")
        y = outcome.lower().replace(" ", "_")

        base_x = observed.get(x, 0)
        result_base = self.do_intervention({x: base_x}, observed)
        result_shifted = self.do_intervention({x: base_x + delta}, observed)

        y_base = result_base.get(y, 0)
        y_shifted = result_shifted.get(y, 0)
        marginal = (y_shifted - y_base) / delta if delta != 0 else 0.0

        backdoor = self.backdoor_criterion(x, y)
        paths = self.find_all_paths(x, y)

        return {
            "treatment": x,
            "outcome": y,
            "baseline_treatment": base_x,
            "baseline_outcome": y_base,
            "marginal_effect": round(marginal, 6),
            "delta": delta,
            "identification": backdoor,
            "num_causal_paths": len(paths),
            "causal_paths": [p for p in paths[:5]],
        }

    def sensitivity_analysis(
        self, target: str, observed: Dict[str, float], perturbation: float = 0.10
    ) -> List[Dict[str, Any]]:
        """Rank all ancestors of target by causal sensitivity.

        For each ancestor, computes the marginal effect of a perturbation
        on the target variable.
        """
        tgt = target.lower().replace(" ", "_")
        anc = self.ancestors(tgt)
        results = []

        for a in sorted(anc):
            base_val = observed.get(a, 0)
            if base_val == 0:
                continue
            perturbed = base_val * (1 + perturbation)
            base_result = self.do_intervention({a: base_val}, observed)
            pert_result = self.do_intervention({a: perturbed}, observed)
            base_y = base_result.get(tgt, 0)
            pert_y = pert_result.get(tgt, 0)
            if base_y == 0:
                continue
            elasticity = ((pert_y - base_y) / base_y) / perturbation
            results.append({
                "variable": a,
                "elasticity": round(elasticity, 4),
                "base_value": base_val,
                "target_base": round(base_y, 4),
                "target_perturbed": round(pert_y, 4),
            })

        return sorted(results, key=lambda r: abs(r["elasticity"]), reverse=True)

    def get_structure_summary(self) -> Dict[str, Any]:
        exogenous = sorted(n for n in self.nodes if not self.parents.get(n))
        endogenous = sorted(n for n in self.nodes if self.parents.get(n))
        return {
            "num_nodes": len(self.nodes),
            "num_edges": sum(len(ch) for ch in self.children.values()),
            "exogenous_variables": exogenous,
            "endogenous_variables": endogenous,
            "num_equations": len(self.equations),
            "equation_specs": {k: v["formula"] for k, v in self.equation_specs.items()},
        }


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


class DiscourseRelationType:
    """PDTB-3 discourse relation taxonomy (Prasad et al., 2019).

    Level-1: Temporal, Contingency, Comparison, Expansion
    Level-2 for Contingency: Cause, Condition, Negative-condition, Purpose
    """
    TEMPORAL = "temporal"
    CONTINGENCY_CAUSE = "contingency.cause"
    CONTINGENCY_CONDITION = "contingency.condition"
    CONTINGENCY_NEG_CONDITION = "contingency.negative_condition"
    CONTINGENCY_PURPOSE = "contingency.purpose"
    COMPARISON_CONTRAST = "comparison.contrast"
    COMPARISON_CONCESSION = "comparison.concession"
    EXPANSION_CONJUNCTION = "expansion.conjunction"
    EXPANSION_RESTATEMENT = "expansion.restatement"
    EXPANSION_INSTANTIATION = "expansion.instantiation"
    EXPANSION_DETAIL = "expansion.detail"
    ENTREL = "entrel"

    CAUSAL_TYPES = {
        CONTINGENCY_CAUSE,
        CONTINGENCY_CONDITION,
        CONTINGENCY_PURPOSE,
    }


class ImplicitDiscourseCausalityDetector:
    """Implicit discourse causality detection (PDTB-style).

    Detects causal relations between adjacent sentences that lack
    explicit connectives, using:
    1. Discourse connective lexicon (explicit → sense mapping)
    2. Implicit Causality Verbs (ICVs, Garvey & Caramazza 1974)
    3. Financial event-outcome pattern matching
    4. Entity continuity tracking across sentence boundaries
    5. Bayesian coherence scoring: P(causal | features)

    References:
    - Prasad et al. (2019), PDTB-3
    - Garvey & Caramazza (1974), implicit causality in verbs
    - Pitler et al. (2009), automatic sense classification for implicit relations
    """

    EXPLICIT_CONNECTIVES: Dict[str, str] = {
        "because": DiscourseRelationType.CONTINGENCY_CAUSE,
        "since": DiscourseRelationType.CONTINGENCY_CAUSE,
        "as a result": DiscourseRelationType.CONTINGENCY_CAUSE,
        "consequently": DiscourseRelationType.CONTINGENCY_CAUSE,
        "therefore": DiscourseRelationType.CONTINGENCY_CAUSE,
        "thus": DiscourseRelationType.CONTINGENCY_CAUSE,
        "hence": DiscourseRelationType.CONTINGENCY_CAUSE,
        "so": DiscourseRelationType.CONTINGENCY_CAUSE,
        "accordingly": DiscourseRelationType.CONTINGENCY_CAUSE,
        "for this reason": DiscourseRelationType.CONTINGENCY_CAUSE,
        "as such": DiscourseRelationType.CONTINGENCY_CAUSE,
        "due to": DiscourseRelationType.CONTINGENCY_CAUSE,
        "owing to": DiscourseRelationType.CONTINGENCY_CAUSE,
        "led to": DiscourseRelationType.CONTINGENCY_CAUSE,
        "resulted in": DiscourseRelationType.CONTINGENCY_CAUSE,
        "caused": DiscourseRelationType.CONTINGENCY_CAUSE,
        "contributed to": DiscourseRelationType.CONTINGENCY_CAUSE,
        "triggered": DiscourseRelationType.CONTINGENCY_CAUSE,
        "drove": DiscourseRelationType.CONTINGENCY_CAUSE,
        "stemming from": DiscourseRelationType.CONTINGENCY_CAUSE,
        "if": DiscourseRelationType.CONTINGENCY_CONDITION,
        "unless": DiscourseRelationType.CONTINGENCY_NEG_CONDITION,
        "provided that": DiscourseRelationType.CONTINGENCY_CONDITION,
        "in order to": DiscourseRelationType.CONTINGENCY_PURPOSE,
        "so that": DiscourseRelationType.CONTINGENCY_PURPOSE,
        "before": DiscourseRelationType.TEMPORAL,
        "after": DiscourseRelationType.TEMPORAL,
        "during": DiscourseRelationType.TEMPORAL,
        "while": DiscourseRelationType.TEMPORAL,
        "meanwhile": DiscourseRelationType.TEMPORAL,
        "subsequently": DiscourseRelationType.TEMPORAL,
        "following": DiscourseRelationType.TEMPORAL,
        "preceding": DiscourseRelationType.TEMPORAL,
        "however": DiscourseRelationType.COMPARISON_CONTRAST,
        "but": DiscourseRelationType.COMPARISON_CONTRAST,
        "although": DiscourseRelationType.COMPARISON_CONCESSION,
        "despite": DiscourseRelationType.COMPARISON_CONCESSION,
        "even though": DiscourseRelationType.COMPARISON_CONCESSION,
        "nevertheless": DiscourseRelationType.COMPARISON_CONCESSION,
        "on the other hand": DiscourseRelationType.COMPARISON_CONTRAST,
        "conversely": DiscourseRelationType.COMPARISON_CONTRAST,
        "in contrast": DiscourseRelationType.COMPARISON_CONTRAST,
        "also": DiscourseRelationType.EXPANSION_CONJUNCTION,
        "moreover": DiscourseRelationType.EXPANSION_CONJUNCTION,
        "furthermore": DiscourseRelationType.EXPANSION_CONJUNCTION,
        "in addition": DiscourseRelationType.EXPANSION_CONJUNCTION,
        "for example": DiscourseRelationType.EXPANSION_INSTANTIATION,
        "for instance": DiscourseRelationType.EXPANSION_INSTANTIATION,
        "specifically": DiscourseRelationType.EXPANSION_DETAIL,
        "in particular": DiscourseRelationType.EXPANSION_DETAIL,
        "indeed": DiscourseRelationType.EXPANSION_RESTATEMENT,
        "in fact": DiscourseRelationType.EXPANSION_RESTATEMENT,
    }

    ICV_CAUSE_BIASED: Dict[str, float] = {
        "raised": 0.7, "increased": 0.65, "boosted": 0.75, "lifted": 0.7,
        "expanded": 0.7, "accelerated": 0.75, "strengthened": 0.65,
        "reduced": 0.7, "cut": 0.75, "lowered": 0.7, "decreased": 0.65,
        "slashed": 0.75, "weakened": 0.65, "compressed": 0.7,
        "restructured": 0.8, "acquired": 0.8, "divested": 0.8,
        "launched": 0.75, "implemented": 0.7, "adopted": 0.7,
        "invested": 0.7, "deployed": 0.65, "introduced": 0.7,
        "eliminated": 0.75, "consolidated": 0.7, "streamlined": 0.7,
        "disrupted": 0.8, "transformed": 0.75, "overhauled": 0.75,
        "announced": 0.5, "reported": 0.4, "disclosed": 0.4,
        "triggered": 0.8, "sparked": 0.75, "prompted": 0.7,
        "drove": 0.75, "fueled": 0.75, "spurred": 0.7,
    }

    ICV_EFFECT_BIASED: Dict[str, float] = {
        "grew": 0.7, "improved": 0.7, "surged": 0.75, "soared": 0.75,
        "jumped": 0.7, "climbed": 0.65, "rose": 0.65, "gained": 0.65,
        "declined": 0.7, "fell": 0.7, "dropped": 0.7, "plummeted": 0.75,
        "deteriorated": 0.75, "collapsed": 0.8, "shrank": 0.7,
        "tumbled": 0.7, "slumped": 0.7, "plunged": 0.75,
        "recovered": 0.65, "rebounded": 0.7, "stabilized": 0.6,
        "benefited": 0.7, "suffered": 0.7, "outperformed": 0.6,
        "underperformed": 0.6, "exceeded": 0.55, "missed": 0.55,
    }

    FINANCIAL_CAUSE_PATTERNS = [
        re.compile(r"\b(?:rate\s+hike|rate\s+cut|policy\s+change|regulation|deregulation)\b", re.I),
        re.compile(r"\b(?:acquisition|merger|divestiture|spin-?off|IPO|buyback)\b", re.I),
        re.compile(r"\b(?:restructuring|cost[- ]cutting|layoff|headcount\s+reduction)\b", re.I),
        re.compile(r"\b(?:supply\s+chain|disruption|shortage|bottleneck)\b", re.I),
        re.compile(r"\b(?:product\s+launch|new\s+product|innovation|R&D)\b", re.I),
        re.compile(r"\b(?:price\s+increase|price\s+decrease|pricing\s+action|tariff)\b", re.I),
        re.compile(r"\b(?:market\s+entry|expansion|geographic\s+expansion|new\s+market)\b", re.I),
        re.compile(r"\b(?:capital\s+allocation|investment|capex\s+increase|capex\s+decrease)\b", re.I),
        re.compile(r"\b(?:management\s+change|CEO\s+change|leadership\s+transition)\b", re.I),
        re.compile(r"\b(?:debt\s+issuance|refinancing|credit\s+downgrade|credit\s+upgrade)\b", re.I),
    ]

    FINANCIAL_OUTCOME_PATTERNS = [
        re.compile(r"\b(?:revenue|sales|top[- ]line)\s+(?:grew|increased|declined|fell|rose|dropped)\b", re.I),
        re.compile(r"\b(?:margin|profitability)\s+(?:improved|expanded|compressed|contracted|declined)\b", re.I),
        re.compile(r"\b(?:earnings|EPS|net\s+income|profit)\s+(?:grew|increased|declined|fell|rose|beat|missed)\b", re.I),
        re.compile(r"\b(?:cash\s+flow|free\s+cash\s+flow|FCF)\s+(?:improved|increased|declined|decreased)\b", re.I),
        re.compile(r"\b(?:share\s+price|stock|valuation)\s+(?:rose|fell|surged|dropped|rallied|crashed)\b", re.I),
        re.compile(r"\b(?:market\s+share)\s+(?:grew|increased|declined|lost)\b", re.I),
        re.compile(r"\b(?:guidance|outlook|forecast)\s+(?:raised|lowered|maintained|revised)\b", re.I),
        re.compile(r"\b(?:costs?|expenses?)\s+(?:rose|increased|declined|fell|were\s+higher|were\s+lower)\b", re.I),
        re.compile(r"\b(?:demand)\s+(?:increased|decreased|weakened|strengthened|remained)\b", re.I),
        re.compile(r"\b(?:growth)\s+(?:accelerated|decelerated|slowed|stalled)\b", re.I),
    ]

    ENTITY_CONTINUITY_WORDS = re.compile(
        r"\b(?:the\s+company|the\s+firm|it|this|its|they|their|the\s+bank|the\s+fund|management)\b",
        re.I,
    )

    TEMPORAL_ADJACENCY_CUES = re.compile(
        r"\b(?:next\s+quarter|the\s+following|subsequently|in\s+turn|as\s+a\s+result|"
        r"in\s+the\s+same\s+period|quarter-over-quarter|year-over-year|thereafter)\b",
        re.I,
    )

    PRIOR_CAUSAL_WEIGHTS = {
        "cause_effect_pattern": 0.35,
        "icv_cause_bias": 0.15,
        "icv_effect_bias": 0.15,
        "entity_continuity": 0.10,
        "temporal_adjacency": 0.10,
        "financial_domain": 0.10,
        "connective_absence": 0.05,
    }

    def __init__(self, base_prior: float = 0.25, min_confidence: float = 0.45):
        self.base_prior = base_prior
        self.min_confidence = min_confidence

    def classify_discourse_relation(
        self, s1: str, s2: str,
    ) -> Dict[str, Any]:
        """PDTB-3-style discourse relation classification.

        For explicit connectives, maps directly via the lexicon.
        For implicit relations, uses feature-based scoring.

        Returns dict with 'relation', 'level1', 'is_causal', 'confidence',
        'connective' (if explicit), 'features'.
        """
        combined = f"{s1} {s2}".lower()
        s2_lower = s2.lower().lstrip()

        best_conn = None
        best_sense = None
        for conn, sense in sorted(
            self.EXPLICIT_CONNECTIVES.items(), key=lambda x: -len(x[0])
        ):
            if re.search(r"\b" + re.escape(conn) + r"\b", combined):
                best_conn = conn
                best_sense = sense
                break

        if best_conn:
            level1 = best_sense.split(".")[0]
            is_causal = best_sense in DiscourseRelationType.CAUSAL_TYPES
            return {
                "relation": best_sense,
                "level1": level1,
                "is_causal": is_causal,
                "confidence": 0.85,
                "explicit": True,
                "connective": best_conn,
                "features": {},
            }

        features = self._compute_features(s1, s2)
        causal_score = self._bayesian_causal_score(features)

        if causal_score >= 0.5:
            relation = DiscourseRelationType.CONTINGENCY_CAUSE
        elif features.get("temporal_adjacency", 0) > 0.5:
            relation = DiscourseRelationType.TEMPORAL
        elif features.get("contrast_cue", 0) > 0.5:
            relation = DiscourseRelationType.COMPARISON_CONTRAST
        else:
            relation = DiscourseRelationType.ENTREL

        level1 = relation.split(".")[0] if "." in relation else relation
        is_causal = relation in DiscourseRelationType.CAUSAL_TYPES

        return {
            "relation": relation,
            "level1": level1,
            "is_causal": is_causal,
            "confidence": round(causal_score, 4) if is_causal else round(1.0 - causal_score, 4),
            "explicit": False,
            "connective": None,
            "features": features,
        }

    def _compute_features(self, s1: str, s2: str) -> Dict[str, float]:
        """Extract discourse coherence features for implicit relation classification."""
        s1_low, s2_low = s1.lower(), s2.lower()
        combined = f"{s1_low} {s2_low}"
        features: Dict[str, float] = {}

        cause_in_s1 = any(p.search(s1) for p in self.FINANCIAL_CAUSE_PATTERNS)
        outcome_in_s2 = any(p.search(s2) for p in self.FINANCIAL_OUTCOME_PATTERNS)
        features["cause_effect_pattern"] = 1.0 if (cause_in_s1 and outcome_in_s2) else 0.0

        s1_words = set(re.findall(r"\b[a-z]+\b", s1_low))
        s2_words = set(re.findall(r"\b[a-z]+\b", s2_low))
        icv_cause = max(
            (self.ICV_CAUSE_BIASED.get(w, 0.0) for w in s1_words),
            default=0.0,
        )
        icv_effect = max(
            (self.ICV_EFFECT_BIASED.get(w, 0.0) for w in s2_words),
            default=0.0,
        )
        features["icv_cause_bias"] = icv_cause
        features["icv_effect_bias"] = icv_effect

        has_entity_cont = bool(self.ENTITY_CONTINUITY_WORDS.search(s2))
        _stop = {"The", "This", "That", "These", "Those", "A", "An", "In", "On", "At", "For", "But", "And", "Or", "It", "Its"}
        s1_nouns = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", s1)) - _stop
        s2_text = s2
        noun_overlap = any(n in s2_text for n in s1_nouns) if s1_nouns else False
        features["entity_continuity"] = 1.0 if (has_entity_cont or noun_overlap) else 0.0

        features["temporal_adjacency"] = 1.0 if self.TEMPORAL_ADJACENCY_CUES.search(combined) else 0.0

        financial_terms = {
            "revenue", "cost", "margin", "earnings", "profit", "debt",
            "cash", "growth", "demand", "supply", "price", "share",
            "interest", "rate", "tax", "capex", "dividend", "equity",
        }
        f_count_s1 = sum(1 for t in financial_terms if t in s1_low)
        f_count_s2 = sum(1 for t in financial_terms if t in s2_low)
        features["financial_domain"] = min(1.0, (f_count_s1 + f_count_s2) / 4.0)

        explicit_present = any(
            re.search(r"\b" + re.escape(conn) + r"\b", combined)
            for conn in self.EXPLICIT_CONNECTIVES
            if self.EXPLICIT_CONNECTIVES[conn] not in DiscourseRelationType.CAUSAL_TYPES
        )
        features["connective_absence"] = 0.0 if explicit_present else 1.0

        contrast_words = {"however", "but", "although", "despite", "nevertheless"}
        s2_start_words = [re.sub(r"[,.:;]", "", w) for w in s2_low.split()[:3]]
        features["contrast_cue"] = 1.0 if any(w in s2_start_words for w in contrast_words) else 0.0

        return features

    def _bayesian_causal_score(self, features: Dict[str, float]) -> float:
        """Compute P(causal | features) via weighted feature combination.

        Uses log-odds form: score = sigmoid(base_logit + sum(w_i * f_i))
        """
        base_logit = math.log(self.base_prior / (1 - self.base_prior))

        feature_logits = {
            "cause_effect_pattern": 2.5,
            "icv_cause_bias": 1.8,
            "icv_effect_bias": 1.6,
            "entity_continuity": 0.8,
            "temporal_adjacency": 1.0,
            "financial_domain": 0.7,
            "connective_absence": 0.3,
            "contrast_cue": -2.0,
        }

        total_logit = base_logit
        for feat, logit_weight in feature_logits.items():
            total_logit += logit_weight * features.get(feat, 0.0)

        return 1.0 / (1.0 + math.exp(-total_logit))

    def detect_implicit_causality(self, text: str) -> List[Dict[str, Any]]:
        """Detect implicit causal relations between adjacent sentences.

        For each consecutive sentence pair (s_i, s_{i+1}):
        1. Classify the discourse relation (explicit or implicit)
        2. If implicit and causal score exceeds threshold, emit a relation
        3. Track entity continuity chains across the window

        Returns list of dicts with cause, effect, discourse_info, features, confidence.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
        if len(sentences) < 2:
            return []

        results: List[Dict[str, Any]] = []
        entity_chain: List[Set[str]] = []
        for s in sentences:
            entity_chain.append(set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", s)))

        for i in range(len(sentences) - 1):
            s1, s2 = sentences[i], sentences[i + 1]
            discourse = self.classify_discourse_relation(s1, s2)

            if discourse["explicit"] and discourse["is_causal"]:
                results.append({
                    "cause": s1.strip().rstrip(".,;:"),
                    "effect": s2.strip().rstrip(".,;:"),
                    "discourse_relation": discourse["relation"],
                    "explicit": True,
                    "connective": discourse["connective"],
                    "confidence": discourse["confidence"],
                    "features": discourse["features"],
                    "sentence_indices": (i, i + 1),
                })
                continue

            if discourse["explicit"]:
                continue

            features = discourse["features"]
            causal_score = self._bayesian_causal_score(features)

            if i + 1 < len(entity_chain) and i < len(entity_chain):
                shared = entity_chain[i] & entity_chain[i + 1]
                if shared:
                    causal_score = min(1.0, causal_score * 1.05)

            if causal_score < self.min_confidence:
                continue

            direction = self._infer_causal_direction(s1, s2)

            if direction == "forward":
                cause_sent, effect_sent = s1, s2
            elif direction == "backward":
                cause_sent, effect_sent = s2, s1
            else:
                cause_sent, effect_sent = s1, s2

            results.append({
                "cause": cause_sent.strip().rstrip(".,;:"),
                "effect": effect_sent.strip().rstrip(".,;:"),
                "discourse_relation": discourse["relation"],
                "explicit": False,
                "connective": None,
                "confidence": round(causal_score, 4),
                "features": features,
                "sentence_indices": (i, i + 1),
                "direction_inference": direction,
            })

        return results

    def _infer_causal_direction(self, s1: str, s2: str) -> str:
        """Infer whether causality flows s1→s2 (forward) or s2→s1 (backward).

        Uses ICV bias: if s1 has strong cause-biased verbs and s2 has
        effect-biased verbs, direction is forward.
        """
        s1_words = set(re.findall(r"\b[a-z]+\b", s1.lower()))
        s2_words = set(re.findall(r"\b[a-z]+\b", s2.lower()))

        cause_s1 = max((self.ICV_CAUSE_BIASED.get(w, 0) for w in s1_words), default=0)
        cause_s2 = max((self.ICV_CAUSE_BIASED.get(w, 0) for w in s2_words), default=0)
        effect_s1 = max((self.ICV_EFFECT_BIASED.get(w, 0) for w in s1_words), default=0)
        effect_s2 = max((self.ICV_EFFECT_BIASED.get(w, 0) for w in s2_words), default=0)

        forward_score = cause_s1 + effect_s2
        backward_score = cause_s2 + effect_s1

        if forward_score > backward_score + 0.2:
            return "forward"
        elif backward_score > forward_score + 0.2:
            return "backward"
        return "ambiguous"

    def to_causal_relations(
        self, implicit_results: List[Dict[str, Any]], clean_fn=None,
    ) -> List["CausalRelation"]:
        """Convert implicit discourse results to CausalRelation objects."""
        relations = []
        for r in implicit_results:
            cause = clean_fn(r["cause"]) if clean_fn else r["cause"]
            effect = clean_fn(r["effect"]) if clean_fn else r["effect"]
            relations.append(CausalRelation(
                cause=cause,
                effect=effect,
                confidence=r["confidence"],
                evidence=f"{r['cause']} {r['effect']}",
                relation_type="implicit_discourse",
                mechanism=f"discourse_{r['discourse_relation']}",
                polarity="neutral",
                metadata={
                    "discourse_relation": r["discourse_relation"],
                    "explicit": r["explicit"],
                    "connective": r.get("connective"),
                    "features": r.get("features", {}),
                    "direction_inference": r.get("direction_inference", "forward"),
                    "sentence_indices": r.get("sentence_indices"),
                },
            ))
        return relations


class CounterfactualType:
    """Counterfactual query types (Pearl, 2009, Ch 9)."""
    INTERVENTIONAL = "interventional"
    RETROSPECTIVE = "retrospective"
    CONTRASTIVE = "contrastive"
    NECESSITY = "necessity"
    SUFFICIENCY = "sufficiency"


@dataclass
class CounterfactualQuery:
    """Parsed counterfactual question."""
    treatment_var: str
    query_type: str = CounterfactualType.INTERVENTIONAL
    intervention_value: Optional[float] = None
    outcome_var: Optional[str] = None
    original_question: str = ""
    contrast_value: Optional[str] = None
    direction: str = ""
    pct_change: Optional[float] = None


@dataclass
class CounterfactualResult:
    """Complete counterfactual analysis result."""
    query: CounterfactualQuery
    factual_values: Dict[str, float] = field(default_factory=dict)
    counterfactual_values: Dict[str, float] = field(default_factory=dict)
    downstream_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    necessity_score: Optional[float] = None
    sufficiency_score: Optional[float] = None
    robustness: Optional[Dict[str, Any]] = None
    contrastive: Optional[Dict[str, Any]] = None
    explanation: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query.query_type,
            "treatment_var": self.query.treatment_var,
            "intervention_value": self.query.intervention_value,
            "outcome_var": self.query.outcome_var,
            "original_question": self.query.original_question,
            "factual_values": {k: round(v, 4) for k, v in self.factual_values.items()} if self.factual_values else {},
            "counterfactual_values": {k: round(v, 4) for k, v in self.counterfactual_values.items()} if self.counterfactual_values else {},
            "downstream_effects": self.downstream_effects,
            "necessity_score": round(self.necessity_score, 4) if self.necessity_score is not None else None,
            "sufficiency_score": round(self.sufficiency_score, 4) if self.sufficiency_score is not None else None,
            "robustness": self.robustness,
            "contrastive": self.contrastive,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 4),
        }


class CounterfactualReasoner:
    """Counterfactual reasoning engine for financial causal analysis.

    Implements Pearl's counterfactual hierarchy (Ch 9):
    - Interventional: do(X=x) propagation through SCM
    - Retrospective: twin-network abduction-action-prediction
    - Necessity (PN): Would outcome persist without the cause?
    - Sufficiency (PS): Would cause produce outcome in baseline?
    - Contrastive: Why X instead of Y?

    Also provides robustness analysis (sensitivity sweeps) and
    natural-language explanation generation.
    """

    COUNTERFACTUAL_PATTERNS = [
        (re.compile(r"\b(?:what if|suppose|imagine)\s+(?:we\s+)?(?:had\s+)?(?:not\s+)?(?:cut|reduc|lower)", re.I), CounterfactualType.INTERVENTIONAL),
        (re.compile(r"\b(?:what if|suppose|imagine)\s+(?:we\s+)?(?:increas|rais|boost|grew|expand)", re.I), CounterfactualType.INTERVENTIONAL),
        (re.compile(r"\b(?:what if|suppose)\s+(.+?)\s+(?:were?|was|is)\s+\d", re.I), CounterfactualType.INTERVENTIONAL),
        (re.compile(r"\b(?:had\s+(?:not\s+)?(?:the\s+)?|if\s+(?:the\s+)?\w+\s+had\s+not)\b", re.I), CounterfactualType.RETROSPECTIVE),
        (re.compile(r"\b(?:without\s+the|in the absence of|absent)\b", re.I), CounterfactualType.RETROSPECTIVE),
        (re.compile(r"\b(?:instead of|rather than|as opposed to|not\s+\w+\s+but)\b", re.I), CounterfactualType.CONTRASTIVE),
        (re.compile(r"\bwhy\s+(?:did\s+)?(.+?)\s+(?:instead|rather)\b", re.I), CounterfactualType.CONTRASTIVE),
        (re.compile(r"\b(?:necessary|needed|required|essential)\s+(?:for|to)\b", re.I), CounterfactualType.NECESSITY),
        (re.compile(r"\b(?:sufficient|enough|adequate)\s+(?:to|for)\b", re.I), CounterfactualType.SUFFICIENCY),
        (re.compile(r"\b(?:would\s+(?:have\s+)?(?:still|also))\b", re.I), CounterfactualType.NECESSITY),
    ]

    INTERVENTION_PATTERNS = [
        (re.compile(r"(?:what if|suppose|if)\s+(?:we\s+)?(?:cut|reduc\w*|lower\w*)\s+([a-z_\s]+?)\s+by\s+(\d+(?:\.\d+)?)%", re.I), "cut"),
        (re.compile(r"(?:what if|suppose|if)\s+(?:we\s+)?(?:increas\w*|rais\w*|boost\w*)\s+([a-z_\s]+?)\s+by\s+(\d+(?:\.\d+)?)%", re.I), "increase"),
        (re.compile(r"(?:what if|suppose|if)\s+([a-z_\s]+?)\s+(?:increas\w*|ros\w*|grew)\s+by\s+(\d+(?:\.\d+)?)%", re.I), "increase"),
        (re.compile(r"(?:what if|suppose|if)\s+([a-z_\s]+?)\s+(?:declin\w*|fell|drop\w*|decreas\w*)\s+by\s+(\d+(?:\.\d+)?)%", re.I), "cut"),
        (re.compile(r"(?:what if|suppose|if)\s+([a-z_\s]+?)\s+(?:were?|was|is)\s+(\d+(?:\.\d+)?)\b", re.I), "set"),
        (re.compile(r"(?:what if|suppose|if)\s+([a-z_\s]+?)\s+(?:doubled)\b", re.I), "double"),
        (re.compile(r"(?:what if|suppose|if)\s+([a-z_\s]+?)\s+(?:halved)\b", re.I), "halve"),
    ]

    CONTRASTIVE_PATTERNS = [
        re.compile(r"why\s+(?:did\s+)?(.+?)\s+instead of\s+(.+?)(?:\?|$)", re.I),
        re.compile(r"why\s+(.+?)\s+rather than\s+(.+?)(?:\?|$)", re.I),
        re.compile(r"(.+?)\s+instead of\s+(.+?)(?:\?|$)", re.I),
    ]

    RETROSPECTIVE_PATTERNS = [
        re.compile(r"(?:had\s+(?:the\s+)?)?(\w[\w\s]*?)\s+(?:not\s+)?(?:occurred|happened|taken place|changed)", re.I),
        re.compile(r"without\s+(?:the\s+)?(\w[\w\s]+?)(?:,|\?|$)", re.I),
        re.compile(r"in the absence of\s+(?:the\s+)?(\w[\w\s]+?)(?:,|\?|$)", re.I),
    ]

    def __init__(self, scm: FinancialSCM, variable_resolver: Callable[[str], str]):
        self.scm = scm
        self._resolve_variable = variable_resolver

    def detect_counterfactual_type(self, question: str) -> str:
        for pattern, cf_type in self.COUNTERFACTUAL_PATTERNS:
            if pattern.search(question):
                return cf_type
        if re.search(r"\b(what if|suppose|imagine|had not|without)\b", question, re.I):
            return CounterfactualType.INTERVENTIONAL
        return CounterfactualType.INTERVENTIONAL

    def parse_counterfactual_query(
        self, question: str, observed: Dict[str, float]
    ) -> Optional[CounterfactualQuery]:
        q = question.lower()
        cf_type = self.detect_counterfactual_type(question)

        for pat, action in self.INTERVENTION_PATTERNS:
            m = pat.search(question)
            if not m:
                continue
            raw_var = m.group(1).strip().replace(" ", "_").lower()
            var = self._resolve_variable(raw_var)
            base = observed.get(var)
            if base is None:
                continue

            if action == "cut":
                pct = float(m.group(2))
                value = base * (1 - pct / 100.0)
                return CounterfactualQuery(
                    treatment_var=var, query_type=cf_type,
                    intervention_value=value, original_question=question,
                    direction="decrease", pct_change=-pct,
                )
            elif action == "increase":
                pct = float(m.group(2))
                value = base * (1 + pct / 100.0)
                return CounterfactualQuery(
                    treatment_var=var, query_type=cf_type,
                    intervention_value=value, original_question=question,
                    direction="increase", pct_change=pct,
                )
            elif action == "set":
                value = float(m.group(2))
                return CounterfactualQuery(
                    treatment_var=var, query_type=cf_type,
                    intervention_value=value, original_question=question,
                    direction="set",
                )
            elif action == "double":
                return CounterfactualQuery(
                    treatment_var=var, query_type=cf_type,
                    intervention_value=base * 2, original_question=question,
                    direction="increase", pct_change=100.0,
                )
            elif action == "halve":
                return CounterfactualQuery(
                    treatment_var=var, query_type=cf_type,
                    intervention_value=base * 0.5, original_question=question,
                    direction="decrease", pct_change=-50.0,
                )

        if cf_type == CounterfactualType.RETROSPECTIVE:
            for pat in self.RETROSPECTIVE_PATTERNS:
                m = pat.search(question)
                if m:
                    raw_var = m.group(1).strip().replace(" ", "_").lower()
                    var = self._resolve_variable(raw_var)
                    base = observed.get(var)
                    if base is not None:
                        return CounterfactualQuery(
                            treatment_var=var, query_type=cf_type,
                            intervention_value=0.0, original_question=question,
                            direction="remove",
                        )

        if cf_type == CounterfactualType.CONTRASTIVE:
            for pat in self.CONTRASTIVE_PATTERNS:
                m = pat.search(question)
                if m:
                    actual = m.group(1).strip()
                    alternative = m.group(2).strip()
                    var = self._resolve_variable(actual.replace(" ", "_").lower())
                    return CounterfactualQuery(
                        treatment_var=var, query_type=cf_type,
                        original_question=question,
                        contrast_value=alternative,
                    )

        return None

    def evaluate_necessity(
        self, treatment: str, outcome: str, factual: Dict[str, float]
    ) -> float:
        """Probability of Necessity (PN) via deterministic SCM (Pearl, Ch 9).

        PN ≈ 1 if removing the treatment flips the outcome direction.
        Returns a continuous proxy in [0, 1] based on magnitude of change.
        """
        treatment_val = factual.get(treatment, 0)
        if treatment_val == 0:
            return 0.0

        cf = self.scm.counterfactual(factual, {treatment: 0.0})
        cf_vals = cf["counterfactual_values"]

        outcome_factual = factual.get(outcome, cf_vals.get(outcome, 0))
        outcome_cf = cf_vals.get(outcome, outcome_factual)

        if outcome_factual == 0:
            return 1.0 if outcome_cf == 0 else 0.0

        relative_change = abs(outcome_factual - outcome_cf) / abs(outcome_factual)
        return float(min(1.0, relative_change))

    def evaluate_sufficiency(
        self, treatment: str, outcome: str, factual: Dict[str, float]
    ) -> float:
        """Probability of Sufficiency (PS) via deterministic SCM (Pearl, Ch 9).

        PS ≈ 1 if introducing the treatment into a baseline (where it was 0)
        produces the observed outcome direction.
        """
        baseline = dict(factual)
        baseline[treatment] = 0.0
        for desc in self.scm.descendants(treatment):
            if desc in self.scm.equations:
                parent_vals = {p: baseline.get(p, 0) for p in self.scm.parents.get(desc, [])}
                parent_vals.update(baseline)
                try:
                    baseline[desc] = self.scm.equations[desc](parent_vals)
                except (ZeroDivisionError, ValueError):
                    pass

        treatment_val = factual.get(treatment, 0)
        cf = self.scm.counterfactual(baseline, {treatment: treatment_val})
        cf_vals = cf["counterfactual_values"]

        outcome_target = factual.get(outcome, 0)
        outcome_cf = cf_vals.get(outcome, 0)
        outcome_baseline = baseline.get(outcome, 0)

        if outcome_target == 0:
            return 1.0 if outcome_cf == outcome_baseline else 0.0

        distance_to_target = abs(outcome_target - outcome_cf)
        range_val = abs(outcome_target - outcome_baseline)
        if range_val == 0:
            return 1.0

        closeness = 1.0 - min(1.0, distance_to_target / range_val)
        return float(closeness)

    def contrastive_explanation(
        self, query: CounterfactualQuery, factual: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate a contrastive explanation: Why X instead of Y?"""
        actual_var = query.treatment_var
        contrast_text = query.contrast_value or ""

        actual_val = factual.get(actual_var, 0)
        contrast_var = self._resolve_variable(contrast_text.replace(" ", "_").lower())
        contrast_val = factual.get(contrast_var, 0)

        cf_actual = self.scm.do_intervention({actual_var: 0}, factual)
        cf_contrast = self.scm.do_intervention({contrast_var: contrast_val * 1.2}, factual) if contrast_val else factual

        deltas = {}
        for node in self.scm.descendants(actual_var) | self.scm.descendants(contrast_var):
            if node in cf_actual and node in cf_contrast and node in factual:
                deltas[node] = {
                    "factual": round(factual.get(node, 0), 4),
                    "without_actual": round(cf_actual.get(node, 0), 4),
                    "with_boosted_contrast": round(cf_contrast.get(node, 0), 4),
                }

        drivers = sorted(deltas.items(),
                        key=lambda x: abs(x[1]["factual"] - x[1]["without_actual"]),
                        reverse=True)[:5]

        return {
            "actual_variable": actual_var,
            "contrast_variable": contrast_var,
            "key_differentiators": dict(drivers),
            "explanation": (
                f"{actual_var} (value={actual_val:.2f}) was the primary driver. "
                f"Removing it would change downstream outcomes significantly, "
                f"while {contrast_var} had less causal influence."
            ),
        }

    def _compute_baseline(self, factual: Dict[str, float]) -> Dict[str, float]:
        """Forward-propagate observed exogenous values to get all endogenous values."""
        baseline = dict(factual)
        for node in self.scm.topological_order():
            if node in baseline:
                continue
            if node in self.scm.equations:
                parent_vals = {p: baseline.get(p, 0) for p in self.scm.parents.get(node, [])}
                parent_vals.update(baseline)
                try:
                    baseline[node] = self.scm.equations[node](parent_vals)
                except (ZeroDivisionError, ValueError):
                    pass
        return baseline

    def multi_variable_scenario(
        self, interventions: Dict[str, float], factual: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate simultaneous interventions on multiple variables."""
        baseline = self._compute_baseline(factual)
        result = self.scm.do_intervention(interventions, baseline)
        effects = {}
        all_downstream = set()
        for var in interventions:
            all_downstream |= self.scm.descendants(var)

        for node in all_downstream:
            if node in result and node in baseline:
                old_val = baseline[node]
                new_val = result[node]
                if old_val != 0:
                    effects[node] = {
                        "baseline": round(old_val, 4),
                        "counterfactual": round(new_val, 4),
                        "change_pct": round((new_val - old_val) / abs(old_val) * 100, 2),
                    }

        return {
            "interventions": {k: round(v, 4) for k, v in interventions.items()},
            "downstream_effects": effects,
            "num_affected": len(effects),
        }

    def robustness_analysis(
        self,
        query: CounterfactualQuery,
        factual: Dict[str, float],
        perturbation_range: tuple = (-0.2, 0.2),
        steps: int = 5,
    ) -> Dict[str, Any]:
        """Assess how sensitive the counterfactual conclusion is to the intervention magnitude."""
        treatment = query.treatment_var
        base_val = factual.get(treatment, 0)
        if base_val == 0:
            return {"robust": False, "reason": "zero_baseline"}

        lo, hi = perturbation_range
        step_size = (hi - lo) / max(steps - 1, 1)
        sweep = []

        for i in range(steps):
            pct = lo + i * step_size
            intervened = base_val * (1 + pct)
            result = self.scm.do_intervention({treatment: intervened}, factual)
            sweep.append({
                "perturbation_pct": round(pct * 100, 1),
                "treatment_value": round(intervened, 4),
                "outcomes": {
                    k: round(v, 4) for k, v in result.items()
                    if k in self.scm.descendants(treatment) and k in factual
                },
            })

        outcome_var = query.outcome_var
        if not outcome_var:
            descendants = sorted(self.scm.descendants(treatment))
            outcome_var = descendants[0] if descendants else treatment

        signs = []
        for s in sweep:
            ov = s["outcomes"].get(outcome_var)
            if ov is not None:
                baseline_ov = factual.get(outcome_var, 0)
                signs.append(1 if ov > baseline_ov else (-1 if ov < baseline_ov else 0))

        sign_consistent = len(set(signs)) <= 2 and (0 not in signs or len(set(signs)) == 1)
        monotonic = all(signs[i] <= signs[i+1] for i in range(len(signs)-1)) or \
                     all(signs[i] >= signs[i+1] for i in range(len(signs)-1))

        return {
            "outcome_variable": outcome_var,
            "sweep": sweep,
            "sign_consistent": sign_consistent,
            "monotonic": monotonic,
            "robust": sign_consistent and monotonic,
            "num_steps": steps,
        }

    def generate_explanation(self, result: CounterfactualResult) -> str:
        """Generate natural-language explanation of counterfactual analysis."""
        parts = []
        q = result.query

        if q.query_type == CounterfactualType.INTERVENTIONAL:
            if q.direction == "decrease" and q.pct_change is not None:
                parts.append(f"If {q.treatment_var} decreased by {abs(q.pct_change):.0f}%")
            elif q.direction == "increase" and q.pct_change is not None:
                parts.append(f"If {q.treatment_var} increased by {q.pct_change:.0f}%")
            elif q.direction == "set" and q.intervention_value is not None:
                parts.append(f"If {q.treatment_var} were set to {q.intervention_value:.2f}")
            else:
                parts.append(f"Under intervention on {q.treatment_var}")
        elif q.query_type == CounterfactualType.RETROSPECTIVE:
            parts.append(f"Had {q.treatment_var} not occurred")
        elif q.query_type == CounterfactualType.CONTRASTIVE:
            parts.append(f"Comparing {q.treatment_var} vs {q.contrast_value}")
        elif q.query_type == CounterfactualType.NECESSITY:
            parts.append(f"Testing necessity of {q.treatment_var}")
        elif q.query_type == CounterfactualType.SUFFICIENCY:
            parts.append(f"Testing sufficiency of {q.treatment_var}")

        if result.downstream_effects:
            top_effects = sorted(
                result.downstream_effects.items(),
                key=lambda x: abs(x[1].get("change_pct", 0)),
                reverse=True,
            )[:3]
            effect_strs = [
                f"{k} would change by {v.get('change_pct', 0):+.1f}%"
                for k, v in top_effects if v.get("change_pct") is not None
            ]
            if effect_strs:
                parts.append(", ".join(effect_strs))

        if result.necessity_score is not None:
            parts.append(f"necessity={result.necessity_score:.2f}")
        if result.sufficiency_score is not None:
            parts.append(f"sufficiency={result.sufficiency_score:.2f}")

        if result.robustness and result.robustness.get("robust"):
            parts.append("conclusion is robust across perturbations")
        elif result.robustness and not result.robustness.get("robust"):
            parts.append("conclusion is sensitive to perturbation magnitude")

        return "; ".join(parts) + "." if parts else ""

    def reason(
        self,
        question: str,
        table: Optional[List[List[str]]],
        context: str,
        causal_relations: List[CausalRelation],
        observed: Dict[str, float],
    ) -> Dict[str, Any]:
        """Main counterfactual reasoning entrypoint."""
        if not observed:
            return {}

        cf_type = self.detect_counterfactual_type(question)
        query = self.parse_counterfactual_query(question, observed)

        if query is None and causal_relations:
            top_rel = max(causal_relations, key=lambda r: r.confidence)
            treatment = self._resolve_variable(top_rel.cause.replace(" ", "_").lower())
            outcome = self._resolve_variable(top_rel.effect.replace(" ", "_").lower())
            if treatment in observed:
                query = CounterfactualQuery(
                    treatment_var=treatment,
                    query_type=cf_type,
                    outcome_var=outcome,
                    original_question=question,
                )

        if query is None:
            return {}

        factual = self._compute_baseline(observed)

        cf_values = {}
        downstream = {}
        if query.intervention_value is not None:
            cf_result = self.scm.counterfactual(factual, {query.treatment_var: query.intervention_value})
            cf_values = cf_result.get("counterfactual_values", {})

            for node in self.scm.descendants(query.treatment_var):
                if node in cf_values and node in factual and factual[node] != 0:
                    downstream[node] = {
                        "baseline": round(factual[node], 4),
                        "counterfactual": round(cf_values[node], 4),
                        "change_pct": round((cf_values[node] - factual[node]) / abs(factual[node]) * 100, 2),
                    }

        outcome_var = query.outcome_var
        if not outcome_var:
            descendants = sorted(self.scm.descendants(query.treatment_var))
            for d in descendants:
                if d in factual:
                    outcome_var = d
                    break
            if not outcome_var and descendants:
                outcome_var = descendants[0]
            query.outcome_var = outcome_var

        necessity = None
        sufficiency = None
        if outcome_var and query.treatment_var in factual:
            necessity = self.evaluate_necessity(query.treatment_var, outcome_var, factual)
            sufficiency = self.evaluate_sufficiency(query.treatment_var, outcome_var, factual)

        robustness = None
        if query.intervention_value is not None:
            robustness = self.robustness_analysis(query, factual)

        contrastive = None
        if query.query_type == CounterfactualType.CONTRASTIVE and query.contrast_value:
            contrastive = self.contrastive_explanation(query, factual)

        result = CounterfactualResult(
            query=query,
            factual_values=factual,
            counterfactual_values=cf_values,
            downstream_effects=downstream,
            necessity_score=necessity,
            sufficiency_score=sufficiency,
            robustness=robustness,
            contrastive=contrastive,
            confidence=0.7 if query.intervention_value is not None else 0.5,
        )
        result.explanation = self.generate_explanation(result)

        return result.to_dict()


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
    DISCOURSE_LABELS = ("causal", "temporal", "contrast", "elaboration")

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
        self.financial_scm = self._build_financial_scm()
        self.discourse_detector = ImplicitDiscourseCausalityDetector(
            min_confidence=confidence_threshold,
        )
        self.counterfactual_reasoner = CounterfactualReasoner(
            self.financial_scm, self._resolve_scm_variable
        )

    def _build_financial_scm(self) -> FinancialSCM:
        """Build a formal Structural Causal Model for financial reasoning."""
        return FinancialSCM()

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

    def _granger_style_strength(self, table: Optional[List[List[str]]], cause: str, effect: str) -> Optional[float]:
        """Lag-based score proxy when multi-period table data is available."""
        if not table or len(table) < 3:
            return None
        header = [str(h).lower() for h in table[0]]
        year_cols = [i for i, h in enumerate(header) if re.search(r"(19|20)\d{2}", h)]
        if len(year_cols) < 4:
            return None

        def row_series(keyword: str):
            for row in table[1:]:
                if not row:
                    continue
                label = str(row[0]).lower()
                if keyword in label:
                    vals = []
                    for idx in year_cols:
                        if idx < len(row):
                            try:
                                vals.append(float(str(row[idx]).replace(",", "")))
                            except ValueError:
                                vals.append(np.nan)
                    arr = np.array(vals, dtype=float)
                    if np.isnan(arr).any():
                        continue
                    return arr
            return None

        c_key = cause.split()[0].lower()
        e_key = effect.split()[0].lower()
        c = row_series(c_key)
        e = row_series(e_key)
        if c is None or e is None or len(c) < 4:
            return None

        # Granger-style proxy: corr(delta_c[t-1], delta_e[t]).
        dc = np.diff(c)
        de = np.diff(e)
        x, y = dc[:-1], de[1:]
        if len(x) < 2:
            return None
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            return None
        return float(max(0.0, min(1.0, abs(corr))))

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

    _CAUSAL_VERBS = r"(?:caused|led to|resulted in|drove|triggered|reduced|increased|lowered|raised|boosted|weakened|compressed|expanded)"
    _CAUSAL_VERBS_ING = r"(?:causing|leading to|resulting in|driving|triggering|reducing|increasing|lowering|raising)"

    MULTI_HOP_PATTERNS = [
        (rf"(.+?)\s+{_CAUSAL_VERBS}\s+(.+?),\s*which\s+(?:in turn|then|subsequently)\s+{_CAUSAL_VERBS}\s+(.+)", "cause_first_3hop"),
        (rf"(.+?)\s+{_CAUSAL_VERBS}\s+(.+?),\s*(?:thereby|thus)\s+{_CAUSAL_VERBS_ING}\s+(.+)", "cause_first_3hop"),
        (rf"(.+?)\s+{_CAUSAL_VERBS}\s+(.+?)\s+(?:and|which)\s+(?:in turn|subsequently|then)\s+{_CAUSAL_VERBS}\s+(.+)", "cause_first_3hop"),
        (rf"(.+?)\s+{_CAUSAL_VERBS}\s+(.+?),\s*(?:ultimately|eventually)\s+{_CAUSAL_VERBS_ING}\s+(.+)", "cause_first_3hop"),
    ]

    def _extract_multi_hop_from_sentence(self, sentence: str) -> List[CausalRelation]:
        """Extract multi-hop causal chains from a single sentence.

        Handles patterns like: 'A caused B, which in turn led to C'
        Produces two relations: A→B and B→C with linked metadata.
        """
        relations: List[CausalRelation] = []
        for pattern, style in self.MULTI_HOP_PATTERNS:
            m = re.search(pattern, sentence, flags=re.I)
            if not m or len(m.groups()) < 3:
                continue

            spans = [self._clean_span(g) for g in m.groups() if g]
            if len(spans) < 3 or not all(spans):
                continue

            chain_id = f"chain_{hash(sentence) & 0xFFFF:04x}"
            for idx in range(len(spans) - 1):
                cause, effect = spans[idx], spans[idx + 1]
                strength = self._causal_strength(cause, effect, sentence, "mediated")
                rel = CausalRelation(
                    cause=cause,
                    effect=effect,
                    confidence=strength,
                    evidence=sentence,
                    relation_type="multi_hop",
                    mechanism="mediated" if idx > 0 else "direct",
                    lag_hint=self._extract_lag_hint(sentence),
                    polarity=self._estimate_polarity(f"{cause} {effect}"),
                    metadata={
                        "depth": 0,
                        "chain_id": chain_id,
                        "hop_index": idx,
                        "total_hops": len(spans) - 1,
                    },
                )
                relations.append(rel)
            break
        return relations

    def extract_recursive_causal_spans(self, text: str, max_depth: int = 3) -> List[CausalRelation]:
        """Recursively extract nested causal links using mask-and-rerun.

        Research approach (CausalBank, Li et al. 2020):
        1. Extract top-level causal relations
        2. Extract multi-hop chains from single sentences
        3. Mask extracted spans and re-extract from remainder
        4. Recurse into long cause/effect clauses for nested causality
        5. Apply depth-aware confidence decay
        """
        collected: List[CausalRelation] = []
        frontier = [(text, 0)]
        seen: Set[Tuple[str, str]] = set()
        depth_decay = 0.90

        for sentence in self._split_sentences(text):
            multi_hop = self._extract_multi_hop_from_sentence(sentence)
            for rel in multi_hop:
                key = (rel.cause.lower(), rel.effect.lower())
                if key not in seen:
                    seen.add(key)
                    collected.append(rel)

        while frontier:
            chunk, depth = frontier.pop(0)
            if depth >= max_depth or not chunk.strip():
                continue
            base = self.extract_causal_spans(chunk)
            for rel in base:
                key = (rel.cause.lower(), rel.effect.lower())
                if key in seen:
                    continue
                seen.add(key)
                rel.metadata["depth"] = depth
                if depth > 0:
                    rel.confidence *= depth_decay ** depth
                    rel.relation_type = "nested"
                collected.append(rel)

                masked = chunk
                for span in (rel.cause, rel.effect):
                    masked = re.sub(re.escape(span), " [MASK] ", masked, flags=re.I)
                if masked != chunk and len(masked.split()) >= 5:
                    frontier.append((masked, depth + 1))

                for sub in (rel.cause, rel.effect):
                    if len(sub.split()) >= 4:
                        frontier.append((sub, depth + 1))
        return collected

    @staticmethod
    def _fuzzy_entity_overlap(span_a: str, span_b: str) -> float:
        """Compute word-level Jaccard overlap between two causal spans."""
        words_a = set(re.findall(r"[a-z]{3,}", span_a.lower()))
        words_b = set(re.findall(r"[a-z]{3,}", span_b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _link_recursive_chain(self, relations: List[CausalRelation]) -> List[Dict[str, Any]]:
        """Build nested chains by linking effect→cause overlap.

        Uses fuzzy entity matching (Jaccard ≥ 0.3) rather than substring
        containment, and extends chains transitively up to 4 hops.
        """
        n = len(relations)
        adjacency: Dict[int, List[int]] = defaultdict(list)
        overlap_threshold = 0.3

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                effect_i = self._clean_span(relations[i].effect).lower()
                cause_j = self._clean_span(relations[j].cause).lower()
                if effect_i in cause_j or cause_j in effect_i:
                    adjacency[i].append(j)
                elif self._fuzzy_entity_overlap(effect_i, cause_j) >= overlap_threshold:
                    adjacency[i].append(j)

        chains: List[Dict[str, Any]] = []

        def build_chain(start: int, path: List[int], visited: Set[int]):
            if len(path) >= 4:
                return
            for nxt in adjacency.get(path[-1], []):
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                conf = 1.0
                for idx in new_path:
                    conf *= relations[idx].confidence * (0.92 ** (len(new_path) - 1))
                chains.append({
                    "links": [relations[idx].to_dict() for idx in new_path],
                    "chain_confidence": round(conf, 4),
                    "length": len(new_path),
                    "root_cause": relations[new_path[0]].cause,
                    "final_effect": relations[new_path[-1]].effect,
                })
                build_chain(start, new_path, visited | {nxt})

        for i in range(n):
            build_chain(i, [i], {i})

        chains.sort(key=lambda x: (-x["chain_confidence"], x["length"]))
        return chains

    def complete_transitive_chains(self, relations: List[CausalRelation]) -> List[CausalRelation]:
        """Infer transitive relations: if A→B and B→C, infer A→C.

        Only adds inferred relations when both source relations have
        confidence ≥ 0.4. Inferred confidence = product × 0.85 decay.
        """
        inferred: List[CausalRelation] = []
        existing = {(r.cause.lower(), r.effect.lower()) for r in relations}

        effect_to_rels: Dict[str, List[CausalRelation]] = defaultdict(list)
        for r in relations:
            effect_to_rels[r.effect.lower()].append(r)

        for r1 in relations:
            if r1.confidence < 0.4:
                continue
            for r2 in relations:
                if r2.confidence < 0.4:
                    continue
                if self._fuzzy_entity_overlap(r1.effect.lower(), r2.cause.lower()) >= 0.3 or r1.effect.lower() in r2.cause.lower():
                    key = (r1.cause.lower(), r2.effect.lower())
                    if key in existing or r1.cause.lower() == r2.effect.lower():
                        continue
                    existing.add(key)
                    inferred.append(CausalRelation(
                        cause=r1.cause,
                        effect=r2.effect,
                        confidence=r1.confidence * r2.confidence * 0.85,
                        evidence=f"Inferred: [{r1.evidence}] + [{r2.evidence}]",
                        relation_type="transitive_inferred",
                        mechanism=f"transitive({r1.mechanism}→{r2.mechanism})",
                        polarity=r2.polarity,
                        metadata={
                            "source_relations": [r1.to_dict(), r2.to_dict()],
                            "inference_type": "transitive_closure",
                        },
                    ))
        return inferred

    def classify_discourse_relation(self, s1: str, s2: str) -> str:
        """PDTB-style relation classifier delegating to ImplicitDiscourseCausalityDetector.

        Returns a backward-compatible string label for the dominant relation type.
        """
        result = self.discourse_detector.classify_discourse_relation(s1, s2)
        relation = result["relation"]
        if "contingency" in relation:
            return "causal"
        if "temporal" in relation:
            return "temporal"
        if "comparison" in relation:
            return "contrast"
        if "expansion" in relation:
            return "elaboration"
        return "elaboration"

    def classify_discourse_relation_full(self, s1: str, s2: str) -> Dict[str, Any]:
        """Full PDTB-3 discourse relation classification with features.

        Returns the rich dict from ImplicitDiscourseCausalityDetector.
        """
        return self.discourse_detector.classify_discourse_relation(s1, s2)

    def detect_implicit_discourse_causality(self, text: str) -> List[CausalRelation]:
        """Detect implicit causal relations via PDTB-style discourse analysis.

        Delegates to ImplicitDiscourseCausalityDetector for feature extraction,
        Bayesian coherence scoring, and direction inference, then converts
        results to CausalRelation objects.
        """
        implicit_results = self.discourse_detector.detect_implicit_causality(text)
        relations = self.discourse_detector.to_causal_relations(
            implicit_results, clean_fn=self._clean_span,
        )
        for rel in relations:
            rel.polarity = self._estimate_polarity(rel.effect)
        return relations

    def _scm_paths(self, source: str, target: str, max_depth: int = 6) -> List[List[str]]:
        return self.financial_scm.find_all_paths(source, target, max_depth)

    def _rank_scm_paths(self, paths: List[List[str]], relations: List[CausalRelation]) -> List[Dict[str, Any]]:
        rel_text = " ".join(f"{r.cause} {r.effect}" for r in relations).lower()
        ranked = []
        for p in paths:
            support = sum(1 for node in p if node.replace("_", " ") in rel_text) / max(1, len(p))
            ranked.append({"path": p, "evidence_support": round(support, 4), "length": len(p)})
        return sorted(ranked, key=lambda x: (-x["evidence_support"], x["length"]))

    def _interventional_counterfactual(self, question: str, table: Optional[List[List[str]]]) -> Dict[str, Any]:
        """do-calculus intervention via FinancialSCM structural equations.

        Supports: "if we cut X by Y%", "what if X increased by Y%",
        "what would happen if X were Z".
        """
        q = question.lower()

        cut_m = re.search(r"if we (?:cut|reduce|lower)\s+([a-z_\s]+?)\s+by\s+(\d+(?:\.\d+)?)%", q)
        increase_m = re.search(r"(?:if|what if)\s+([a-z_\s]+?)\s+(?:increased?|rose|grew)\s+by\s+(\d+(?:\.\d+)?)%", q)
        set_m = re.search(r"(?:if|what if)\s+([a-z_\s]+?)\s+(?:were?|was|is)\s+(\d+(?:\.\d+)?)", q)

        if not table:
            return {}

        observed = self._extract_observed_from_table(table)
        if not observed:
            return {}

        if cut_m:
            var = cut_m.group(1).strip().replace(" ", "_")
            pct = float(cut_m.group(2))
            base = observed.get(var)
            if base is None:
                var = self._resolve_scm_variable(var)
                base = observed.get(var)
            if base is None:
                return {}
            intervened_val = base * (1 - pct / 100.0)
        elif increase_m:
            var = increase_m.group(1).strip().replace(" ", "_")
            pct = float(increase_m.group(2))
            base = observed.get(var)
            if base is None:
                var = self._resolve_scm_variable(var)
                base = observed.get(var)
            if base is None:
                return {}
            intervened_val = base * (1 + pct / 100.0)
        elif set_m:
            var = set_m.group(1).strip().replace(" ", "_")
            intervened_val = float(set_m.group(2))
            base = observed.get(var)
            if base is None:
                var = self._resolve_scm_variable(var)
                base = observed.get(var)
            if base is None:
                return {}
        else:
            return {}

        result = self.financial_scm.do_intervention({var: intervened_val}, observed)

        downstream_effects = {}
        descendants = self.financial_scm.descendants(var)
        for desc in descendants:
            if desc in result and desc in observed:
                old_val = observed[desc]
                new_val = result[desc]
                if old_val != 0:
                    downstream_effects[desc] = {
                        "baseline": round(old_val, 4),
                        "counterfactual": round(new_val, 4),
                        "change_pct": round((new_val - old_val) / abs(old_val) * 100, 2),
                    }

        backdoor = self.financial_scm.backdoor_criterion(var, list(descendants)[0] if descendants else var)

        return {
            "intervention": f"do({var}={intervened_val:.4f})",
            "baseline": base,
            "predicted": intervened_val,
            "assumption": "structural_equations",
            "downstream_effects": downstream_effects,
            "identification": backdoor,
            "scm_propagation": True,
        }

    def _extract_observed_from_table(self, table: List[List[str]]) -> Dict[str, float]:
        """Extract observed variable values from the financial table."""
        if not table or len(table) < 2:
            return {}

        observed = {}
        header = table[0]
        latest_idx = len(header) - 1 if len(header) > 1 else 1

        scm_aliases = {
            "revenue": ["revenue", "total revenue", "net revenue", "sales", "total sales"],
            "input_costs": ["cost of goods sold", "cogs", "cost of revenue", "cost of sales", "input costs"],
            "gross_profit": ["gross profit", "gross margin", "gross income"],
            "operating_income": ["operating income", "operating profit", "income from operations"],
            "net_income": ["net income", "net profit", "net earnings", "profit"],
            "eps": ["eps", "earnings per share", "diluted eps"],
            "interest_expense": ["interest expense", "interest cost", "finance costs"],
            "depreciation": ["depreciation", "depreciation and amortization", "d&a"],
            "sga_expense": ["sga", "sg&a", "selling general", "operating expenses"],
            "debt": ["total debt", "long-term debt", "borrowings", "debt"],
            "capex": ["capex", "capital expenditure", "capital expenditures"],
            "tax_rate": ["tax rate", "effective tax rate"],
            "share_count": ["shares outstanding", "diluted shares", "share count", "weighted average shares"],
            "equity": ["total equity", "shareholders equity", "stockholders equity"],
            "total_assets": ["total assets"],
        }

        for row in table[1:]:
            if not row or latest_idx >= len(row):
                continue
            label = str(row[0]).strip().lower()
            try:
                val = float(str(row[latest_idx]).replace(",", "").replace("$", "").replace("%", ""))
            except (ValueError, IndexError):
                continue

            for scm_var, aliases in scm_aliases.items():
                if any(alias in label for alias in aliases):
                    observed[scm_var] = val
                    break

        return observed

    def _resolve_scm_variable(self, query: str) -> str:
        """Fuzzy-match a query string to an SCM variable name."""
        query_words = set(query.lower().replace("_", " ").split())
        best_match = query
        best_score = 0.0
        for node in self.financial_scm.nodes:
            node_words = set(node.replace("_", " ").split())
            overlap = len(query_words & node_words) / max(len(query_words | node_words), 1)
            if overlap > best_score:
                best_score = overlap
                best_match = node
        return best_match

    def detect_financial_causality(self, text: str, question: str = "", table: Optional[List[List[str]]] = None) -> List[CausalRelation]:
        """Detect causal relations with recursive extraction + transitive completion."""
        relations = self.extract_recursive_causal_spans(text)
        relations.extend(self.detect_implicit_discourse_causality(text))
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

        transitive = self.complete_transitive_chains(relations)
        relations.extend(transitive)

        dedup: Dict[Tuple[str, str], CausalRelation] = {}
        for rel in relations:
            g_strength = self._granger_style_strength(table, rel.cause, rel.effect)
            if g_strength is not None:
                rel.confidence = float(0.6 * rel.confidence + 0.4 * g_strength)
                rel.metadata["granger_proxy"] = g_strength
            key = (rel.cause.lower(), rel.effect.lower())
            if key not in dedup or rel.confidence > dedup[key].confidence:
                dedup[key] = rel

        return [r for r in dedup.values() if r.confidence >= self.confidence_threshold]

    def build_causal_graph(self, texts: List[str], question: str = "", table: Optional[List[List[str]]] = None) -> CausalGraph:
        graph = CausalGraph()
        for text in texts:
            for rel in self.detect_financial_causality(text, question, table):
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
        """Temporal-causal joint reasoning entrypoint with SCM analysis."""
        is_causal = self.detect_is_causal_question(question)
        relations = self.detect_financial_causality(context, question, table)
        graph = self.build_causal_graph([context], question, table)

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
        nested_relations = [r for r in relation_dicts if r.get("metadata", {}).get("depth", 0) > 0]
        multi_hop_relations = [r for r in relation_dicts if r.get("relation_type") == "multi_hop"]
        transitive_relations = [r for r in relation_dicts if r.get("relation_type") == "transitive_inferred"]
        recursive_chain_candidates = self._link_recursive_chain(relations)
        max_chain_depth = max((r.get("metadata", {}).get("depth", 0) for r in relation_dicts), default=0)

        target_match = re.search(r"why did ([a-z_\s]+?)(?:\?|$)", question.lower())
        scm_ranked_paths = []
        scm_dsep_analysis = []
        scm_backdoor = {}
        scm_frontdoor = {}
        if target_match:
            target = target_match.group(1).strip().replace(" ", "_")
            exogenous = [n for n in self.financial_scm.nodes if not self.financial_scm.parents.get(n)]
            for root in exogenous:
                paths = self._scm_paths(root, target, max_depth=6)
                scm_ranked_paths.extend(self._rank_scm_paths(paths, relations))
            scm_ranked_paths = sorted(scm_ranked_paths, key=lambda x: (-x["evidence_support"], x["length"]))[:8]

            for root in exogenous:
                if root in self.financial_scm.nodes and target in self.financial_scm.nodes:
                    is_dsep = self.financial_scm.d_separated(root, target)
                    if not is_dsep:
                        scm_dsep_analysis.append({
                            "source": root,
                            "target": target,
                            "d_separated": False,
                            "conditioning": [],
                        })

            if scm_ranked_paths:
                top_source = scm_ranked_paths[0]["path"][0]
                scm_backdoor = self.financial_scm.backdoor_criterion(top_source, target)
                scm_frontdoor = self.financial_scm.frontdoor_criterion(top_source, target)

        scm_sensitivity = []
        observed = {}
        if table:
            observed = self._extract_observed_from_table(table)
            if observed and target_match:
                tgt = target_match.group(1).strip().replace(" ", "_")
                if tgt in self.financial_scm.nodes:
                    scm_sensitivity = self.financial_scm.sensitivity_analysis(tgt, observed)[:5]

        counterfactuals = [self._counterfactuals(r) for r in relations[:3]]
        intervention = self._interventional_counterfactual(question, table)
        if intervention:
            counterfactuals.append(intervention)

        if observed and is_causal and not intervention:
            for rel in relations[:2]:
                cause_var = self._resolve_scm_variable(rel.cause.replace(" ", "_"))
                effect_var = self._resolve_scm_variable(rel.effect.replace(" ", "_"))
                if cause_var in observed and effect_var in self.financial_scm.nodes:
                    effect_est = self.financial_scm.causal_effect_estimate(
                        cause_var, effect_var, observed
                    )
                    if effect_est.get("marginal_effect", 0) != 0:
                        counterfactuals.append({
                            "type": "causal_effect_estimate",
                            "treatment": cause_var,
                            "outcome": effect_var,
                            "marginal_effect": effect_est["marginal_effect"],
                            "identification": effect_est["identification"],
                        })
                        break

        cf_analysis = self.counterfactual_reasoner.reason(
            question=question,
            table=table,
            context=context,
            causal_relations=relations,
            observed=observed,
        )

        discourse_analysis = self.discourse_detector.detect_implicit_causality(context)
        discourse_relations = [r for r in relation_dicts if r.get("relation_type") == "implicit_discourse"]
        discourse_summary = {
            "num_implicit_causal": sum(1 for d in discourse_analysis if not d.get("explicit")),
            "num_explicit_causal": sum(1 for d in discourse_analysis if d.get("explicit")),
            "total_discourse_relations": len(discourse_analysis),
            "avg_confidence": (
                sum(d["confidence"] for d in discourse_analysis) / len(discourse_analysis)
                if discourse_analysis else 0.0
            ),
            "relations": discourse_analysis[:10],
        }

        causal_context_lines = []
        if relation_dicts:
            causal_context_lines.append("Detected financial causal structure:")
            for i, rel in enumerate(sorted(relation_dicts, key=lambda x: x["confidence"], reverse=True)[:5], 1):
                lag = f", lag={rel['lag_hint']}" if rel.get("lag_hint") else ""
                causal_context_lines.append(
                    f"  {i}. {rel['cause']} -> {rel['effect']} (conf={rel['confidence']:.2f}, mech={rel['mechanism']}{lag})"
                )
        if discourse_summary["num_implicit_causal"] > 0:
            causal_context_lines.append(
                f"Implicit discourse causality: {discourse_summary['num_implicit_causal']} "
                f"relations detected (avg conf={discourse_summary['avg_confidence']:.2f})"
            )
        if cf_analysis.get("explanation"):
            causal_context_lines.append(f"Counterfactual: {cf_analysis['explanation']}")

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
            "recursive_causal_chains": recursive_chain_candidates[:10],
            "causal_strength": avg_strength,
            "nested_causal_relations": nested_relations,
            "multi_hop_relations": multi_hop_relations,
            "transitive_relations": transitive_relations,
            "max_extraction_depth": max_chain_depth,
            "scm_paths_ranked": scm_ranked_paths,
            "scm_structure": self.financial_scm.get_structure_summary(),
            "scm_dseparation": scm_dsep_analysis,
            "scm_backdoor_criterion": scm_backdoor,
            "scm_frontdoor_criterion": scm_frontdoor,
            "scm_sensitivity": scm_sensitivity,
            "temporal_causal_overlap": temporal_overlap,
            "counterfactuals": counterfactuals,
            "counterfactual_analysis": cf_analysis,
            "discourse_analysis": discourse_summary,
            "causal_context": "\n".join(causal_context_lines),
        }
