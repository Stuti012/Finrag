"""Integrated Financial QA Pipeline.

Orchestrates the full question answering process:
1. Question classification
2. Hybrid retrieval (table + text)
3. Reasoning module selection and execution
4. Evidence aggregation
5. Answer generation via LLM
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

from .data.finqa_loader import FinQAExample, classify_question_type
from .reasoning.causality_detector import CausalityDetector
from .reasoning.numerical_reasoner import NumericalReasoner
from .reasoning.question_classifier import QuestionClassifier
from .reasoning.temporal_reasoner import TemporalReasoner
from .retrieval.hybrid_retriever import (
    FinancialDocumentIndexer,
    HybridRetriever,
)
from .utils.financial_utils import format_table_for_llm

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class LLMInterface:
    """Interface for LLM inference via local model or HuggingFace Inference API.

    Supports two modes:
    1. Local: loads model via transformers (requires gated model access + GPU)
    2. API: uses HuggingFace Inference API via OpenAI-compatible client
       Set HF_TOKEN env var and use_api=True (or set api_base/api_key directly)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        use_api: bool = False,
        api_base: str = None,
        api_key: str = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.api_client = None

        # Try API mode first if requested or if HF_TOKEN is available
        if use_api or (api_key or os.environ.get("HF_TOKEN")):
            self._init_api(model_name, api_base, api_key)
            if self.api_client is not None:
                return

        # Fall back to local model loading
        if HAS_TRANSFORMERS:
            try:
                print(f"Loading LLM locally: {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                load_kwargs = {
                    "device_map": device_map,
                    "trust_remote_code": True,
                }

                if load_in_4bit:
                    try:
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        load_kwargs["quantization_config"] = bnb_config
                    except Exception:
                        load_kwargs["torch_dtype"] = torch.float16

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, **load_kwargs
                )
                self.model.eval()
                print(f"LLM loaded successfully: {model_name}")
            except Exception as e:
                print(f"Warning: Could not load LLM ({e}). Using rule-based fallback.")
                self.model = None
                self.tokenizer = None

    def _init_api(self, model_name: str, api_base: str = None, api_key: str = None):
        """Initialize API-based inference via OpenAI-compatible client."""
        if not HAS_OPENAI:
            print("Warning: openai package not installed. Run: pip install openai")
            return

        resolved_key = api_key or os.environ.get("HF_TOKEN")
        if not resolved_key:
            print("Warning: No API key found. Set HF_TOKEN env var or pass api_key.")
            return

        resolved_base = api_base or "https://router.huggingface.co/v1"

        try:
            self.api_client = OpenAI(base_url=resolved_base, api_key=resolved_key)
            print(f"LLM API initialized: {model_name} via {resolved_base}")
        except Exception as e:
            print(f"Warning: Could not initialize API client ({e}).")
            self.api_client = None

    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generate text from prompt using API or local model."""
        if self.api_client is not None:
            return self._generate_api(prompt, max_new_tokens)
        if self.model is not None and self.tokenizer is not None:
            return self._generate_local(prompt, max_new_tokens)
        return self._rule_based_fallback(prompt)

    def _generate_api(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generate text via HuggingFace Inference API."""
        max_tokens = max_new_tokens or self.max_new_tokens
        try:
            completion = self.api_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=max(self.temperature, 0.01),
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"API generation error: {e}")
            return self._rule_based_fallback(prompt)

    def _generate_local(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generate text from prompt using the locally loaded LLM."""
        max_tokens = max_new_tokens or self.max_new_tokens

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(self.temperature, 0.01),
                    do_sample=self.temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return generated.strip()

        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._rule_based_fallback(prompt)

    def _rule_based_fallback(self, prompt: str) -> str:
        """Simple rule-based fallback when LLM is not available."""
        return "[LLM not available - using computed result from reasoning modules]"

    @property
    def is_available(self) -> bool:
        return self.model is not None or self.api_client is not None


class FinancialQAPipeline:
    """Complete Financial QA Pipeline integrating all reasoning modules.

    Architecture:
    1. Question Classifier -> routes to appropriate modules
    2. Hybrid Retriever -> retrieves relevant tables and text
    3. Numerical Reasoner -> Program-of-Thought execution
    4. Temporal Reasoner -> temporal graph and trend analysis
    5. Causality Detector -> causal relation extraction
    6. LLM -> final answer generation with verified evidence
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        load_llm: bool = True,
        load_in_4bit: bool = True,
        use_api: bool = False,
        api_base: str = None,
        api_key: str = None,
    ):
        # Initialize components
        self.classifier = QuestionClassifier()
        self.retriever = HybridRetriever(embedding_model=embedding_model)
        self.indexer = FinancialDocumentIndexer(self.retriever)
        self.numerical_reasoner = NumericalReasoner()
        self.temporal_reasoner = TemporalReasoner()
        self.causality_detector = CausalityDetector()

        # Initialize LLM
        if load_llm:
            self.llm = LLMInterface(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                use_api=use_api,
                api_base=api_base,
                api_key=api_key,
            )
        else:
            self.llm = LLMInterface.__new__(LLMInterface)
            self.llm.model = None
            self.llm.tokenizer = None
            self.llm.api_client = None
            self.llm.model_name = model_name
            self.llm.max_new_tokens = 512
            self.llm.temperature = 0.1

    def answer(self, example: FinQAExample) -> Dict[str, Any]:
        """Answer a FinQA question using the full pipeline.

        Args:
            example: A FinQAExample with question, table, and context.

        Returns:
            Dict with answer, reasoning traces, and module outputs.
        """
        start_time = time.time()

        result = {
            "id": example.id,
            "question": example.question,
            "gold_answer": example.answer,
            "predicted_answer": None,
            "classification": {},
            "retrieval": {},
            "numerical": {},
            "temporal": {},
            "causal": {},
            "reasoning_trace": [],
            "time_seconds": 0,
        }

        # Step 1: Classify the question (from question text only, no gold program)
        classification = self.classifier.classify(example.question)
        active_modules = self.classifier.get_active_modules(example.question)
        joint_temporal_causal = classification.get("temporal_causal_joint", 0.0)
        result["classification"] = {
            "scores": classification,
            "active_modules": active_modules,
            "primary_type": self.classifier.get_primary_type(example.question),
            "temporal_causal_joint": joint_temporal_causal,
        }
        result["reasoning_trace"].append(
            f"Classification: {result['classification']['primary_type']} "
            f"(active: {active_modules})"
        )

        # Step 2: Retrieve relevant context (interleaved retrieve-then-reason)
        retrieval_result = self.indexer.retrieve_for_question(example.question, example)
        result["retrieval"] = {
            "table_contexts": [
                {"text": r["document"], "score": r["score"]}
                for r in retrieval_result.get("table_contexts", [])
            ],
            "text_contexts": [
                {"text": r["document"], "score": r["score"]}
                for r in retrieval_result.get("text_contexts", [])
            ],
            "interleaved_steps": [],
        }

        # Step 3: Run reasoning modules (pass 1)
        context_text = example.context_text
        if result["retrieval"]["text_contexts"]:
            context_text = " ".join([c["text"] for c in result["retrieval"]["text_contexts"][:3]])[:2500]
        temporal_signals = self._run_reasoners(result, active_modules, example, context_text, pass_label="pass1")

        # Interleaved targeted re-retrieval (IRCoT-inspired)
        gap_query = self._identify_reasoning_gap(result, active_modules, example.question)
        if gap_query:
            targeted = self._targeted_reretrieve(example, gap_query)
            result["retrieval"]["interleaved_steps"].append(
                {"gap_query": gap_query, "hits": len(targeted.get("text_contexts", []))}
            )
            if targeted.get("text_contexts"):
                merged_context = " ".join([context_text] + [r["document"] for r in targeted["text_contexts"][:2]])[:3500]
                temporal_signals = self._run_reasoners(result, active_modules, example, merged_context, pass_label="pass2")

        # Step 4: Cross-module attention and answer aggregation
        result["cross_module_attention"] = self._compute_cross_module_attention(result)
        if joint_temporal_causal >= 0.4:
            result["reasoning_trace"].append(f"Temporal-causal joint reasoning activated (score={joint_temporal_causal:.2f})")
        predicted = self._aggregate_answer(result, example)
        result["predicted_answer"] = predicted
        result["verification_backward"] = self._verify_backward_chain(result, predicted)

        # Step 5: Verify
        if result["numerical"].get("result") is not None:
            verification = self.numerical_reasoner.verify_result(
                result["numerical"]["result"], example.answer
            )
            result["verification"] = verification

        result["time_seconds"] = time.time() - start_time
        return result


    def _compute_cross_module_attention(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Compute lightweight cross-module attention weights for answer aggregation."""
        numerical_conf = 0.0
        if result.get("numerical", {}).get("success") and result.get("numerical", {}).get("result") is not None:
            numerical_conf = 0.9

        temporal_entities = len(result.get("temporal", {}).get("temporal_entities", []))
        temporal_conf = min(0.85, 0.2 + temporal_entities * 0.08) if result.get("temporal") else 0.0

        causal_conf = float(result.get("causal", {}).get("causal_strength", 0.0))
        if result.get("causal", {}).get("causal_relations"):
            causal_conf = max(causal_conf, 0.35)

        retrieval_signal = 0.0
        table_ctx = result.get("retrieval", {}).get("table_contexts", [])
        text_ctx = result.get("retrieval", {}).get("text_contexts", [])
        if table_ctx or text_ctx:
            scores = [c.get("score", 0.0) for c in table_ctx + text_ctx]
            retrieval_signal = min(0.8, sum(scores) / max(1, len(scores))) if scores else 0.2

        raw = {
            "numerical": numerical_conf,
            "temporal": temporal_conf,
            "causal": causal_conf,
            "retrieval": retrieval_signal,
        }
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    def _build_temporal_signals(self, temp_result: Dict[str, Any]) -> Dict[str, Any]:
        entities = []
        for ent in temp_result.get("temporal_entities", []):
            if isinstance(ent, dict):
                entities.append(ent.get("label") or ent.get("value"))
            else:
                entities.append(str(ent))
        return {
            "entities": [e for e in entities if e],
            "trend": temp_result.get("trend_analysis", {}).get("trend") if isinstance(temp_result.get("trend_analysis"), dict) else None,
            "context": temp_result.get("temporal_context", ""),
        }

    def _run_reasoners(
        self,
        result: Dict[str, Any],
        active_modules: List[str],
        example: FinQAExample,
        context_text: str,
        pass_label: str,
    ) -> Dict[str, Any]:
        temporal_signals = {}
        if "numerical" in active_modules:
            num_result = self.numerical_reasoner.reason(
                question=example.question,
                table=example.table,
                context=context_text,
            )
            result["numerical"] = num_result
            if num_result.get("success") and num_result.get("result") is not None:
                result["reasoning_trace"].append(
                    f"{pass_label} Numerical: {num_result['method']} -> {num_result['result']}"
                )

        if "temporal" in active_modules:
            temp_result = self.temporal_reasoner.reason(
                question=example.question,
                table=example.table,
                context=context_text,
            )
            result["temporal"] = temp_result
            temporal_signals = self._build_temporal_signals(temp_result)
            if temp_result.get("temporal_context"):
                result["reasoning_trace"].append(
                    f"{pass_label} Temporal: {temp_result['temporal_context']}"
                )

        if "causal" in active_modules:
            causal_result = self.causality_detector.reason(
                question=example.question,
                context=context_text,
                table=example.table,
                temporal_signals=temporal_signals,
            )
            result["causal"] = causal_result
            if causal_result.get("causal_relations"):
                result["reasoning_trace"].append(
                    f"{pass_label} Causal: {len(causal_result['causal_relations'])} relations found"
                )
        return temporal_signals

    def _identify_reasoning_gap(
        self,
        result: Dict[str, Any],
        active_modules: List[str],
        question: str,
    ) -> Optional[str]:
        if "causal" in active_modules and not result.get("causal", {}).get("causal_relations"):
            return f"{question} why caused drivers impact"
        if "temporal" in active_modules and len(result.get("temporal", {}).get("temporal_entities", [])) < 2:
            return f"{question} before after since timeline"
        if "numerical" in active_modules:
            num = result.get("numerical", {})
            if num.get("method") == "program_of_thought" and not num.get("success"):
                return f"{question} table values formula"
        return None

    def _targeted_reretrieve(self, example: FinQAExample, query: str) -> Dict[str, Any]:
        try:
            return self.indexer.retrieve_for_question(query, example)
        except Exception:
            return {"table_contexts": [], "text_contexts": []}

    def _verify_backward_chain(self, result: Dict[str, Any], predicted: str) -> Dict[str, Any]:
        supports = []
        if result.get("numerical", {}).get("success"):
            supports.append("numerical")
        if result.get("temporal", {}).get("temporal_context"):
            supports.append("temporal")
        if result.get("causal", {}).get("causal_relations"):
            supports.append("causal")

        evidence_text = " ".join(
            c.get("text", "") for c in result.get("retrieval", {}).get("text_contexts", [])[:3]
        ).lower()
        answer_tokens = [t for t in re.findall(r"[a-z0-9.%\-]+", str(predicted).lower()) if len(t) > 2]
        overlap = sum(1 for t in answer_tokens if t in evidence_text)
        reconstructed = bool(supports) and (overlap > 0 or "numerical" in supports)
        confidence = min(1.0, 0.22 * len(supports) + (0.2 if overlap else 0.0))
        return {
            "reconstructed": reconstructed,
            "support_modules": supports,
            "token_overlap": overlap,
            "confidence": confidence,
        }

    def _aggregate_answer(
        self, result: Dict, example: FinQAExample
    ) -> str:
        """Aggregate outputs from all modules into a final answer."""
        attn = result.get("cross_module_attention", {})

        # Priority 1: Numerical computation result (most precise)
        if result["numerical"].get("success") and result["numerical"].get("result") is not None and attn.get("numerical", 0) >= 0.15:
            computed = result["numerical"]["result"]
            if isinstance(computed, float):
                return self._format_numerical_answer(computed)
            return str(computed)

        # Priority 2: If we have a PoT prompt, use LLM with self-refinement
        if result["numerical"].get("method") == "program_of_thought" and self.llm.is_available:
            pot_prompt = result["numerical"].get("pot_prompt", "")
            if pot_prompt:
                refinement = self._run_self_refining_pot(
                    example=example,
                    initial_prompt=pot_prompt,
                    max_iterations=3,
                )
                result["numerical"]["self_refinement"] = refinement
                if refinement.get("success"):
                    refined_value = refinement.get("result")
                    if isinstance(refined_value, float):
                        return self._format_numerical_answer(refined_value)
                    return str(refined_value)

        # Priority 3: LLM-based answer generation
        if self.llm.is_available:
            prompt = self._build_answer_prompt(result, example)
            answer = self.llm.generate(prompt, max_new_tokens=128)
            # Extract just the answer from LLM response
            answer = self._extract_answer_from_llm(answer)
            return answer

        # Fallback: return empty
        return ""

    def _run_self_refining_pot(
        self,
        example: FinQAExample,
        initial_prompt: str,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Iteratively repair PoT code using execution feedback.

        Research features (Madaan et al., NeurIPS 2023 — Self-Refine):
        - Execution trace forwarded to LLM so it can see extracted values
        - Progressive hint escalation across iterations
        - Self-consistency majority voting when multiple attempts succeed
        """
        attempts: List[Dict[str, Any]] = []
        prompt = initial_prompt
        previous_code = ""

        for iteration in range(1, max_iterations + 1):
            llm_response = self.llm.generate(prompt)
            code = self.numerical_reasoner.extract_code_from_response(llm_response)

            if not code:
                feedback = "No executable Python code was generated."
                attempts.append({
                    "iteration": iteration,
                    "success": False,
                    "error": feedback,
                })
                prompt = self.numerical_reasoner.generate_refinement_prompt(
                    question=example.question,
                    table=example.table,
                    context=example.context_text,
                    previous_code=previous_code or "# No code produced",
                    error_feedback=feedback,
                    iteration=iteration,
                )
                continue

            exec_result = self.numerical_reasoner.execute_python_program(code)
            plausible, plausibility_reason = self.numerical_reasoner.is_plausible_result(
                exec_result.get("result"),
                example.question,
            )
            iteration_success = (
                exec_result.get("success")
                and exec_result.get("result") is not None
                and plausible
            )

            attempts.append({
                "iteration": iteration,
                "code": code,
                "execution_success": exec_result.get("success"),
                "result": exec_result.get("result"),
                "error": exec_result.get("error"),
                "plausible": plausible,
                "plausibility_reason": plausibility_reason,
                "success": iteration_success,
            })

            if iteration_success:
                return {
                    "success": True,
                    "result": exec_result.get("result"),
                    "attempts": attempts,
                }

            feedback = exec_result.get("error") or plausibility_reason
            previous_code = code
            prompt = self.numerical_reasoner.generate_refinement_prompt(
                question=example.question,
                table=example.table,
                context=example.context_text,
                previous_code=code,
                error_feedback=feedback,
                iteration=iteration,
                exec_namespace=exec_result.get("namespace", {}),
            )

        voted = self.numerical_reasoner.select_by_majority_vote(attempts)
        if voted is not None:
            return {
                "success": True,
                "result": voted,
                "attempts": attempts,
                "best_effort": True,
                "selection_method": "majority_vote",
            }

        return {
            "success": False,
            "result": None,
            "attempts": attempts,
        }

    def _build_answer_prompt(
        self, result: Dict, example: FinQAExample
    ) -> str:
        """Build a prompt for the LLM incorporating all reasoning outputs."""
        table_str = format_table_for_llm(example.table, max_rows=30)
        context = example.context_text[:1500]

        reasoning_info = []
        attn = result.get("cross_module_attention", {})
        if attn:
            reasoning_info.append("Cross-module attention: " + ", ".join([f"{k}={v:.2f}" for k, v in attn.items()]))
        if result.get("temporal", {}).get("temporal_context"):
            reasoning_info.append(f"Temporal info: {result['temporal']['temporal_context']}")
        if result.get("causal", {}).get("causal_context"):
            reasoning_info.append(f"Causal info: {result['causal']['causal_context']}")

        reasoning_str = "\n".join(reasoning_info) if reasoning_info else "No additional reasoning context."

        prompt = f"""Answer the following financial question based on the provided data. Give ONLY the answer value, no explanation.

TABLE:
{table_str}

CONTEXT:
{context}

REASONING:
{reasoning_str}

QUESTION: {example.question}

ANSWER:"""
        return prompt

    @staticmethod
    def _format_numerical_answer(value: float) -> str:
        """Format a numerical answer preserving precision to match gold answers.

        FinQA gold answers use up to 5 decimal places for fractions and
        whole numbers for integers. This formatter adapts precision to
        the magnitude of the result.
        """
        if not isinstance(value, (int, float)):
            return str(value)
        if value != value:  # NaN
            return "0"
        if abs(value) == float("inf"):
            return "0"
        # Exact integers
        if value == int(value) and abs(value) < 1e12:
            return str(int(value))
        # Very small numbers — round to match FinQA gold answer convention
        abs_val = abs(value)
        if abs_val < 1e-4:
            # FinQA rounds very small numbers aggressively
            formatted_2g = f"{value:.2g}"
            formatted_1g = f"{value:.1g}"
            # If 1-sig-fig is a clean power of 10, prefer it (matches FinQA style)
            try:
                v1 = float(formatted_1g)
                if abs(v1 - value) / max(abs(value), 1e-15) < 0.05:
                    return formatted_1g
            except ValueError:
                pass
            return formatted_2g
        # Numbers < 1: typically ratios/percentages — use 5 decimal places
        if abs_val < 1:
            return f"{value:.5f}".rstrip("0").rstrip(".")
        # Numbers 1–100: use 5 significant figures
        if abs_val < 100:
            return f"{value:.5g}"
        # Larger numbers: use 5 significant figures
        return f"{value:.5g}"

    def _extract_answer_from_llm(self, response: str) -> str:
        """Extract the answer value from LLM response."""
        response = response.strip()
        # Take first line/sentence
        first_line = response.split("\n")[0].strip()
        # Remove common prefixes
        for prefix in ["the answer is", "answer:", "result:", "the result is"]:
            if first_line.lower().startswith(prefix):
                first_line = first_line[len(prefix):].strip()
        # Remove trailing period
        first_line = first_line.rstrip(".")
        return first_line

    def batch_answer(
        self,
        examples: List[FinQAExample],
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Answer a batch of FinQA questions."""
        results = []
        for i, example in enumerate(examples):
            if verbose and (i + 1) % 10 == 0:
                print(f"Processing {i+1}/{len(examples)}...")
            result = self.answer(example)
            results.append(result)
        return results
