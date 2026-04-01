"""Integrated Financial QA Pipeline.

Orchestrates the full question answering process:
1. Question classification
2. Hybrid retrieval (table + text)
3. Reasoning module selection and execution
4. Evidence aggregation
5. Answer generation via LLM
"""

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


class LLMInterface:
    """Interface for open-source LLM inference (Llama-based)."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

        if HAS_TRANSFORMERS:
            try:
                print(f"Loading LLM: {model_name}...")
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

    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generate text from prompt using the loaded LLM."""
        if self.model is None or self.tokenizer is None:
            return self._rule_based_fallback(prompt)

        max_tokens = max_new_tokens or self.max_new_tokens

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
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
        return self.model is not None


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
            )
        else:
            self.llm = LLMInterface.__new__(LLMInterface)
            self.llm.model = None
            self.llm.tokenizer = None
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

        # Step 1: Classify the question
        classification = self.classifier.classify(example.question, example.program)
        active_modules = self.classifier.get_active_modules(example.question, example.program)
        result["classification"] = {
            "scores": classification,
            "active_modules": active_modules,
            "primary_type": self.classifier.get_primary_type(example.question, example.program),
        }
        result["reasoning_trace"].append(
            f"Classification: {result['classification']['primary_type']} "
            f"(active: {active_modules})"
        )

        # Step 2: Retrieve relevant context
        retrieval_result = self.indexer.retrieve_for_question(
            example.question, example
        )
        result["retrieval"] = {
            "table_contexts": [
                {"text": r["document"][:200], "score": r["score"]}
                for r in retrieval_result.get("table_contexts", [])
            ],
            "text_contexts": [
                {"text": r["document"][:200], "score": r["score"]}
                for r in retrieval_result.get("text_contexts", [])
            ],
        }

        # Step 3: Run reasoning modules
        context_text = example.context_text

        # 3a: Numerical reasoning
        if "numerical" in active_modules:
            num_result = self.numerical_reasoner.reason(
                question=example.question,
                table=example.table,
                program=example.program,
                context=context_text,
            )
            result["numerical"] = num_result
            if num_result.get("success") and num_result.get("result") is not None:
                result["reasoning_trace"].append(
                    f"Numerical: {num_result['method']} -> {num_result['result']}"
                )

        # 3b: Temporal reasoning
        if "temporal" in active_modules:
            temp_result = self.temporal_reasoner.reason(
                question=example.question,
                table=example.table,
                context=context_text,
            )
            result["temporal"] = temp_result
            if temp_result.get("temporal_context"):
                result["reasoning_trace"].append(
                    f"Temporal: {temp_result['temporal_context']}"
                )

        # 3c: Causal reasoning
        if "causal" in active_modules:
            causal_result = self.causality_detector.reason(
                question=example.question,
                context=context_text,
                table=example.table,
            )
            result["causal"] = causal_result
            if causal_result.get("causal_relations"):
                result["reasoning_trace"].append(
                    f"Causal: {len(causal_result['causal_relations'])} relations found"
                )

        # Step 4: Determine answer
        predicted = self._aggregate_answer(result, example)
        result["predicted_answer"] = predicted

        # Step 5: Verify
        if result["numerical"].get("result") is not None:
            verification = self.numerical_reasoner.verify_result(
                result["numerical"]["result"], example.answer
            )
            result["verification"] = verification

        result["time_seconds"] = time.time() - start_time
        return result

    def _aggregate_answer(
        self, result: Dict, example: FinQAExample
    ) -> str:
        """Aggregate outputs from all modules into a final answer."""
        # Priority 1: Numerical computation result (most precise)
        if result["numerical"].get("success") and result["numerical"].get("result") is not None:
            computed = result["numerical"]["result"]
            if isinstance(computed, float):
                # Format nicely
                if abs(computed) < 1e-6:
                    return "0"
                if computed == int(computed) and abs(computed) < 1e10:
                    return str(int(computed))
                return f"{computed:.2f}" if abs(computed) < 1e6 else f"{computed:.4g}"
            return str(computed)

        # Priority 2: If we have a PoT prompt, use LLM to generate program
        if result["numerical"].get("method") == "program_of_thought" and self.llm.is_available:
            pot_prompt = result["numerical"].get("pot_prompt", "")
            if pot_prompt:
                llm_response = self.llm.generate(pot_prompt)
                code = self.numerical_reasoner.extract_code_from_response(llm_response)
                if code:
                    exec_result = self.numerical_reasoner.execute_python_program(code)
                    if exec_result["success"] and exec_result["result"] is not None:
                        return str(exec_result["result"])

        # Priority 3: LLM-based answer generation
        if self.llm.is_available:
            prompt = self._build_answer_prompt(result, example)
            answer = self.llm.generate(prompt, max_new_tokens=128)
            # Extract just the answer from LLM response
            answer = self._extract_answer_from_llm(answer)
            return answer

        # Fallback: return empty
        return ""

    def _build_answer_prompt(
        self, result: Dict, example: FinQAExample
    ) -> str:
        """Build a prompt for the LLM incorporating all reasoning outputs."""
        table_str = format_table_for_llm(example.table, max_rows=15)
        context = example.context_text[:500]

        reasoning_info = []
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
