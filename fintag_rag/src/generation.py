"""Answer Generation Module (thesis Section 3.5.6, Algorithm 6).

a = LLM(q', R(q'), a_num): the language model only synthesizes fluent text
around evidence that has already been retrieved, temporally validated, and
(for numeric questions) computed deterministically -- it is explicitly told
not to perform its own arithmetic, which is what eliminates the numerical
hallucination failure mode discussed throughout the thesis.

Three backends, tried in order of preference:
  1. OpenAI API           (config.use_openai=True, matches the thesis's GPT-3.5-Turbo)
  2. Local open model     (default; no gated access required)
  3. Template fallback     (deterministic, no model download -- keeps the
                            pipeline runnable even with no GPU/network)
"""

from __future__ import annotations

from typing import List, Optional

from .causal import CausalRelation
from .config import FinTAGRAGConfig
from .ontology import EnrichedQuery
from .retrieval import RetrievedChunk
from .symbolic import SymbolicResult

_SYSTEM_PROMPT = (
    "You are a financial question-answering assistant. You are given retrieved, "
    "temporally-verified evidence and, when applicable, a numeric result that has "
    "already been computed deterministically. Do NOT perform your own arithmetic and "
    "do NOT change the given numeric result -- restate it exactly and explain it using "
    "the evidence. Answer in 1-3 concise sentences suitable for a financial analyst."
)

# Used only by the retrieval-only / LLM-only baselines (Section 4.3), where the
# thesis's comparison intentionally leaves arithmetic to the language model
# itself, so its accuracy can be contrasted against FinTAG-RAG's symbolic engine.
_BASELINE_SYSTEM_PROMPT = (
    "You are a financial question-answering assistant. Using only the evidence "
    "provided, perform any arithmetic needed and answer the question. "
    "End your response with a final line of the exact form 'ANSWER: <value>' "
    "where <value> is the final numeric or textual answer, with no extra words."
)


def extract_final_number(text: str) -> Optional[float]:
    """Pull the numeric value out of an 'ANSWER: <value>' line (or, failing
    that, the last number-like token in the text)."""
    import re

    from .data import parse_financial_number

    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    candidate = m.group(1).strip() if m else text.strip()
    number_match = re.search(r"-?\$?\(?[\d,]+\.?\d*%?\)?", candidate)
    if number_match:
        value = parse_financial_number(number_match.group(0))
        if value is not None:
            return value
    # Fall back to scanning the whole text for the last numeric token.
    all_numbers = re.findall(r"-?\$?\(?[\d,]+\.?\d*%?\)?", text)
    for token in reversed(all_numbers):
        value = parse_financial_number(token)
        if value is not None:
            return value
    return None


def _format_evidence(evidence_texts: List[str]) -> str:
    if not evidence_texts:
        return "(no evidence retrieved)"
    return "\n".join(f"- {t}" for t in evidence_texts[:5])


def _build_user_prompt(
    enriched_query: EnrichedQuery,
    evidence_texts: List[str],
    symbolic_result: Optional[SymbolicResult],
    causal_relations: Optional[List[CausalRelation]],
) -> str:
    parts = [f"Question: {enriched_query.original}", "", "Evidence:", _format_evidence(evidence_texts)]
    if symbolic_result is not None:
        if symbolic_result.success:
            parts += ["", f"Verified computed result ({symbolic_result.operation}): {symbolic_result.value}"]
        else:
            parts += ["", f"Note: no verified numeric result could be computed ({symbolic_result.error})."]
    if causal_relations:
        parts += ["", "Detected causal relations:"]
        parts += [f"- {r.cause} -> {r.effect} (confidence {r.confidence:.2f})" for r in causal_relations[:3]]
    parts += ["", "Answer:"]
    return "\n".join(parts)


def _template_fallback(
    symbolic_result: Optional[SymbolicResult],
    evidence_texts: List[str],
    causal_relations: Optional[List[CausalRelation]],
) -> str:
    sentences = []
    if symbolic_result is not None and symbolic_result.success:
        op_phrases = {
            "percentage_change": f"a {symbolic_result.value:.2f}% change",
            "difference": f"a difference of {symbolic_result.value:,.2f}",
            "ratio": f"a ratio of {symbolic_result.value:.4f}",
            "sum": f"a total of {symbolic_result.value:,.2f}",
            "average": f"an average of {symbolic_result.value:,.2f}",
            "lookup": f"{symbolic_result.value:,.2f}",
        }
        sentences.append(f"The verified computation gives {op_phrases.get(symbolic_result.operation, symbolic_result.value)}.")
    elif symbolic_result is not None and not symbolic_result.success:
        sentences.append(f"A verified numeric answer could not be computed ({symbolic_result.error}).")
    if causal_relations:
        top = causal_relations[0]
        sentences.append(f"The retrieved evidence attributes this to: {top.cause}.")
    if not sentences:
        sentences.append(evidence_texts[0] if evidence_texts else "No answer could be generated from the retrieved evidence.")
    return " ".join(sentences)


class AnswerGenerator:
    def __init__(self, config: FinTAGRAGConfig = None):
        self.config = config or FinTAGRAGConfig()
        self._model = None
        self._tokenizer = None

    def _load_local_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        return self._model, self._tokenizer

    def _generate_local(self, user_prompt: str) -> str:
        import torch

        model, tokenizer = self._load_local_model()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=self.config.llm_max_new_tokens,
                do_sample=self.config.llm_temperature > 0,
                temperature=max(self.config.llm_temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][input_ids.shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_openai(self, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.config.openai_model,
            temperature=self.config.llm_temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _generate_local_raw(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        model, tokenizer = self._load_local_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=self.config.llm_max_new_tokens,
                do_sample=self.config.llm_temperature > 0,
                temperature=max(self.config.llm_temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][input_ids.shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_openai_raw(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.config.openai_model,
            temperature=self.config.llm_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def generate_llm_only(self, question: str, evidence_texts: List[str]) -> str:
        """Baseline path (Section 4.3): the LLM must perform its own arithmetic
        over the given (unfiltered / uncomputed) evidence -- no symbolic engine."""
        user_prompt = f"Question: {question}\n\nEvidence:\n{_format_evidence(evidence_texts)}\n\nAnswer:"
        if self.config.use_openai:
            try:
                return self._generate_openai_raw(_BASELINE_SYSTEM_PROMPT, user_prompt)
            except Exception:
                pass
        try:
            return self._generate_local_raw(_BASELINE_SYSTEM_PROMPT, user_prompt)
        except Exception:
            return "ANSWER: " + (evidence_texts[0] if evidence_texts else "unknown")

    def generate(
        self,
        enriched_query: EnrichedQuery,
        evidence_chunks: List[RetrievedChunk],
        symbolic_result: Optional[SymbolicResult] = None,
        causal_relations: Optional[List[CausalRelation]] = None,
    ) -> str:
        evidence_texts = [c.chunk.text for c in evidence_chunks]
        user_prompt = _build_user_prompt(enriched_query, evidence_texts, symbolic_result, causal_relations)

        if self.config.use_openai:
            try:
                return self._generate_openai(user_prompt)
            except Exception:
                pass  # fall through to local / template

        try:
            return self._generate_local(user_prompt)
        except Exception:
            return _template_fallback(symbolic_result, evidence_texts, causal_relations)
