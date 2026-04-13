"""Neural sequence-to-program induction with constrained DSL decoding."""

import re
import importlib
from importlib.util import find_spec
from typing import List

from ..utils.financial_utils import format_table_for_llm

HAS_TRANSFORMERS = find_spec("transformers") is not None and find_spec("torch") is not None
if HAS_TRANSFORMERS:
    torch = importlib.import_module("torch")
    tfm = importlib.import_module("transformers")
    AutoModelForCausalLM = tfm.AutoModelForCausalLM
    AutoTokenizer = tfm.AutoTokenizer


class NeuralProgramInducer:
    """Optional neural front-end for FinQA DSL induction.

    Uses an autoregressive model and post-decode DSL constraints.
    """

    DSL_OPS = {
        "add", "subtract", "multiply", "divide", "greater", "exp",
        "table_sum", "table_average", "table_max", "table_min",
    }

    def __init__(
        self,
        model_name: str,
        enabled: bool = True,
        max_new_tokens: int = 96,
    ):
        self.enabled = enabled and HAS_TRANSFORMERS
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

        if not self.enabled:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if hasattr(torch, "float16") else None,
            )
            self.model.eval()
        except Exception:
            self.model = None
            self.tokenizer = None

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _prompt(self, question: str, table: List[List[str]], context: str) -> str:
        table_str = format_table_for_llm(table)
        return f"""Generate a FinQA DSL program for the financial question.
Allowed operations: {', '.join(sorted(self.DSL_OPS))}
Output only comma-separated DSL steps like: subtract(10, 5), divide(#0, 5)

Question: {question}
Table:
{table_str}
Context: {context[:600] if context else 'N/A'}
Program:"""

    def _constrain_and_parse(self, raw: str) -> List[str]:
        candidates = []
        for m in re.finditer(r"([a-z_]+\([^\n\)]*\))", raw.lower()):
            step = m.group(1).strip()
            op = step.split("(")[0]
            if op not in self.DSL_OPS:
                continue
            if step.count("(") != step.count(")"):
                continue
            candidates.append(step)

        cleaned = []
        for c in candidates:
            # conservative arg cleanup
            c = re.sub(r"\s+", "", c)
            cleaned.append(c)
        return cleaned[:6]

    def induce(self, question: str, table: List[List[str]], context: str = "") -> List[str]:
        if not self.is_available:
            return []

        prompt = self._prompt(question, table, context)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    num_beams=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return self._constrain_and_parse(decoded)
        except Exception:
            return []
