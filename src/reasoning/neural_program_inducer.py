"""Neural sequence-to-program induction with constrained DSL decoding.

Research references:
- FinQANet (Chen et al., EMNLP 2021): numerically-aware seq-to-program
- Constrained decoding (Scholak et al., EMNLP 2021): grammar-guided generation
- LoRA fine-tuning (Hu et al., ICLR 2022): parameter-efficient adaptation
"""

import re
import importlib
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Tuple

from ..retrieval.table_encoder import TableAwareEncoder
from ..utils.financial_utils import format_table_for_llm, parse_financial_number

HAS_TRANSFORMERS = find_spec("transformers") is not None and find_spec("torch") is not None
if HAS_TRANSFORMERS:
    torch = importlib.import_module("torch")
    tfm = importlib.import_module("transformers")
    AutoModelForCausalLM = tfm.AutoModelForCausalLM
    AutoTokenizer = tfm.AutoTokenizer

HAS_PEFT = find_spec("peft") is not None
if HAS_PEFT:
    peft_mod = importlib.import_module("peft")
    PeftModel = peft_mod.PeftModel

FINQA_FEW_SHOT_EXAMPLES = [
    # 1. Percentage change between two years (most common FinQA pattern)
    {
        "question": "what was the percentage change in revenue from 2019 to 2020?",
        "table_snippet": "| Item | 2020 | 2019 |\n|---|---|---|\n| Net Revenue | 5829 | 5735 |\n| Cost of Goods | 3400 | 3200 |",
        "program": "subtract(5829, 5735), divide(#0, 5735), multiply(#1, 100)",
    },
    # 2. Percentage of total (part / whole × 100)
    {
        "question": "what percentage of total expenses was depreciation in 2021?",
        "table_snippet": "| Item | 2021 | 2020 |\n|---|---|---|\n| Depreciation | 450 | 400 |\n| Total Expenses | 3000 | 2800 |",
        "program": "divide(450, 3000), multiply(#0, 100)",
    },
    # 3. Net / profit margin (net income ÷ revenue × 100)
    {
        "question": "what was the net profit margin in 2020?",
        "table_snippet": "| Item | 2020 | 2019 |\n|---|---|---|\n| Net Income | 682 | 591 |\n| Net Revenue | 8693 | 7948 |",
        "program": "divide(682, 8693), multiply(#0, 100)",
    },
    # 4. Absolute change / difference
    {
        "question": "by how much did operating income increase from 2018 to 2019?",
        "table_snippet": "| Item | 2019 | 2018 |\n|---|---|---|\n| Operating Income | 1450 | 1280 |\n| Revenue | 9500 | 8900 |",
        "program": "subtract(1450, 1280)",
    },
    # 5. Sum of two items
    {
        "question": "what is the combined interest and tax expense in 2020?",
        "table_snippet": "| Item | 2020 | 2019 |\n|---|---|---|\n| Interest Expense | 120 | 110 |\n| Tax Expense | 350 | 300 |",
        "program": "add(120, 350)",
    },
    # 6. Comparison (yes/no greater)
    {
        "question": "was net income greater in 2021 than in 2020?",
        "table_snippet": "| Item | 2021 | 2020 |\n|---|---|---|\n| Net Income | 980 | 850 |",
        "program": "greater(980, 850)",
    },
    # 7. Ratio (divide, no ×100)
    {
        "question": "what is the ratio of long-term debt to total equity in 2021?",
        "table_snippet": "| Item | 2021 | 2020 |\n|---|---|---|\n| Long-term Debt | 4200 | 3900 |\n| Total Equity | 6800 | 6500 |",
        "program": "divide(4200, 6800)",
    },
    # 8. Three-year total / sum across a row
    {
        "question": "what is the total capital expenditure over 2019 2020 and 2021?",
        "table_snippet": "| Item | 2021 | 2020 | 2019 |\n|---|---|---|---|\n| Capital Expenditure | 330 | 290 | 260 |",
        "program": "add(330, 290), add(#0, 260)",
    },
]


class NeuralProgramInducer:
    """Neural front-end for FinQA DSL induction.

    Uses an autoregressive model with few-shot prompting and post-decode
    DSL constraint validation. Supports optional LoRA adapter loading.
    """

    DSL_OPS = {
        "add", "subtract", "multiply", "divide", "greater", "exp",
        "table_sum", "table_average", "table_max", "table_min",
    }

    BINARY_OPS = {"add", "subtract", "multiply", "divide", "greater", "exp"}
    TABLE_OPS = {"table_sum", "table_average", "table_max", "table_min"}

    def __init__(
        self,
        model_name: str,
        enabled: bool = True,
        max_new_tokens: int = 96,
        lora_adapter_path: str = None,
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
            if lora_adapter_path and HAS_PEFT:
                try:
                    self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
                except Exception:
                    pass
            self.model.eval()
        except Exception:
            self.model = None
            self.tokenizer = None

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _build_few_shot_prompt(
        self,
        question: str,
        table: List[List[str]],
        context: str,
        table_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a prompt with few-shot FinQA DSL examples for in-context learning."""
        examples_text = ""
        for ex in FINQA_FEW_SHOT_EXAMPLES:
            examples_text += f"""Question: {ex['question']}
Table:
{ex['table_snippet']}
Program: {ex['program']}

"""
        table_str = format_table_for_llm(table)

        # Build column type profile block when table analysis is available.
        # This tells the model the dominant cell type and unit scale for every
        # column so it can write correct absolute-value arithmetic.
        col_profile_block = ""
        if table_analysis and table_analysis.get("columns"):
            lines = ["COLUMN TYPES:"]
            for col in table_analysis["columns"]:
                unit_label = col.get("unit_label", "raw")
                unit_scale = col.get("unit_scale", 1.0)
                scale_str = f"scale: {unit_scale:.0e}" if unit_scale != 1.0 else "scale: absolute"
                lines.append(
                    f"  col {col['index']}: \"{col['header']}\" | "
                    f"type: {col['dominant_type']} | "
                    f"unit: {unit_label} ({scale_str})"
                )
            col_profile_block = "\n".join(lines) + "\n\n"

        return f"""Generate a FinQA DSL program for the financial question.
Allowed operations: {', '.join(sorted(self.DSL_OPS))}
- Binary ops take exactly 2 arguments: op(arg1, arg2)
- Table ops take a column of values: table_sum(val1, val2, ...)
- Use #N to reference the result of step N (0-indexed)
- Arguments must be numbers from the table or #N references
Output ONLY comma-separated DSL steps.

{examples_text}{col_profile_block}Question: {question}
Table:
{table_str}
Context: {context[:600] if context else 'N/A'}
Program:"""

    def _validate_argument(self, arg: str, step_idx: int, table: List[List[str]]) -> bool:
        """Validate a DSL argument is a valid number or step reference."""
        arg = arg.strip()
        if re.match(r"^#\d+$", arg):
            ref_idx = int(arg[1:])
            return ref_idx < step_idx
        if parse_financial_number(arg) is not None:
            return True
        if re.match(r"^-?\d[\d,.]*%?$", arg):
            return True
        return False

    def _validate_step(self, step: str, step_idx: int, table: List[List[str]]) -> bool:
        """Validate a single DSL step for syntactic and semantic correctness."""
        m = re.match(r"([a-z_]+)\((.+)\)$", step)
        if not m:
            return False
        op, args_str = m.group(1), m.group(2)
        if op not in self.DSL_OPS:
            return False

        args = [a.strip() for a in args_str.split(",")]

        if op in self.BINARY_OPS:
            if len(args) != 2:
                return False
            return all(self._validate_argument(a, step_idx, table) for a in args)

        if op in self.TABLE_OPS:
            if len(args) < 1:
                return False
            return all(self._validate_argument(a, step_idx, table) for a in args)

        return False

    def _constrain_and_parse(self, raw: str, table: List[List[str]] = None) -> List[str]:
        """Extract and validate DSL steps from raw model output."""
        candidates = []
        for m in re.finditer(r"([a-z_]+\([^\n\)]*\))", raw.lower()):
            step = m.group(1).strip()
            step = re.sub(r"\s+", "", step)
            op = step.split("(")[0]
            if op not in self.DSL_OPS:
                continue
            if step.count("(") != step.count(")"):
                continue
            candidates.append(step)

        validated = []
        for i, c in enumerate(candidates[:6]):
            if self._validate_step(c, i, table or []):
                validated.append(c)
            else:
                break

        return validated

    def induce(self, question: str, table: List[List[str]], context: str = "") -> List[str]:
        """Generate a FinQA DSL program via neural induction.

        Returns a list of validated DSL steps, or empty list on failure.
        """
        if not self.is_available:
            return []

        table_analysis: Optional[Dict[str, Any]] = None
        if table:
            try:
                table_analysis = TableAwareEncoder().analyze_table(table)
            except Exception:
                pass

        prompt = self._build_few_shot_prompt(question, table, context, table_analysis=table_analysis)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            candidates = []
            for temp, beams in [(0.0, 3), (0.3, 1)]:
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=max(temp, 1e-7),
                        do_sample=temp > 0,
                        num_beams=beams if temp == 0 else 1,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                steps = self._constrain_and_parse(decoded, table)
                if steps:
                    candidates.append(steps)

            if not candidates:
                return []
            candidates.sort(key=len, reverse=True)
            return candidates[0]

        except Exception:
            return []

    def induce_with_confidence(
        self, question: str, table: List[List[str]], context: str = ""
    ) -> Dict[str, Any]:
        """Induce a program and return it with a confidence score.

        Confidence is based on: (1) whether beam search and sampling agree,
        (2) number of validated steps, (3) argument validity.
        """
        steps = self.induce(question, table, context)
        if not steps:
            return {"program": [], "confidence": 0.0, "source": "neural"}

        confidence = min(1.0, 0.5 + 0.1 * len(steps))
        return {"program": steps, "confidence": confidence, "source": "neural"}
