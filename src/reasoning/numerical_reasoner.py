"""Numerical Reasoning Module using Program-of-Thought approach.

Generates executable Python programs from financial questions instead of
relying on LLM arithmetic, ensuring verifiable and deterministic results.
"""

import re
import signal
import traceback
from typing import Any, Dict, List, Optional, Tuple

from ..utils.financial_utils import (
    FINQA_OPS,
    extract_numbers_from_text,
    extract_table_value,
    format_table_for_llm,
    parse_financial_number,
)


class ProgramExecutionError(Exception):
    """Raised when a generated program fails to execute."""
    pass


class NumericalReasoner:
    """Program-of-Thought numerical reasoning for financial QA.

    Instead of relying on the LLM to do arithmetic in text generation,
    this module:
    1. Parses the FinQA DSL program
    2. Extracts values from tables
    3. Executes computations symbolically
    4. Returns verifiable, deterministic results
    """

    # FinQA DSL operations
    OPERATIONS = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else float("inf"),
        "greater": lambda a, b: "yes" if a > b else "no",
        "exp": lambda a, b: a ** b,
    }

    TABLE_OPS = {
        "table_sum": lambda vals: sum(vals),
        "table_average": lambda vals: sum(vals) / len(vals) if vals else 0,
        "table_max": lambda vals: max(vals) if vals else 0,
        "table_min": lambda vals: min(vals) if vals else 0,
    }

    def __init__(self, execution_timeout: int = 5, tolerance: float = 1e-4):
        self.execution_timeout = execution_timeout
        self.tolerance = tolerance

    def parse_finqa_program(self, program: List[str]) -> List[Dict[str, Any]]:
        """Parse FinQA DSL program into structured steps.

        FinQA programs look like:
            ['subtract(5829, 5735)', 'divide(#0, 5735)', 'multiply(#1, 100)']
        or as a single string with commas.
        """
        if not program:
            return []

        # Handle case where program is a single joined string
        if len(program) == 1 and ", " in program[0]:
            program = [p.strip() for p in program[0].split(", ")]

        steps = []
        for i, step_str in enumerate(program):
            step_str = step_str.strip()
            if not step_str:
                continue

            # Parse operation(arg1, arg2) format
            match = re.match(r"(\w+)\((.+)\)", step_str)
            if not match:
                continue

            op_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments
            args = []
            for arg in args_str.split(","):
                arg = arg.strip()
                if arg.startswith("#"):
                    # Reference to previous step result
                    ref_idx = int(arg[1:])
                    args.append({"type": "reference", "step": ref_idx})
                elif arg.startswith("const_"):
                    # Named constant
                    val_str = arg.replace("const_", "")
                    val = parse_financial_number(val_str)
                    args.append({"type": "constant", "value": val if val is not None else 0})
                elif arg.startswith("table_"):
                    # Table reference
                    args.append({"type": "table_ref", "ref": arg})
                else:
                    val = parse_financial_number(arg)
                    if val is not None:
                        args.append({"type": "number", "value": val})
                    else:
                        args.append({"type": "string", "value": arg})

            steps.append({
                "index": i,
                "operation": op_name,
                "args": args,
                "raw": step_str,
            })

        return steps

    def resolve_table_reference(
        self, ref: str, table: List[List[str]]
    ) -> Optional[float]:
        """Resolve a table reference like 'table_max(col)' or row/col index."""
        if not table:
            return None

        # Handle table_X(col_name) patterns
        for op_name, op_func in self.TABLE_OPS.items():
            if ref.startswith(op_name):
                match = re.match(rf"{op_name}\((.+)\)", ref)
                if match:
                    col_name = match.group(1).strip()
                    values = self._get_column_values(table, col_name)
                    if values:
                        return op_func(values)

        return None

    def _get_column_values(
        self, table: List[List[str]], col_identifier: str
    ) -> List[float]:
        """Extract all numeric values from a table column."""
        if not table or len(table) < 2:
            return []

        header = table[0]
        col_idx = None

        # Try exact header match
        for i, h in enumerate(header):
            if str(h).strip().lower() == col_identifier.strip().lower():
                col_idx = i
                break

        # Try partial match
        if col_idx is None:
            for i, h in enumerate(header):
                if col_identifier.strip().lower() in str(h).strip().lower():
                    col_idx = i
                    break

        # Try numeric index
        if col_idx is None:
            try:
                col_idx = int(col_identifier)
            except ValueError:
                return []

        values = []
        for row in table[1:]:
            if col_idx < len(row):
                val = parse_financial_number(str(row[col_idx]))
                if val is not None:
                    values.append(val)

        return values

    def execute_program(
        self,
        steps: List[Dict[str, Any]],
        table: List[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a parsed FinQA program step by step.

        Returns dict with:
            - result: final computed answer
            - steps: list of intermediate results
            - success: whether execution completed
            - error: error message if failed
        """
        intermediate_results = {}
        step_details = []

        try:
            for step in steps:
                op = step["operation"]
                args = step["args"]
                idx = step["index"]

                # Resolve argument values
                resolved_args = []
                for arg in args:
                    if arg["type"] == "reference":
                        ref_idx = arg["step"]
                        if ref_idx in intermediate_results:
                            resolved_args.append(intermediate_results[ref_idx])
                        else:
                            return {
                                "result": None,
                                "steps": step_details,
                                "success": False,
                                "error": f"Reference #%d not found" % ref_idx,
                            }
                    elif arg["type"] in ("number", "constant"):
                        resolved_args.append(arg["value"])
                    elif arg["type"] == "table_ref":
                        val = self.resolve_table_reference(arg["ref"], table)
                        if val is not None:
                            resolved_args.append(val)
                        else:
                            # Try parsing as number
                            val = parse_financial_number(arg["ref"])
                            resolved_args.append(val if val is not None else 0)
                    elif arg["type"] == "string":
                        # Might be a table value
                        val = parse_financial_number(arg["value"])
                        resolved_args.append(val if val is not None else 0)

                # Execute operation
                if op in self.OPERATIONS:
                    if len(resolved_args) >= 2:
                        result = self.OPERATIONS[op](resolved_args[0], resolved_args[1])
                    elif len(resolved_args) == 1:
                        result = resolved_args[0]
                    else:
                        result = 0
                elif op in self.TABLE_OPS:
                    result = self.TABLE_OPS[op](resolved_args)
                else:
                    result = resolved_args[0] if resolved_args else 0

                intermediate_results[idx] = result
                step_details.append({
                    "step": idx,
                    "operation": op,
                    "args": resolved_args,
                    "result": result,
                    "raw": step.get("raw", ""),
                })

            # Final result is the last step
            final = intermediate_results.get(len(steps) - 1, None)
            return {
                "result": final,
                "steps": step_details,
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "result": None,
                "steps": step_details,
                "success": False,
                "error": str(e),
            }

    def execute_python_program(self, code: str, timeout: int = None) -> Dict[str, Any]:
        """Execute a Python program string and return the result.

        Used for LLM-generated Program-of-Thought code.
        """
        timeout = timeout or self.execution_timeout
        namespace = {
            "math": __import__("math"),
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "float": float,
            "int": int,
            "str": str,
        }

        try:
            exec(code, namespace)
            # Look for 'answer' or 'result' variable
            result = namespace.get("answer", namespace.get("result", None))
            return {
                "result": result,
                "success": True,
                "error": None,
                "namespace": {k: v for k, v in namespace.items()
                             if not k.startswith("_") and k not in ("math",)},
            }
        except Exception as e:
            return {
                "result": None,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "namespace": {},
            }

    def generate_pot_prompt(
        self,
        question: str,
        table: List[List[str]],
        context: str = "",
    ) -> str:
        """Generate a Program-of-Thought prompt for the LLM.

        This prompt instructs the LLM to output executable Python code
        instead of performing arithmetic in natural language.
        """
        table_str = format_table_for_llm(table)

        prompt = f"""You are a financial analyst. Given the following financial data, write a Python program to answer the question.

IMPORTANT RULES:
1. Extract exact values from the table data provided
2. Perform all calculations step by step in Python
3. Store the final answer in a variable called 'answer'
4. Use only basic Python (arithmetic, comparisons)
5. Round percentages to 2 decimal places

TABLE:
{table_str}

CONTEXT:
{context[:500] if context else "No additional context."}

QUESTION: {question}

Write a Python program to compute the answer. Output ONLY the Python code, nothing else.

```python
# Step-by-step computation
"""
        return prompt

    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from an LLM response."""
        # Try to find code block
        code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        code_match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # If no code block, try to find lines that look like Python
        lines = response.strip().split("\n")
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if (
                "=" in stripped
                or stripped.startswith("#")
                or stripped.startswith("print")
                or stripped.startswith("answer")
                or stripped.startswith("result")
                or any(stripped.startswith(kw) for kw in ["if ", "for ", "def ", "import "])
            ):
                code_lines.append(line)

        return "\n".join(code_lines)

    def reason(
        self,
        question: str,
        table: List[List[str]],
        program: List[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """Main reasoning entry point.

        If a gold program (FinQA DSL) is provided, execute it directly.
        Otherwise, return a prompt for the LLM to generate a Python program.
        """
        result = {
            "question": question,
            "method": None,
            "program_steps": [],
            "result": None,
            "success": False,
            "reasoning_trace": [],
        }

        # If we have a FinQA DSL program, parse and execute it
        if program and any(p.strip() for p in program):
            steps = self.parse_finqa_program(program)
            if steps:
                exec_result = self.execute_program(steps, table)
                result["method"] = "finqa_dsl"
                result["program_steps"] = exec_result["steps"]
                result["result"] = exec_result["result"]
                result["success"] = exec_result["success"]
                result["reasoning_trace"] = [
                    f"Step {s['step']}: {s['raw']} = {s['result']}"
                    for s in exec_result["steps"]
                ]
                if not exec_result["success"]:
                    result["error"] = exec_result["error"]
                return result

        # No program available - generate prompt for LLM
        result["method"] = "program_of_thought"
        result["pot_prompt"] = self.generate_pot_prompt(question, table, context)
        result["success"] = False  # Needs LLM to generate code
        return result

    def verify_result(
        self, computed: Any, gold_answer: str, tolerance: float = None
    ) -> Dict[str, Any]:
        """Verify if a computed result matches the gold answer."""
        tol = tolerance or self.tolerance

        if computed is None:
            return {"match": False, "reason": "no_result"}

        computed_str = str(computed)
        gold_str = str(gold_answer).strip()

        # Boolean comparison
        if gold_str.lower() in ("yes", "no", "true", "false"):
            match = computed_str.lower() == gold_str.lower()
            return {"match": match, "reason": "boolean_comparison"}

        # Numerical comparison
        try:
            computed_val = float(computed)
            gold_val = float(gold_str)
            if gold_val == 0:
                match = abs(computed_val) < tol
            else:
                match = abs(computed_val - gold_val) / max(abs(gold_val), 1e-10) < tol
            return {
                "match": match,
                "computed": computed_val,
                "gold": gold_val,
                "relative_error": abs(computed_val - gold_val) / max(abs(gold_val), 1e-10),
                "reason": "numerical_comparison",
            }
        except (ValueError, TypeError):
            pass

        # String comparison
        match = computed_str.strip().lower() == gold_str.strip().lower()
        return {"match": match, "reason": "string_comparison"}
