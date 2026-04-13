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
        # Steps are delimited by '), ' — split carefully to avoid breaking args
        if len(program) == 1 and "), " in program[0]:
            parts = program[0].split("), ")
            program = []
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                if i < len(parts) - 1:
                    part = part + ")"
                program.append(part)

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

    def _get_row_values(
        self, table: List[List[str]], row_identifier: str
    ) -> List[float]:
        """Extract all numeric values from a table row identified by its label.

        FinQA table_average/table_sum/table_max/table_min operations use
        row-based lookup: table_average(row_label, none) means "average all
        numeric values across columns for the row whose first column matches
        row_label".
        """
        if not table or len(table) < 2:
            return []

        row_id = row_identifier.strip().lower()

        # Find the matching row by first-column label
        matched_row = None
        for row in table[1:]:  # skip header
            if not row:
                continue
            label = str(row[0]).strip().lower()
            if label == row_id:
                matched_row = row
                break

        # Try partial match if exact match failed
        if matched_row is None:
            for row in table[1:]:
                if not row:
                    continue
                label = str(row[0]).strip().lower()
                if row_id in label or label in row_id:
                    matched_row = row
                    break

        if matched_row is None:
            return []

        # Extract all numeric values from columns (skip the label column)
        values = []
        for cell in matched_row[1:]:
            val = parse_financial_number(str(cell))
            if val is not None:
                values.append(val)

        return values

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
                    # FinQA table ops use row-based lookup:
                    # table_average(row_label, none) -> average numeric values in that row
                    # First, try row-based lookup using the raw string arg
                    row_label = None
                    for arg in args:
                        if arg["type"] == "string" and arg["value"].lower() != "none":
                            row_label = arg["value"]
                            break
                    if row_label and table:
                        row_values = self._get_row_values(table, row_label)
                        if row_values:
                            result = self.TABLE_OPS[op](row_values)
                        else:
                            # Fallback: try column-based lookup
                            col_values = self._get_column_values(table, row_label)
                            result = self.TABLE_OPS[op](col_values) if col_values else 0
                    elif resolved_args:
                        # Use resolved numeric args directly
                        numeric_args = [a for a in resolved_args if isinstance(a, (int, float))]
                        result = self.TABLE_OPS[op](numeric_args) if numeric_args else 0
                    else:
                        result = 0
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

    def is_plausible_result(self, result: Any, question: str) -> Tuple[bool, str]:
        """Validate whether an executed numerical result is plausible."""
        if result is None:
            return False, "No result produced"

        if isinstance(result, str):
            if result.strip().lower() in {"yes", "no", "true", "false"}:
                return True, "Boolean result"
            numeric = parse_financial_number(result)
            if numeric is None:
                return False, "Non-numeric string result"
            result = numeric

        if isinstance(result, (int, float)):
            if result != result or abs(result) == float("inf"):
                return False, "Result is NaN/Inf"

            q = question.lower()
            if any(token in q for token in ["percent", "percentage", "%"]):
                if result < -1000 or result > 1000:
                    return False, "Percentage result outside plausible range"

            if any(token in q for token in ["revenue", "income", "sales"]):
                if result < 0:
                    return False, "Revenue/income appears negative"

            if abs(result) > 1e15:
                return False, "Result magnitude is implausibly large"

            return True, "Plausible numeric result"

        return False, "Unsupported result type"

    def generate_refinement_prompt(
        self,
        question: str,
        table: List[List[str]],
        context: str,
        previous_code: str,
        error_feedback: str,
    ) -> str:
        """Generate a prompt that asks the model to repair prior code."""
        table_str = format_table_for_llm(table)
        return f"""You previously wrote Python code for a financial QA task, but it failed or produced an implausible answer.

QUESTION: {question}

TABLE:
{table_str}

CONTEXT:
{context[:500] if context else "No additional context."}

PREVIOUS CODE:
```python
{previous_code}
```

FEEDBACK:
{error_feedback}

Please fix the code. Requirements:
1. Use only values from the provided table/context
2. Compute step by step
3. Store final value in variable 'answer'
4. Output ONLY Python code
"""

    def _lookup_table_value(
        self,
        table: List[List[str]],
        row_hint: str,
        col_hint: str,
    ) -> Optional[float]:
        """Look up a value from the table using row and column hints.

        Performs fuzzy matching on row labels and column headers.
        """
        if not table or len(table) < 2:
            return None

        header = [str(h).strip().lower() for h in table[0]]
        row_hint_lower = row_hint.lower().strip()
        col_hint_lower = col_hint.lower().strip()

        # Find best matching column
        col_idx = None
        best_col_score = 0
        for j, h in enumerate(header):
            # Exact match
            if col_hint_lower == h:
                col_idx = j
                break
            # Substring match
            if col_hint_lower in h or h in col_hint_lower:
                score = len(set(col_hint_lower.split()) & set(h.split()))
                if score > best_col_score:
                    best_col_score = score
                    col_idx = j

        if col_idx is None:
            return None

        # Find best matching row
        for row in table[1:]:
            if not row:
                continue
            row_label = str(row[0]).strip().lower()
            if row_hint_lower == row_label:
                if col_idx < len(row):
                    return parse_financial_number(str(row[col_idx]))
            if row_hint_lower in row_label or row_label in row_hint_lower:
                if col_idx < len(row):
                    return parse_financial_number(str(row[col_idx]))

        return None

    def _find_values_for_years(
        self,
        table: List[List[str]],
        row_keywords: List[str],
        year1: str,
        year2: str,
    ) -> Optional[Tuple[float, float]]:
        """Find two values from a table for two different years/columns.

        Searches for the best matching row using keywords, then extracts
        values from columns matching year1 and year2.
        """
        if not table or len(table) < 2:
            return None

        header = [str(h).strip().lower() for h in table[0]]

        # Find column indices for the two years
        def find_col(year_str):
            y = year_str.lower().strip()
            for j, h in enumerate(header):
                if y in h:
                    return j
            return None

        col1 = find_col(year1)
        col2 = find_col(year2)
        if col1 is None or col2 is None:
            return None

        # Score all rows against keywords
        scored_rows = []
        for row in table[1:]:
            if not row:
                continue
            if col1 >= len(row) or col2 >= len(row):
                continue
            # Must have parseable values in both columns
            v1 = parse_financial_number(str(row[col1]))
            v2 = parse_financial_number(str(row[col2]))
            if v1 is None or v2 is None:
                continue

            row_label = str(row[0]).strip().lower()
            # Substring match score
            score = sum(1 for kw in row_keywords if kw.lower() in row_label)
            # Word overlap score
            row_words = set(re.findall(r"[a-z]+", row_label))
            kw_set = set(kw.lower() for kw in row_keywords)
            word_score = len(row_words & kw_set)
            # Bonus for exact multi-word phrase match
            phrase_bonus = 0
            for kw in row_keywords:
                if len(kw) > 3 and kw.lower() in row_label:
                    phrase_bonus += 1
            total_score = score + word_score + phrase_bonus * 0.5

            scored_rows.append((total_score, row, v1, v2))

        if not scored_rows:
            return None

        # Sort by score descending
        scored_rows.sort(key=lambda x: -x[0])

        # Only use a row if it has a positive keyword match score,
        # OR if there's only one data row (no ambiguity)
        if scored_rows[0][0] > 0:
            return (scored_rows[0][2], scored_rows[0][3])

        # Fallback: if only one valid row exists, use it
        if len(scored_rows) == 1:
            return (scored_rows[0][2], scored_rows[0][3])

        # If all scores are 0, try to find a "total" or summary row as fallback
        for score, row, v1, v2 in scored_rows:
            row_label = str(row[0]).strip().lower()
            if any(term in row_label for term in ["total", "net", "gross", "revenue", "income"]):
                return (v1, v2)

        # Last resort: first valid row
        return (scored_rows[0][2], scored_rows[0][3])

    def _extract_keywords_from_question(self, question: str) -> List[str]:
        """Extract meaningful keywords from question for table matching."""
        q = question.lower()
        stopwords = {
            "what", "was", "the", "is", "are", "were", "how", "much", "many",
            "did", "does", "in", "of", "for", "from", "to", "and", "or", "a",
            "an", "on", "by", "as", "at", "be", "it", "its", "this", "that",
            "between", "total", "change", "percentage", "percent", "increase",
            "decrease", "difference", "ratio", "compared", "during",
        }
        # Remove years and numbers
        q_cleaned = re.sub(r"\b\d{4}\b", "", q)
        q_cleaned = re.sub(r"\$?\d[\d,.]*%?", "", q_cleaned)
        words = [w.strip("?.,!;:()") for w in q_cleaned.split()]
        return [w for w in words if w and w not in stopwords and len(w) > 1]

    def _extract_years_from_table_header(self, table: List[List[str]]) -> List[str]:
        """Extract year strings from table column headers."""
        if not table:
            return []
        header = table[0]
        years = []
        for h in header:
            found = re.findall(r"\b((?:19|20)\d{2})\b", str(h))
            years.extend(found)
        return sorted(set(years))

    def _find_two_values_from_question(
        self,
        question: str,
        table: List[List[str]],
        context: str = "",
    ) -> Optional[Tuple[float, float, str, str]]:
        """Extract two values from the table based on question analysis.

        Returns (val_new, val_old, description_new, description_old) or None.
        Used for change/difference/ratio type questions.
        """
        q = question.lower()
        keywords = self._extract_keywords_from_question(question)

        # Extract years from question
        years = re.findall(r"\b((?:19|20)\d{2})\b", q)

        if years and len(years) >= 2 and table:
            # Find values for the two years
            vals = self._find_values_for_years(table, keywords, years[-1], years[-2])
            if vals:
                return (vals[0], vals[1], years[-1], years[-2])
            # Try reversed
            vals = self._find_values_for_years(table, keywords, years[0], years[1])
            if vals:
                return (vals[0], vals[1], years[0], years[1])

        # If no years in question, try extracting from table headers
        if len(years) < 2 and table:
            header_years = self._extract_years_from_table_header(table)
            if len(header_years) >= 2:
                # Use the two most recent years
                y_new, y_old = header_years[-1], header_years[-2]
                vals = self._find_values_for_years(table, keywords, y_new, y_old)
                if vals:
                    return (vals[0], vals[1], y_new, y_old)

        # Try to find two rows that match different parts of the question
        if table and len(table) >= 3:
            # Look for numbers mentioned in question text that might be table values
            numbers_in_q = extract_numbers_from_text(question)
            year_set = set(int(y) for y in years) if years else set()
            non_year_nums = [n for n in numbers_in_q
                             if not (n == int(n) and int(n) in year_set)]
            if len(non_year_nums) >= 2:
                return (non_year_nums[0], non_year_nums[1], "val1", "val2")

        # Fallback: numbers from context text
        if context:
            numbers_in_context = extract_numbers_from_text(context)
            year_set = set(int(y) for y in years) if years else set()
            non_year_nums = [n for n in numbers_in_context
                             if not (n == int(n) and int(n) in year_set)]
            if len(non_year_nums) >= 2:
                return (non_year_nums[0], non_year_nums[1], "val1", "val2")

        return None

    def _find_two_row_values(
        self,
        question: str,
        table: List[List[str]],
        col_hint: str = None,
    ) -> Optional[Tuple[float, float]]:
        """Find values from two different table rows that match question keywords.

        For questions like "what percentage of total X is Y?" this finds
        the row for Y and the row for total/X, returning their values
        from the same column.
        """
        if not table or len(table) < 3:
            return None

        q = question.lower()
        header = [str(h).strip().lower() for h in table[0]]

        # Find the best column
        col_idx = 1  # Default: first data column
        if col_hint:
            for j, h in enumerate(header):
                if col_hint.lower() in h:
                    col_idx = j
                    break
        else:
            years_in_q = re.findall(r"\b((?:19|20)\d{2})\b", q)
            if years_in_q:
                for j, h in enumerate(header):
                    if years_in_q[-1] in h:
                        col_idx = j
                        break
            # Also try "total" column
            if col_idx == 1:
                for j, h in enumerate(header):
                    if "total" in h and j > 0:
                        col_idx = j
                        break

        # Score each row against the question using multi-word phrase matching
        q_words = set(re.findall(r"[a-z]+", q))
        stopwords = {"what", "was", "the", "is", "are", "were", "how", "much",
                     "did", "in", "of", "for", "from", "to", "and", "a", "as",
                     "percentage", "percent", "total", "due"}
        q_content_words = q_words - stopwords

        row_scores = []
        for row in table[1:]:
            if not row:
                continue
            row_label = str(row[0]).strip().lower()
            if col_idx >= len(row):
                continue
            val = parse_financial_number(str(row[col_idx]))
            if val is None:
                continue

            label_words = set(re.findall(r"[a-z]+", row_label))
            overlap = len(label_words & q_content_words)

            # Bonus: consecutive word match (phrase matching)
            label_lower = row_label.lower()
            for kw in q_content_words:
                if kw in label_lower:
                    overlap += 0.5

            row_scores.append((overlap, row_label, val))

        if len(row_scores) < 2:
            return None

        row_scores.sort(key=lambda x: -x[0])

        # For "percentage of total" questions: part is top match, total is "total" row
        part_val = row_scores[0][2]
        total_val = None

        # Look for a "total" row
        for score, label, val in row_scores:
            if "total" in label and abs(val) != abs(part_val):
                total_val = val
                break

        # If question says "of X", find row matching X
        of_match = re.search(r"(?:percent(?:age)?|share|portion|fraction)\s+of\s+(.+?)(?:\s+(?:is|was|are|were|that|due)|\s*\?|$)", q)
        if of_match and total_val is None:
            target = of_match.group(1).strip()
            target_words = set(re.findall(r"[a-z]+", target)) - stopwords
            best_score = 0
            for score, label, val in row_scores:
                label_words = set(re.findall(r"[a-z]+", label))
                match_score = len(label_words & target_words)
                if match_score > best_score and abs(val) != abs(part_val):
                    best_score = match_score
                    total_val = val

        if total_val is None and len(row_scores) >= 2:
            total_val = row_scores[1][2]

        if total_val is not None:
            return (part_val, total_val)

        return None

    def _find_same_row_values(
        self,
        question: str,
        table: List[List[str]],
    ) -> Optional[Tuple[float, float]]:
        """Find two values from the SAME row but different columns.

        For questions like "what is the average X per Y for Z?"
        where X and Y are different columns in the same row.
        """
        if not table or len(table) < 2:
            return None

        q = question.lower()
        q_words = set(re.findall(r"[a-z]+", q))
        stopwords = {"what", "was", "the", "is", "are", "were", "how", "much",
                     "did", "in", "of", "for", "from", "to", "and", "a", "as",
                     "percentage", "percent", "average", "per", "ratio"}
        q_content_words = q_words - stopwords

        # Find best matching row
        best_row = None
        best_score = 0
        for row in table[1:]:
            if not row or len(row) < 3:
                continue
            label = str(row[0]).strip().lower()
            label_words = set(re.findall(r"[a-z]+", label))
            score = len(label_words & q_content_words)
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is None or len(best_row) < 3:
            return None

        # Extract all numeric values from the row
        row_vals = []
        for cell in best_row[1:]:
            val = parse_financial_number(str(cell))
            if val is not None:
                row_vals.append(val)

        if len(row_vals) >= 2:
            return (row_vals[0], row_vals[1])

        return None

    def _detect_operation_from_question(self, question: str) -> str:
        """Detect the most likely arithmetic operation from question phrasing.

        Returns one of: 'pct_change', 'pct_of', 'subtract', 'add', 'divide',
        'multiply', 'greater', 'table_agg', or 'unknown'.
        """
        q = question.lower().strip()

        # Percentage of total / share / portion
        if re.search(
            r"(?:percent(?:age)?|portion|share|fraction)\s+of\b|"
            r"as a percent(?:age)?\s+of|"
            r"what (?:percent|percentage|portion|share|fraction)\s+(?:of|is|was|are|were|did)",
            q
        ):
            return "pct_of"

        # Percentage change
        if re.search(
            r"(?:percentage|percent|%)\s*(?:change|increase|decrease|growth|decline|"
            r"difference|rise|drop|reduction|improvement|gain|loss)",
            q
        ):
            return "pct_change"

        # Generic "what percentage" / "what percent" often means pct_change
        if re.search(r"what (?:is|was|were|are)\s+the\s+(?:percentage|percent)\b", q):
            return "pct_change"

        # Comparison
        if re.search(
            r"(?:greater|more|larger|higher|bigger|less|lower|smaller|exceed)\s+than|"
            r"(?:is|was|were)\s+.+?\s+(?:greater|more|less|higher|lower)\s+than",
            q
        ):
            return "greater"

        # Sum / total / combined
        if re.search(
            r"(?:total|sum|combined|aggregate|altogether|in total|cumulative)\b",
            q
        ):
            return "add"

        # Average / mean
        if re.search(r"\b(?:average|mean)\b", q):
            return "table_agg"

        # Ratio / proportion / per / divided by
        if re.search(
            r"\b(?:ratio|per\b|divided\s+by|proportion|times|multiplied)\b", q
        ):
            return "divide"

        # Change / difference / increase / decrease
        if re.search(
            r"(?:change|difference|increase|decrease|decline|growth|net change|"
            r"how much (?:did|more|less|higher|lower|greater|was)|"
            r"by how much|what was the .+? (?:increase|decrease|change|growth|decline))",
            q
        ):
            return "subtract"

        return "unknown"

    def _find_single_value_from_question(
        self,
        question: str,
        table: List[List[str]],
        context: str = "",
    ) -> Optional[float]:
        """Extract a single value from the table matching the question.

        For simple lookup questions like "what was the revenue in 2019?"
        """
        if not table or len(table) < 2:
            return None

        q = question.lower()
        keywords = self._extract_keywords_from_question(question)
        years = re.findall(r"\b((?:19|20)\d{2})\b", q)
        header = [str(h).strip().lower() for h in table[0]]

        # Find the target column (prefer year-based)
        col_idx = None
        if years:
            for y in reversed(years):
                for j, h in enumerate(header):
                    if y in h:
                        col_idx = j
                        break
                if col_idx is not None:
                    break

        if col_idx is None:
            # Use last numeric column
            for j in range(len(header) - 1, 0, -1):
                for row in table[1:]:
                    if j < len(row) and parse_financial_number(str(row[j])) is not None:
                        col_idx = j
                        break
                if col_idx is not None:
                    break

        if col_idx is None:
            return None

        # Find best matching row
        best_row = None
        best_score = -1
        for row in table[1:]:
            if not row or col_idx >= len(row):
                continue
            row_label = str(row[0]).strip().lower()
            label_words = set(re.findall(r"[a-z]+", row_label))
            kw_set = set(kw.lower() for kw in keywords)
            score = len(label_words & kw_set)
            # Bonus for longer phrase matches
            for kw in keywords:
                if kw.lower() in row_label:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None and best_score > 0:
            val = parse_financial_number(str(best_row[col_idx]))
            return val

        return None

    def induce_program(
        self,
        question: str,
        table: List[List[str]],
        context: str = "",
    ) -> Optional[List[str]]:
        """Induce a FinQA DSL program from question text and table data.

        Uses pattern matching and table lookup to generate executable programs
        without relying on gold annotations. This is the core of our
        rule-based Program-of-Thought approach.

        Strategy:
        1. Detect operation type from question phrasing
        2. Extract relevant values from the TABLE
        3. Build the computation program with proper operand ordering
        """
        q = question.lower().strip()
        keywords = self._extract_keywords_from_question(question)
        years = re.findall(r"\b((?:19|20)\d{2})\b", q)
        op_type = self._detect_operation_from_question(question)

        # --- Percentage of total ---
        if op_type == "pct_of":
            # Try row-based lookup (most common for "what % of total is X")
            row_vals = self._find_two_row_values(question, table)
            if row_vals:
                part, total = row_vals
                # Sanity: part should generally be <= total in absolute value
                if abs(part) > abs(total) and total != 0:
                    part, total = total, part
                return [f"divide({part}, {total})", "multiply(#0, const_100)"]
            # Try year-based values
            vals = self._find_two_values_from_question(question, table, context)
            if vals:
                v1, v2 = vals[0], vals[1]
                # The denominator is typically the larger value (total)
                if abs(v1) > abs(v2):
                    part, total = v2, v1
                else:
                    part, total = v1, v2
                if total == 0:
                    return None
                return [f"divide({part}, {total})", "multiply(#0, const_100)"]

        # --- Percentage change ---
        if op_type == "pct_change":
            vals = self._find_two_values_from_question(question, table, context)
            if vals:
                v_new, v_old, d_new, d_old = vals
                # For "percentage change from Y1 to Y2": (Y2 - Y1) / Y1 * 100
                # years[-1] is typically the more recent year, years[-2] the older
                if v_old == 0:
                    return None
                return [
                    f"subtract({v_new}, {v_old})",
                    f"divide(#0, {v_old})",
                    "multiply(#1, const_100)",
                ]

        # --- Change / difference ---
        if op_type == "subtract":
            vals = self._find_two_values_from_question(question, table, context)
            if vals:
                v1, v2, d1, d2 = vals
                # Check if the question implies direction
                if re.search(r"(?:decrease|decline|drop|loss|reduction|fell|lower)", q):
                    # decrease = old - new (positive result expected)
                    return [f"subtract({v2}, {v1})"]
                return [f"subtract({v1}, {v2})"]

        # --- Sum / total ---
        if op_type == "add":
            # Try to find multiple values to sum
            vals = self._find_two_values_from_question(question, table, context)
            if vals:
                v1, v2, d1, d2 = vals
                return [f"add({v1}, {v2})"]
            # Try table aggregation for "total of column"
            if table:
                col_match = re.search(r"total\s+(?:of\s+)?(.+?)(?:\s*\?|$)", q)
                if col_match:
                    target = col_match.group(1).strip()
                    return [f"table_sum({target}, none)"]

        # --- Table aggregation (average/mean) ---
        if op_type == "table_agg":
            table_agg_match = re.search(
                r"(?:average|mean)\s+(?:of|for|in|value\s+of)\s+(.+?)(?:\s*\?|$)", q
            )
            if table_agg_match and table:
                target = table_agg_match.group(1).strip()
                return [f"table_average({target}, none)"]

        # --- Division / ratio ---
        if op_type == "divide":
            # First try: two values from different years
            vals = self._find_two_values_from_question(question, table, context)
            if vals:
                v1, v2, d1, d2 = vals
                if v2 != 0:
                    return [f"divide({v1}, {v2})"]
            # Second try: two row values from same column
            row_vals = self._find_two_row_values(question, table)
            if row_vals:
                v1, v2 = row_vals
                if v2 != 0:
                    return [f"divide({v1}, {v2})"]
            # Third: same-row values
            if table:
                row_vals = self._find_same_row_values(question, table)
                if row_vals and row_vals[1] != 0:
                    return [f"divide({row_vals[0]}, {row_vals[1]})"]

        # --- Comparison ---
        if op_type == "greater":
            vals = self._find_two_values_from_question(question, table, context)
            if vals:
                v1, v2, d1, d2 = vals
                return [f"greater({v1}, {v2})"]

        # --- Fallback strategies (operation type was 'unknown') ---

        # Fallback 1: If years are available, try to find values and subtract
        effective_years = years
        if len(effective_years) < 2 and table:
            header_years = self._extract_years_from_table_header(table)
            if len(header_years) >= 2:
                effective_years = header_years

        if len(effective_years) >= 2 and table:
            vals = self._find_values_for_years(
                table, keywords, effective_years[-1], effective_years[-2]
            )
            if vals:
                v1, v2 = vals
                # Check question for clues about operation
                if re.search(r"(?:percentage|percent|%)", q):
                    if v2 != 0:
                        return [
                            f"subtract({v1}, {v2})",
                            f"divide(#0, {v2})",
                            "multiply(#1, const_100)",
                        ]
                elif re.search(r"(?:ratio|divided|per\b|times|proportion)", q):
                    if v2 != 0:
                        return [f"divide({v1}, {v2})"]
                elif re.search(r"(?:total|sum|combined|and)", q):
                    return [f"add({v1}, {v2})"]
                else:
                    # Default to subtraction (most common FinQA operation)
                    return [f"subtract({v1}, {v2})"]

        # Fallback 2: Single value lookup for simple questions
        if re.search(r"(?:what (?:is|was|were|are)|how (?:much|many))\b", q):
            single_val = self._find_single_value_from_question(question, table, context)
            if single_val is not None:
                # Check if it's a simple lookup (no computation keywords)
                if not re.search(
                    r"(?:change|difference|increase|decrease|percentage|percent|ratio|"
                    r"growth|decline|total|sum|average|compared|more|less)", q
                ):
                    return [f"add({single_val}, const_0)"]

        # Fallback 3: Numbers in question text
        numbers_in_q = extract_numbers_from_text(question)
        year_set = set(int(y) for y in years) if years else set()
        non_year_nums = [n for n in numbers_in_q
                         if not (n == int(n) and int(n) in year_set)]
        if len(non_year_nums) >= 2:
            a, b = non_year_nums[0], non_year_nums[1]
            if re.search(r"(?:percentage|percent)", q):
                if b != 0:
                    return [f"divide({a}, {b})", "multiply(#0, const_100)"]
            if re.search(r"(?:increase|more|higher|grew|gain|rose)", q):
                return [f"subtract({a}, {b})"]
            if re.search(r"(?:decrease|less|lower|fell|decline|drop|loss)", q):
                return [f"subtract({b}, {a})"]
            return [f"subtract({a}, {b})"]

        return None

    def reason(
        self,
        question: str,
        table: List[List[str]],
        program: List[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """Main reasoning entry point.

        Attempts to induce a program from question+table patterns.
        Falls back to LLM-based Program-of-Thought if no program can be induced.
        """
        result = {
            "question": question,
            "method": None,
            "program_steps": [],
            "result": None,
            "success": False,
            "reasoning_trace": [],
        }

        # Step 1: Try rule-based program induction from question + table
        induced_program = self.induce_program(question, table, context)

        if induced_program:
            steps = self.parse_finqa_program(induced_program)
            if steps:
                exec_result = self.execute_program(steps, table)
                result["method"] = "induced_program"
                result["induced_program"] = induced_program
                result["program_steps"] = exec_result["steps"]
                result["result"] = exec_result["result"]
                result["success"] = exec_result["success"]
                result["reasoning_trace"] = [
                    f"Induced: {induced_program}",
                ] + [
                    f"Step {s['step']}: {s['raw']} = {s['result']}"
                    for s in exec_result["steps"]
                ]
                if not exec_result["success"]:
                    result["error"] = exec_result["error"]
                if exec_result["success"]:
                    return result

        # Step 2: No induced program - generate prompt for LLM
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
