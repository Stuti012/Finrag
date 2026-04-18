"""Financial utility functions for parsing, formatting, and computation."""

import re
from typing import Any, Dict, List, Optional, Tuple


# FinQA DSL operation definitions
FINQA_OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b if b != 0 else float("inf"),
    "greater": lambda a, b: a > b,
    "exp": lambda a, b: a ** b,
    "table_sum": lambda values: sum(values),
    "table_average": lambda values: sum(values) / len(values) if values else 0,
    "table_max": lambda values: max(values) if values else 0,
    "table_min": lambda values: min(values) if values else 0,
}


def parse_financial_number(text: str) -> Optional[float]:
    """Parse a financial number from text, handling various formats."""
    if not isinstance(text, str):
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    cleaned = text.strip()
    if not cleaned:
        return None

    # Remove currency symbols
    cleaned = re.sub(r"[$\u20ac\u00a3\u00a5]", "", cleaned).strip()
    cleaned = cleaned.replace(",", "")

    # Handle FinQA redundant parenthetical format: "-13 ( 13 )" or "- 13 ( 13 )"
    # The parenthetical restates the negative value; strip it
    paren_match = re.match(r"^(-?\s*[\d.]+)\s*\(\s*[\d.]+\s*\)$", cleaned)
    if paren_match:
        cleaned = paren_match.group(1).replace(" ", "")
    else:
        cleaned = cleaned.replace(" ", "")

    # Handle parenthetical negatives: (123) -> -123
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    # Handle percentage
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]

    # Handle suffixes (millions, billions, etc.)
    multiplier = 1.0
    suffix_map = {
        "k": 1e3, "K": 1e3,
        "m": 1e6, "M": 1e6,
        "b": 1e9, "B": 1e9,
        "t": 1e12, "T": 1e12,
    }
    if cleaned and cleaned[-1] in suffix_map:
        multiplier = suffix_map[cleaned[-1]]
        cleaned = cleaned[:-1]

    try:
        value = float(cleaned) * multiplier
        return value
    except ValueError:
        return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    if not answer:
        return ""
    ans = str(answer).strip().lower()
    ans = ans.replace("%", "").replace("$", "").replace(",", "")
    ans = ans.strip()
    return ans


def answers_match(predicted: str, gold: str, tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches gold answer within tolerance.

    Also handles percentage/decimal equivalence: if one answer is 100x the other
    (e.g., predicted=25.0 vs gold=0.25), they are considered equivalent.
    """
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact string match
    if pred_norm == gold_norm:
        return True

    # Try numerical comparison
    try:
        pred_val = float(pred_norm)
        gold_val = float(gold_norm)
        if gold_val == 0:
            return abs(pred_val) < tolerance
        rel_err = abs(pred_val - gold_val) / max(abs(gold_val), 1e-10)
        if rel_err < tolerance:
            return True

        # Check percentage/decimal equivalence (off by factor of 100)
        if gold_val != 0 and pred_val != 0:
            ratio = pred_val / gold_val
            # pred is 100x gold (e.g., predicted 25 when gold is 0.25)
            if abs(ratio - 100.0) / 100.0 < tolerance:
                return True
            # pred is 1/100 of gold (e.g., predicted 0.25 when gold is 25)
            if abs(ratio - 0.01) / 0.01 < tolerance:
                return True

        # Check sign-insensitive match (some FinQA answers differ only in sign)
        if abs(abs(pred_val) - abs(gold_val)) / max(abs(gold_val), 1e-10) < tolerance:
            return True

        return False
    except (ValueError, TypeError):
        pass

    # Boolean comparison
    bool_map = {"yes": "true", "no": "false", "true": "true", "false": "false"}
    if pred_norm in bool_map and gold_norm in bool_map:
        return bool_map[pred_norm] == bool_map[gold_norm]

    return False


def extract_years_from_text(text: str) -> List[int]:
    """Extract all 4-digit years from text."""
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return sorted(set(int(y) for y in years))


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract all numbers from text."""
    # Match numbers with optional decimal, commas, currency, percent
    pattern = r"[-+]?\$?\d[\d,]*\.?\d*%?"
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        val = parse_financial_number(m)
        if val is not None:
            numbers.append(val)
    return numbers


def format_table_for_llm(table: List[List[str]], max_rows: int = 30) -> str:
    """Format a table into a readable string for LLM consumption."""
    if not table:
        return "No table data available."

    lines = []
    header = table[0]
    lines.append("| " + " | ".join(str(h) for h in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for row in table[1:max_rows + 1]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(str(c) for c in padded[:len(header)]) + " |")

    if len(table) > max_rows + 1:
        lines.append(f"... ({len(table) - max_rows - 1} more rows)")

    return "\n".join(lines)


def extract_table_value(
    table: List[List[str]], row_label: str, col_label: str
) -> Optional[float]:
    """Extract a numeric value from a table given row and column labels."""
    if not table or len(table) < 2:
        return None

    header = [str(h).strip().lower() for h in table[0]]

    # Find column index
    col_idx = None
    for i, h in enumerate(header):
        if col_label.lower().strip() in h:
            col_idx = i
            break

    if col_idx is None:
        return None

    # Find row — prefer exact match, then best substring match by specificity
    target = row_label.lower().strip()
    best_row = None
    best_specificity = -1

    for row in table[1:]:
        if not row or col_idx >= len(row):
            continue
        cell = str(row[0]).strip().lower()
        if cell == target:
            return parse_financial_number(str(row[col_idx]))
        if target in cell:
            specificity = len(target) / max(len(cell), 1)
            if specificity > best_specificity:
                best_specificity = specificity
                best_row = row

    if best_row is not None:
        return parse_financial_number(str(best_row[col_idx]))

    return None
