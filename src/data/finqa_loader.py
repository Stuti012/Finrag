"""FinQA Dataset loader and preprocessing."""

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FinQAExample:
    """A single FinQA example with table, text, question, and program."""
    id: str
    question: str
    table: List[List[str]]
    pre_text: List[str]
    post_text: List[str]
    program: List[str]
    answer: str
    table_header: List[str] = field(default_factory=list)
    gold_evidence: List[str] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        """Combine pre_text and post_text into a single context string."""
        return " ".join(self.pre_text + self.post_text)

    @property
    def table_text(self) -> str:
        """Linearize the table into a readable string."""
        if not self.table:
            return ""
        lines = []
        for row in self.table:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)

    @property
    def program_str(self) -> str:
        """Convert program list to readable string."""
        return " ".join(self.program) if self.program else ""

    def get_table_as_dict(self) -> List[Dict[str, str]]:
        """Convert table to list of row dicts keyed by header."""
        if not self.table or len(self.table) < 2:
            return []
        header = self.table[0]
        rows = []
        for row in self.table[1:]:
            row_dict = {}
            for i, cell in enumerate(row):
                key = header[i] if i < len(header) else f"col_{i}"
                row_dict[key] = cell
            rows.append(row_dict)
        return rows


def parse_number(text: str) -> Optional[float]:
    """Parse a number from financial text, handling currency, commas, percentages."""
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip()
    cleaned = cleaned.replace("$", "").replace(",", "").replace(" ", "")
    # Handle parenthetical negatives: (123) -> -123
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
        if is_percent:
            value = value / 100.0
        return value
    except ValueError:
        return None


def download_finqa_dataset(save_dir: str = "./finqa_data") -> Dict[str, str]:
    """Download FinQA dataset from the official GitHub repository."""
    os.makedirs(save_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/"
    files = {
        "train": "train.json",
        "dev": "dev.json",
        "test": "test.json",
    }

    paths = {}
    for split, filename in files.items():
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            url = base_url + filename
            print(f"Downloading {split} set from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  Saved to {filepath}")
            except Exception as e:
                print(f"  Warning: Could not download {split}: {e}")
                continue
        paths[split] = filepath

    return paths


def load_finqa_split(filepath: str, max_examples: Optional[int] = None) -> List[FinQAExample]:
    """Load a single FinQA JSON split file and return structured examples."""
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    examples = []
    for item in raw_data:
        if max_examples and len(examples) >= max_examples:
            break

        qa_entry = item.get("qa", {})
        table = item.get("table", [])

        example = FinQAExample(
            id=item.get("id", f"finqa_{len(examples)}"),
            question=qa_entry.get("question", ""),
            table=table,
            pre_text=item.get("pre_text", []),
            post_text=item.get("post_text", []),
            program=qa_entry.get("program", "").split(", ") if isinstance(qa_entry.get("program", ""), str) else qa_entry.get("program", []),
            answer=str(qa_entry.get("exe_ans", qa_entry.get("answer", ""))),
            table_header=table[0] if table else [],
            gold_evidence=qa_entry.get("gold_inds", {}).values() if isinstance(qa_entry.get("gold_inds"), dict) else [],
        )
        # Convert gold_evidence generator to list
        example.gold_evidence = list(example.gold_evidence)
        examples.append(example)

    return examples


def load_finqa_dataset(
    data_dir: str = "./finqa_data",
    download: bool = True,
    max_train: Optional[int] = None,
    max_dev: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Dict[str, List[FinQAExample]]:
    """Load the full FinQA dataset, downloading if necessary."""
    if download:
        paths = download_finqa_dataset(data_dir)
    else:
        paths = {
            split: os.path.join(data_dir, f"{split}.json")
            for split in ["train", "dev", "test"]
        }

    dataset = {}
    limits = {"train": max_train, "dev": max_dev, "test": max_test}

    for split, filepath in paths.items():
        if os.path.exists(filepath):
            print(f"Loading {split} split from {filepath}...")
            dataset[split] = load_finqa_split(filepath, max_examples=limits.get(split))
            print(f"  Loaded {len(dataset[split])} examples")
        else:
            print(f"  Warning: {filepath} not found, skipping {split}")

    return dataset


def classify_question_type(question: str, program: List[str] = None) -> Dict[str, bool]:
    """Classify a question into reasoning types needed.

    Returns dict with boolean flags for:
        - numerical: requires mathematical computation
        - temporal: involves time-based reasoning
        - causal: involves cause-effect reasoning
        - table_lookup: requires table data extraction
    """
    q_lower = question.lower()

    # Numerical reasoning indicators
    numerical_keywords = [
        "percentage", "percent", "ratio", "growth", "increase", "decrease",
        "change", "difference", "total", "sum", "average", "how much",
        "how many", "what was the", "calculate", "compute", "net",
        "margin", "rate", "proportion",
    ]
    numerical_ops = ["add", "subtract", "multiply", "divide", "greater", "exp", "table_"]

    is_numerical = any(kw in q_lower for kw in numerical_keywords)
    if program:
        prog_str = " ".join(program).lower()
        is_numerical = is_numerical or any(op in prog_str for op in numerical_ops)

    # Temporal reasoning indicators
    temporal_keywords = [
        "year", "quarter", "month", "period", "fiscal", "from", "to",
        "between", "prior", "previous", "subsequent", "trend", "over time",
        "growth rate", "yoy", "year-over-year", "q1", "q2", "q3", "q4",
        "2018", "2019", "2020", "2021", "2022", "2023",
        "highest", "lowest", "most recent",
    ]
    is_temporal = any(kw in q_lower for kw in temporal_keywords)

    # Causal reasoning indicators
    causal_keywords = [
        "why", "cause", "because", "due to", "result of", "led to",
        "driven by", "attributed to", "impact", "effect", "consequence",
        "reason", "factor", "explain", "what caused", "what led",
    ]
    is_causal = any(kw in q_lower for kw in causal_keywords)

    # Table lookup indicators
    table_keywords = [
        "table", "row", "column", "value of", "what is the", "what was the",
        "list", "which", "how many items",
    ]
    is_table = any(kw in q_lower for kw in table_keywords)

    return {
        "numerical": is_numerical,
        "temporal": is_temporal,
        "causal": is_causal,
        "table_lookup": is_table,
    }
