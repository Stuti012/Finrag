"""FinQA dataset loading and corpus construction.

Downloads the official FinQA benchmark (Chen et al., 2021) used throughout
the thesis, parses it into structured examples, and builds the shared,
multi-document retrieval corpus described in Section 3.4 ("Dataset
Preprocessing"): documents are chunked into overlapping windows, tagged with
fiscal-year metadata, and pooled across many filings so that retrieval must
contend with the same cross-year, cross-document ambiguity the thesis
identifies as the central challenge of financial QA.

Dataset source: https://github.com/czyssrs/FinQA
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

FINQA_BASE_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/"
FINQA_FILES = {"train": "train.json", "dev": "dev.json", "test": "test.json"}

# Matches fiscal years like 2009, 1999; deliberately narrow (1980-2039) to
# avoid false positives on large financial figures such as "2009000".
_YEAR_RE = re.compile(r"(?<!\d)(19[8-9]\d|20[0-3]\d)(?!\d)")
_QUARTER_RE = re.compile(r"\bq([1-4])\s*[,\s]*\s*((?:19|20)\d{2})\b", re.IGNORECASE)
_FY_RE = re.compile(r"\bfy\s?[' ]?((?:19|20)\d{2}|\d{2})\b", re.IGNORECASE)


def download_finqa_dataset(save_dir: str = "./finqa_data") -> Dict[str, str]:
    """Download the three official FinQA splits if not already present locally."""
    os.makedirs(save_dir, exist_ok=True)
    paths = {}
    for split, filename in FINQA_FILES.items():
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            url = FINQA_BASE_URL + filename
            print(f"Downloading {split} split from {url} ...")
            urllib.request.urlretrieve(url, filepath)
        paths[split] = filepath
    return paths


def _parse_program_string(prog_str: str) -> List[str]:
    """FinQA programs are serialized as e.g. 'divide(637, const_5), multiply(#0, const_100)'."""
    if not prog_str or not prog_str.strip():
        return []
    parts = prog_str.split("), ")
    steps = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i < len(parts) - 1:
            part = part + ")"
        steps.append(part)
    return steps


def extract_fiscal_years(text: str) -> List[int]:
    """Extract explicit fiscal year tokens (plain years, 'FY2021', 'Q3 2021') from text."""
    years = set()
    for m in _YEAR_RE.finditer(text):
        years.add(int(m.group(1)))
    for m in _FY_RE.finditer(text):
        y = m.group(1)
        y = int(y) if len(y) == 4 else 2000 + int(y)
        years.add(y)
    for m in _QUARTER_RE.finditer(text):
        years.add(int(m.group(2)))
    return sorted(years)


@dataclass
class FinQAExample:
    """A single FinQA question, with its source table/text and gold annotations."""

    id: str
    question: str
    table: List[List[str]]
    pre_text: List[str]
    post_text: List[str]
    program: List[str]
    answer: str
    gold_evidence: List[str] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return " ".join(self.pre_text + self.post_text)

    @property
    def table_text(self) -> str:
        if not self.table:
            return ""
        return "\n".join(" | ".join(str(c) for c in row) for row in self.table)

    @property
    def fiscal_years(self) -> List[int]:
        return extract_fiscal_years(self.question + " " + self.table_text + " " + self.context_text)

    @property
    def question_years(self) -> List[int]:
        """Fiscal years referenced explicitly in the question only (used for temporal eval)."""
        return extract_fiscal_years(self.question)


def load_finqa_split(filepath: str, max_examples: Optional[int] = None) -> List[FinQAExample]:
    with open(filepath, "r") as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        if max_examples and len(examples) >= max_examples:
            break
        qa = item.get("qa", {})
        gold_inds = qa.get("gold_inds", {})
        example = FinQAExample(
            id=item.get("id", f"finqa_{len(examples)}"),
            question=qa.get("question", ""),
            table=item.get("table", []),
            pre_text=item.get("pre_text", []),
            post_text=item.get("post_text", []),
            program=_parse_program_string(qa.get("program", ""))
            if isinstance(qa.get("program", ""), str)
            else qa.get("program", []),
            answer=str(qa.get("exe_ans", qa.get("answer", ""))),
            gold_evidence=list(gold_inds.values()) if isinstance(gold_inds, dict) else [],
        )
        examples.append(example)
    return examples


def load_finqa_dataset(
    data_dir: str = "./finqa_data",
    download: bool = True,
    max_train: Optional[int] = None,
    max_dev: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Dict[str, List[FinQAExample]]:
    if download:
        paths = download_finqa_dataset(data_dir)
    else:
        paths = {s: os.path.join(data_dir, f) for s, f in FINQA_FILES.items()}

    dataset = {}
    limits = {"train": max_train, "dev": max_dev, "test": max_test}
    for split, path in paths.items():
        if os.path.exists(path):
            dataset[split] = load_finqa_split(path, limits.get(split))
    return dataset


# ---------------------------------------------------------------------------
# Corpus construction: chunk every document (table + narrative text) into the
# shared, multi-document retrieval index described in Section 3.4.
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    """A single structured (metric, year, value) triple pulled from a table row."""

    metric: str
    year: Optional[int]
    value: float


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    kind: str  # "table" or "text"
    fiscal_years: List[int]
    facts: List[Fact] = field(default_factory=list)


def _word_chunks(words: List[str], size: int, overlap: int) -> List[List[str]]:
    if not words:
        return []
    step = max(1, size - overlap)
    return [words[i : i + size] for i in range(0, len(words), step) if words[i : i + size]]


def parse_financial_number(text: str) -> Optional[float]:
    """Parse a number from financial text: currency symbols, commas, %, and
    accounting-style parenthetical negatives, e.g. "(1,234)" -> -1234.0.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip().replace("$", "").replace(",", "").replace(" ", "")
    if not cleaned or cleaned in {"-", "--", "n/a", "na"}:
        return None
    is_negative = cleaned.startswith("(") and cleaned.endswith(")")
    if is_negative:
        cleaned = cleaned[1:-1]
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if is_percent:
        value = value / 100.0
    return -value if is_negative else value


_NUMBER_TOKEN_RE = re.compile(r"-?\$?\(?[\d,]+\.?\d*%?\)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def _extract_text_facts(text: str, fallback_years: List[int], max_facts: int = 30) -> List["Fact"]:
    """Lightweight sentence-level numeric fact extraction for narrative chunks.

    Table cells give clean, reliably-labeled operands (see `_extract_table_facts`),
    but text chunks vastly outnumber table chunks in the corpus and are what most
    often survives retrieval + filtering -- if they carry no facts at all, the
    symbolic reasoning engine has nothing to compute over whenever the one
    relevant table chunk didn't make it through. This extracts a (noisy, but
    real) operand for each number mentioned near a financial term, labeled with
    the few words preceding it and the fiscal year mentioned in that sentence
    (or the chunk's only unambiguous year, if there is exactly one).
    """
    facts: List[Fact] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence_years = extract_fiscal_years(sentence)
        year = sentence_years[0] if len(sentence_years) == 1 else (fallback_years[0] if len(fallback_years) == 1 else None)
        for m in _NUMBER_TOKEN_RE.finditer(sentence):
            raw = m.group(0)
            # A bare, unformatted 4-digit token in the fiscal-year range (no $,
            # no comma, no %, no parens) is almost certainly a year reference,
            # not a financial figure -- e.g. "...as compared to 2014..." must
            # not be extracted as a value of 2014.0.
            if re.fullmatch(r"(19[8-9]\d|20[0-3]\d)", raw):
                continue
            value = parse_financial_number(raw)
            if value is None or abs(value) < 1:
                continue  # skip degenerate matches (bare "1", stray footnote markers, etc.)
            label_words = sentence[: m.start()].split()[-6:]
            label = " ".join(label_words).strip(" ,;:-()").lower()
            if not label:
                continue
            facts.append(Fact(metric=label, year=year, value=value))
            if len(facts) >= max_facts:
                return facts
    return facts


def _extract_table_facts(table: List[List[str]]) -> tuple:
    """Turn a raw FinQA table into (readable_text, [Fact, ...]).

    FinQA tables use the first row as column headers (usually blank or a
    label for column 0, followed by fiscal periods) and each subsequent row
    as one financial-statement line item. We pair each numeric cell with its
    column-header period and row-label metric name.
    """
    if not table or len(table) < 2:
        return "", []
    header = table[0]
    statements: List[str] = []
    facts: List[Fact] = []
    for row in table[1:]:
        if not row:
            continue
        metric = str(row[0]).strip().lower() or "value"
        for col_idx in range(1, len(row)):
            period_label = header[col_idx] if col_idx < len(header) else ""
            value = parse_financial_number(row[col_idx])
            if value is None:
                continue
            years = extract_fiscal_years(str(period_label))
            year = years[0] if years else None
            period_desc = str(period_label).strip() or "unspecified period"
            statements.append(f"{metric} ({period_desc}): {row[col_idx]}")
            facts.append(Fact(metric=metric, year=year, value=value))
    return "; ".join(statements), facts


def build_corpus(
    examples: List[FinQAExample],
    chunk_size_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
) -> List[Chunk]:
    """Build the shared retrieval corpus from a list of FinQA documents.

    Each example contributes one linearized table chunk (row-by-row
    flattening with the header repeated per row, per Section 3.4) plus one or
    more overlapping narrative-text chunks. Every chunk is deduplicated by
    source document so identical filings appearing across multiple QA pairs
    are only indexed once.
    """
    seen_docs = set()
    chunks: List[Chunk] = []

    for ex in examples:
        # FinQA ids are typically "<doc>-<question_index>"; group by document.
        doc_id = ex.id.rsplit("-", 1)[0] if "-" in ex.id else ex.id
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)

        # A short, document-identifying caption (typically names the company,
        # e.g. "entergy corporation and subsidiaries management's financial
        # discussion..."). Raw FinQA tables carry no such context on their
        # own -- without this, a table chunk is retrieval-invisible to any
        # entity-name matching (Section 3.5.1) and can't be distinguished
        # from another company's identically-shaped table.
        doc_label = " ".join((ex.pre_text[0] if ex.pre_text else "").split()[:20]) or doc_id

        if ex.table:
            table_text, facts = _extract_table_facts(ex.table)
            if table_text.strip():
                labeled_table_text = f"{doc_label} -- {table_text}" if doc_label else table_text
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::table",
                        doc_id=doc_id,
                        text=labeled_table_text,
                        kind="table",
                        fiscal_years=extract_fiscal_years(labeled_table_text + " " + " ".join(ex.table[0])),
                        facts=facts,
                    )
                )

        narrative = " ".join(ex.pre_text + ex.post_text)
        words = narrative.split()
        for i, wchunk in enumerate(_word_chunks(words, chunk_size_tokens, chunk_overlap_tokens)):
            text = " ".join(wchunk)
            if not text.strip():
                continue
            chunk_years = extract_fiscal_years(text)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::text::{i}",
                    doc_id=doc_id,
                    text=text,
                    kind="text",
                    fiscal_years=chunk_years,
                    facts=_extract_text_facts(text, chunk_years),
                )
            )

    return chunks
