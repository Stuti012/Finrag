"""Table-aware encoding for financial QA (TAPAS/TaPEx-inspired).

Implements structured table understanding without requiring pretrained
table-specific models. Key ideas from:
- TAPAS (Herzig et al., ACL 2020): positional embeddings for row/column
- TaPEx (Liu et al., ICLR 2022): table linearization as executable format
- GRAPPA (Yu et al., ICLR 2021): pre-training on synthetic table ops

Architecture:
1. Cell-type classification (numeric, text, date, percentage, currency, empty)
2. Positional encoding (row index, column index, header membership)
3. Multiple linearization strategies (markdown, HTML, tagged, structured)
4. Cell-level attention for question-table relevance scoring
5. Numeric-aware column aggregation features
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.financial_utils import parse_financial_number


class CellType(Enum):
    EMPTY = "empty"
    TEXT = "text"
    NUMERIC = "numeric"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    DATE = "date"
    HEADER = "header"


class TableAwareEncoder:
    """Encodes financial tables with structural, positional, and type information.

    Produces enriched table representations that preserve the relational
    structure lost by naive linearization. Supports multiple serialization
    formats and cell-level question-table relevance scoring.
    """

    CURRENCY_RE = re.compile(r"^\s*[$\u20ac\u00a3\u00a5]")
    PERCENTAGE_RE = re.compile(r"\d+\.?\d*\s*%")
    DATE_RE = re.compile(r"\b(19|20)\d{2}\b|Q[1-4]|FY\d{2,4}")
    NUMERIC_RE = re.compile(r"^[\s($]*-?\d[\d,]*\.?\d*\s*[)%]?\s*$")

    def __init__(self):
        self._cell_cache: Dict[str, CellType] = {}

    def classify_cell(self, value: str) -> CellType:
        if value in self._cell_cache:
            return self._cell_cache[value]

        v = str(value).strip()
        if not v or v in ("-", "—", "n/a", "N/A", "nan", "None"):
            result = CellType.EMPTY
        elif self.CURRENCY_RE.search(v):
            result = CellType.CURRENCY
        elif self.PERCENTAGE_RE.search(v):
            result = CellType.PERCENTAGE
        elif self.DATE_RE.search(v) and not self.NUMERIC_RE.match(v.replace(",", "")):
            result = CellType.DATE
        elif self.NUMERIC_RE.match(v.replace(",", "").replace("$", "").replace("(", "").replace(")", "")):
            result = CellType.NUMERIC
        else:
            result = CellType.TEXT

        self._cell_cache[value] = result
        return result

    def analyze_table(self, table: List[List[str]]) -> Dict[str, Any]:
        """Produce a rich structural analysis of the table.

        Returns cell types, column profiles, row profiles, and
        summary statistics used for encoding and relevance scoring.
        """
        if not table or not table[0]:
            return {"rows": 0, "cols": 0, "cells": [], "columns": [], "header": []}

        header = [str(h).strip() for h in table[0]]
        n_cols = len(header)
        n_rows = len(table) - 1

        cell_types = []
        cell_values = []
        for ri, row in enumerate(table):
            row_types = []
            row_values = []
            for ci, cell in enumerate(row):
                cell_str = str(cell).strip()
                if ri == 0:
                    ct = CellType.HEADER
                else:
                    ct = self.classify_cell(cell_str)
                row_types.append(ct)

                parsed = parse_financial_number(cell_str) if ct in (
                    CellType.NUMERIC, CellType.PERCENTAGE, CellType.CURRENCY
                ) else None
                row_values.append(parsed)
            cell_types.append(row_types)
            cell_values.append(row_values)

        columns = []
        for ci in range(n_cols):
            col_vals = []
            col_types = set()
            for ri in range(1, len(table)):
                if ci < len(cell_types[ri]):
                    col_types.add(cell_types[ri][ci])
                    if cell_values[ri][ci] is not None:
                        col_vals.append(cell_values[ri][ci])

            is_temporal = bool(self.DATE_RE.search(header[ci])) if ci < len(header) else False
            dominant_type = CellType.NUMERIC if CellType.NUMERIC in col_types else (
                CellType.TEXT if CellType.TEXT in col_types else CellType.EMPTY
            )

            col_profile: Dict[str, Any] = {
                "index": ci,
                "header": header[ci] if ci < len(header) else "",
                "dominant_type": dominant_type.value,
                "is_temporal": is_temporal,
                "is_label": ci == 0 and dominant_type == CellType.TEXT,
                "num_count": len(col_vals),
            }
            if col_vals:
                col_profile["min"] = min(col_vals)
                col_profile["max"] = max(col_vals)
                col_profile["mean"] = sum(col_vals) / len(col_vals)
                col_profile["sum"] = sum(col_vals)

            columns.append(col_profile)

        row_labels = []
        for ri in range(1, len(table)):
            label = str(table[ri][0]).strip() if table[ri] else ""
            row_labels.append(label)

        return {
            "rows": n_rows,
            "cols": n_cols,
            "header": header,
            "cell_types": cell_types,
            "cell_values": cell_values,
            "columns": columns,
            "row_labels": row_labels,
        }

    def linearize_tagged(self, table: List[List[str]]) -> str:
        """Tagged linearization with cell-type and position annotations.

        Produces: [H] col0 | col1 | ... [R0] val0 | val1 | ...
        Each cell is annotated with its type for downstream parsing.
        """
        if not table:
            return ""
        analysis = self.analyze_table(table)
        header = analysis["header"]

        parts = ["[H] " + " | ".join(header)]
        for ri in range(1, len(table)):
            row = table[ri]
            cells = []
            for ci, cell in enumerate(row):
                cell_str = str(cell).strip()
                ct = analysis["cell_types"][ri][ci] if ci < len(analysis["cell_types"][ri]) else CellType.TEXT
                type_tag = ct.value[0].upper()
                cells.append(f"[{type_tag}]{cell_str}")
            parts.append(f"[R{ri}] " + " | ".join(cells))
        return "\n".join(parts)

    def linearize_html(self, table: List[List[str]]) -> str:
        """HTML-style linearization preserving structure for LLMs.

        Many LLMs trained on web data understand HTML table markup better
        than custom delimiters (TaPEx observation).
        """
        if not table:
            return "<table></table>"

        lines = ["<table>", "<thead><tr>"]
        for h in table[0]:
            lines.append(f"<th>{str(h).strip()}</th>")
        lines.append("</tr></thead>")
        lines.append("<tbody>")
        for row in table[1:]:
            lines.append("<tr>")
            for cell in row:
                lines.append(f"<td>{str(cell).strip()}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")
        return "".join(lines)

    def linearize_structured(self, table: List[List[str]]) -> str:
        """Structured linearization with explicit row-column references.

        Format: row "Label" : col "Header1" = val1 , col "Header2" = val2 ; ...
        This makes cell addresses explicit for program synthesis.
        """
        if not table or len(table) < 2:
            return ""

        header = [str(h).strip() for h in table[0]]
        lines = []
        for row in table[1:]:
            label = str(row[0]).strip() if row else ""
            pairs = []
            for ci in range(1, len(row)):
                col_name = header[ci] if ci < len(header) else f"col{ci}"
                val = str(row[ci]).strip()
                if val and val not in ("-", "—", "n/a"):
                    pairs.append(f'col "{col_name}" = {val}')
            if pairs:
                lines.append(f'row "{label}" : {" , ".join(pairs)}')
        return " ; ".join(lines)

    def linearize_for_embedding(self, table: List[List[str]]) -> str:
        """Embedding-optimized linearization that front-loads key information.

        Produces a compact representation suitable for dense retrieval:
        Headers first, then row labels, then a summary of numeric ranges.
        """
        if not table:
            return ""
        analysis = self.analyze_table(table)
        header = analysis["header"]

        parts = [f"Table columns: {', '.join(header)}"]

        labels = analysis.get("row_labels", [])
        if labels:
            parts.append(f"Row items: {', '.join(labels[:15])}")

        temporal_cols = [c for c in analysis["columns"] if c.get("is_temporal")]
        if temporal_cols:
            periods = [c["header"] for c in temporal_cols]
            parts.append(f"Time periods: {', '.join(periods)}")

        numeric_cols = [c for c in analysis["columns"] if c.get("num_count", 0) > 0 and not c.get("is_label")]
        for col in numeric_cols[:4]:
            if "min" in col and "max" in col:
                parts.append(
                    f"{col['header']}: range [{col['min']:.2f}, {col['max']:.2f}], "
                    f"mean={col['mean']:.2f}"
                )

        return " | ".join(parts)

    def compute_cell_embeddings(
        self, table: List[List[str]], encoder=None
    ) -> Optional[np.ndarray]:
        """Compute cell-level embeddings with positional encoding.

        Each cell embedding = encode(cell_text) + positional_encoding(row, col).
        Positional encoding uses sinusoidal functions (Vaswani et al., 2017).
        """
        if encoder is None or not table:
            return None

        analysis = self.analyze_table(table)
        n_rows = len(table)
        n_cols = analysis["cols"]

        cell_texts = []
        positions = []
        for ri, row in enumerate(table):
            for ci, cell in enumerate(row):
                header_label = analysis["header"][ci] if ci < len(analysis["header"]) else ""
                if ri == 0:
                    cell_texts.append(str(cell).strip())
                else:
                    cell_texts.append(f"{header_label}: {str(cell).strip()}")
                positions.append((ri, ci))

        if not cell_texts:
            return None

        try:
            base_embs = encoder.encode(cell_texts, show_progress_bar=False)
        except Exception:
            return None

        dim = base_embs.shape[1]
        pos_enc = np.zeros_like(base_embs)
        for idx, (ri, ci) in enumerate(positions):
            for d in range(dim):
                if d % 4 == 0:
                    pos_enc[idx, d] = np.sin(ri / (10000 ** (d / dim)))
                elif d % 4 == 1:
                    pos_enc[idx, d] = np.cos(ri / (10000 ** (d / dim)))
                elif d % 4 == 2:
                    pos_enc[idx, d] = np.sin(ci / (10000 ** (d / dim)))
                else:
                    pos_enc[idx, d] = np.cos(ci / (10000 ** (d / dim)))

        combined = base_embs + 0.1 * pos_enc

        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = combined / np.maximum(norms, 1e-10)

        return combined.reshape(n_rows, n_cols, dim)

    def question_table_relevance(
        self,
        question: str,
        table: List[List[str]],
        encoder=None,
    ) -> Dict[str, Any]:
        """Compute cell-level relevance scores for question-table matching.

        Returns per-cell attention weights and identifies the most relevant
        rows and columns for the given question.
        """
        analysis = self.analyze_table(table)
        if not analysis["rows"]:
            return {"relevant_rows": [], "relevant_cols": [], "cell_scores": []}

        q_lower = question.lower()
        q_words = set(re.findall(r"[a-z]{3,}", q_lower))
        q_numbers = set(re.findall(r"\d+\.?\d*", question))

        row_scores = []
        for ri in range(1, len(table)):
            row = table[ri]
            score = 0.0
            row_text = " ".join(str(c).strip().lower() for c in row)
            row_words = set(re.findall(r"[a-z]{3,}", row_text))
            word_overlap = len(q_words & row_words)
            score += word_overlap * 0.3

            row_nums = set(re.findall(r"\d+\.?\d*", row_text))
            num_overlap = len(q_numbers & row_nums)
            score += num_overlap * 0.5

            label = str(row[0]).strip().lower() if row else ""
            if any(w in label for w in q_words if len(w) > 3):
                score += 0.4

            row_scores.append({"row_index": ri, "label": label, "score": score})

        col_scores = []
        for ci, col_info in enumerate(analysis["columns"]):
            header = col_info["header"].lower()
            header_words = set(re.findall(r"[a-z]{3,}", header))
            score = len(q_words & header_words) * 0.4

            if col_info.get("is_temporal") and re.search(r"\b(year|quarter|trend|change|growth)\b", q_lower):
                score += 0.3

            if any(y in header for y in re.findall(r"(19|20)\d{2}", question)):
                score += 0.5

            col_scores.append({"col_index": ci, "header": col_info["header"], "score": score})

        cell_scores = []
        if encoder is not None:
            cell_embs = self.compute_cell_embeddings(table, encoder)
            if cell_embs is not None:
                try:
                    q_emb = encoder.encode([question], show_progress_bar=False)[0]
                    q_norm = np.linalg.norm(q_emb) + 1e-10
                    n_rows, n_cols, dim = cell_embs.shape
                    for ri in range(n_rows):
                        for ci in range(n_cols):
                            c_emb = cell_embs[ri, ci]
                            cos_sim = float(np.dot(q_emb, c_emb) / (q_norm * (np.linalg.norm(c_emb) + 1e-10)))
                            cell_scores.append({
                                "row": ri, "col": ci,
                                "score": (cos_sim + 1.0) / 2.0,
                            })
                except Exception:
                    pass

        row_scores.sort(key=lambda x: x["score"], reverse=True)
        col_scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "relevant_rows": [r for r in row_scores if r["score"] > 0][:5],
            "relevant_cols": [c for c in col_scores if c["score"] > 0][:5],
            "cell_scores": cell_scores[:20] if cell_scores else [],
            "table_analysis": {
                "rows": analysis["rows"],
                "cols": analysis["cols"],
                "header": analysis["header"],
                "column_profiles": analysis["columns"],
            },
        }

    def extract_relevant_subtable(
        self,
        question: str,
        table: List[List[str]],
        max_rows: int = 10,
        max_cols: int = 8,
    ) -> List[List[str]]:
        """Extract a focused subtable containing only question-relevant rows/columns.

        Reduces noise for downstream reasoning by pruning irrelevant cells.
        Always retains the label column (col 0) and header row.
        """
        if not table or len(table) < 2:
            return table or []

        relevance = self.question_table_relevance(question, table)

        relevant_row_indices = {0}
        for r in relevance.get("relevant_rows", []):
            relevant_row_indices.add(r["row_index"])

        if len(relevant_row_indices) <= 1:
            relevant_row_indices = set(range(min(len(table), max_rows + 1)))

        relevant_col_indices = {0}
        for c in relevance.get("relevant_cols", []):
            relevant_col_indices.add(c["col_index"])

        if len(relevant_col_indices) <= 1:
            relevant_col_indices = set(range(min(len(table[0]) if table else 0, max_cols)))

        sorted_rows = sorted(relevant_row_indices)[:max_rows + 1]
        sorted_cols = sorted(relevant_col_indices)[:max_cols]

        subtable = []
        for ri in sorted_rows:
            if ri < len(table):
                row = [table[ri][ci] if ci < len(table[ri]) else "" for ci in sorted_cols]
                subtable.append(row)

        return subtable

    def encode_for_retrieval(
        self,
        table: List[List[str]],
        strategy: str = "multi",
    ) -> List[Dict[str, str]]:
        """Produce multiple encoded representations of a table for indexing.

        Strategy 'multi' creates several complementary views:
        1. Embedding-optimized summary (for dense retrieval)
        2. Structured row-column format (for BM25 keyword matching)
        3. Per-row documents (for fine-grained retrieval)
        """
        if not table:
            return []

        docs = []

        if strategy in ("multi", "summary"):
            docs.append({
                "text": self.linearize_for_embedding(table),
                "type": "table_summary",
            })

        if strategy in ("multi", "structured"):
            docs.append({
                "text": self.linearize_structured(table),
                "type": "table_structured",
            })

        if strategy in ("multi", "rows"):
            header = [str(h).strip() for h in table[0]] if table else []
            for ri, row in enumerate(table[1:], start=1):
                label = str(row[0]).strip() if row else ""
                pairs = []
                for ci in range(len(row)):
                    col_name = header[ci] if ci < len(header) else f"col{ci}"
                    val = str(row[ci]).strip()
                    if val and val not in ("-", "—"):
                        pairs.append(f"{col_name}: {val}")
                if pairs:
                    docs.append({
                        "text": f"{label} - {', '.join(pairs)}",
                        "type": "table_row",
                        "row_index": ri,
                    })

        return docs
