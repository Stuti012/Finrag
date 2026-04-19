"""Tests for TableAwareEncoder: cell classification, linearization strategies,
table analysis, question-table relevance, and retrieval integration."""

import pytest
from src.retrieval.table_encoder import TableAwareEncoder, CellType
from src.retrieval.hybrid_retriever import FinancialDocumentIndexer, HybridRetriever


SAMPLE_TABLE = [
    ["Item", "2021", "2022", "2023"],
    ["Revenue", "$1,000", "$1,200", "$1,500"],
    ["Cost of Goods Sold", "$400", "$480", "$600"],
    ["Net Income", "$200", "$250", "$320"],
    ["EPS", "2.00", "2.50", "3.20"],
    ["Gross Margin %", "60%", "60%", "60%"],
]


class TestCellClassification:
    def setup_method(self):
        self.enc = TableAwareEncoder()

    def test_empty_cells(self):
        for val in ("", "-", "—", "n/a", "N/A"):
            assert self.enc.classify_cell(val) == CellType.EMPTY

    def test_currency_cells(self):
        assert self.enc.classify_cell("$1,000") == CellType.CURRENCY
        assert self.enc.classify_cell("€500") == CellType.CURRENCY
        assert self.enc.classify_cell("$1.5M") == CellType.CURRENCY

    def test_percentage_cells(self):
        assert self.enc.classify_cell("60%") == CellType.PERCENTAGE
        assert self.enc.classify_cell("12.5 %") == CellType.PERCENTAGE

    def test_numeric_cells(self):
        assert self.enc.classify_cell("1000") == CellType.NUMERIC
        assert self.enc.classify_cell("2.50") == CellType.NUMERIC
        assert self.enc.classify_cell("(123)") == CellType.NUMERIC

    def test_date_cells(self):
        assert self.enc.classify_cell("Q3 2023") == CellType.DATE
        assert self.enc.classify_cell("FY2022") == CellType.DATE

    def test_text_cells(self):
        assert self.enc.classify_cell("Revenue") == CellType.TEXT
        assert self.enc.classify_cell("Cost of Goods Sold") == CellType.TEXT

    def test_header_cells_via_analyze(self):
        analysis = self.enc.analyze_table(SAMPLE_TABLE)
        for ct in analysis["cell_types"][0]:
            assert ct == CellType.HEADER


class TestTableAnalysis:
    def setup_method(self):
        self.enc = TableAwareEncoder()

    def test_basic_analysis(self):
        analysis = self.enc.analyze_table(SAMPLE_TABLE)
        assert analysis["rows"] == 5
        assert analysis["cols"] == 4
        assert analysis["header"] == ["Item", "2021", "2022", "2023"]

    def test_column_profiles(self):
        analysis = self.enc.analyze_table(SAMPLE_TABLE)
        cols = analysis["columns"]
        assert cols[0]["is_label"] is True
        assert cols[1].get("is_temporal") is True
        assert cols[1]["num_count"] > 0

    def test_row_labels(self):
        analysis = self.enc.analyze_table(SAMPLE_TABLE)
        assert "Revenue" in analysis["row_labels"]
        assert "Net Income" in analysis["row_labels"]

    def test_numeric_column_stats(self):
        analysis = self.enc.analyze_table(SAMPLE_TABLE)
        year_col = analysis["columns"][3]
        assert "min" in year_col
        assert "max" in year_col
        assert "mean" in year_col

    def test_empty_table(self):
        analysis = self.enc.analyze_table([])
        assert analysis["rows"] == 0


class TestLinearization:
    def setup_method(self):
        self.enc = TableAwareEncoder()

    def test_tagged_linearization(self):
        result = self.enc.linearize_tagged(SAMPLE_TABLE)
        assert "[H]" in result
        assert "[R1]" in result
        assert "Revenue" in result

    def test_tagged_contains_type_tags(self):
        result = self.enc.linearize_tagged(SAMPLE_TABLE)
        assert "[C]" in result or "[N]" in result or "[P]" in result or "[T]" in result

    def test_html_linearization(self):
        result = self.enc.linearize_html(SAMPLE_TABLE)
        assert "<table>" in result
        assert "<th>" in result
        assert "<td>" in result
        assert "Revenue" in result
        assert "</table>" in result

    def test_structured_linearization(self):
        result = self.enc.linearize_structured(SAMPLE_TABLE)
        assert 'row "Revenue"' in result
        assert 'col "2023"' in result

    def test_embedding_linearization(self):
        result = self.enc.linearize_for_embedding(SAMPLE_TABLE)
        assert "Table columns:" in result
        assert "Row items:" in result

    def test_empty_table_linearization(self):
        assert self.enc.linearize_tagged([]) == ""
        assert self.enc.linearize_html([]) == "<table></table>"
        assert self.enc.linearize_structured([]) == ""
        assert self.enc.linearize_for_embedding([]) == ""


class TestQuestionTableRelevance:
    def setup_method(self):
        self.enc = TableAwareEncoder()

    def test_revenue_question_finds_revenue_row(self):
        result = self.enc.question_table_relevance(
            "What was the revenue in 2023?", SAMPLE_TABLE
        )
        relevant_rows = result["relevant_rows"]
        assert len(relevant_rows) > 0
        labels = [r["label"] for r in relevant_rows]
        assert any("revenue" in l.lower() for l in labels)

    def test_year_question_finds_year_column(self):
        result = self.enc.question_table_relevance(
            "What was the revenue in 2023?", SAMPLE_TABLE
        )
        relevant_cols = result["relevant_cols"]
        assert len(relevant_cols) > 0
        headers = [c["header"] for c in relevant_cols]
        assert "2023" in headers

    def test_eps_question(self):
        result = self.enc.question_table_relevance(
            "Calculate the EPS growth rate from 2021 to 2023", SAMPLE_TABLE
        )
        relevant_rows = result["relevant_rows"]
        labels = [r["label"].lower() for r in relevant_rows]
        assert any("eps" in l for l in labels)

    def test_table_analysis_in_result(self):
        result = self.enc.question_table_relevance(
            "What is the net income?", SAMPLE_TABLE
        )
        assert "table_analysis" in result
        assert result["table_analysis"]["rows"] == 5
        assert result["table_analysis"]["cols"] == 4


class TestSubtableExtraction:
    def setup_method(self):
        self.enc = TableAwareEncoder()

    def test_extracts_relevant_subset(self):
        subtable = self.enc.extract_relevant_subtable(
            "What was the revenue in 2023?", SAMPLE_TABLE
        )
        assert len(subtable) > 0
        assert subtable[0][0] == "Item"

    def test_retains_header(self):
        subtable = self.enc.extract_relevant_subtable(
            "Calculate EPS growth", SAMPLE_TABLE
        )
        assert subtable[0] == SAMPLE_TABLE[0] or subtable[0][0] == "Item"

    def test_max_rows_respected(self):
        large_table = [["Item", "2023"]] + [[f"Row{i}", str(i)] for i in range(50)]
        subtable = self.enc.extract_relevant_subtable(
            "What is Row5?", large_table, max_rows=5
        )
        assert len(subtable) <= 6  # 5 data rows + header


class TestEncodeForRetrieval:
    def setup_method(self):
        self.enc = TableAwareEncoder()

    def test_multi_strategy_produces_multiple_docs(self):
        docs = self.enc.encode_for_retrieval(SAMPLE_TABLE, strategy="multi")
        assert len(docs) > 2
        types = {d["type"] for d in docs}
        assert "table_summary" in types
        assert "table_structured" in types
        assert "table_row" in types

    def test_summary_strategy(self):
        docs = self.enc.encode_for_retrieval(SAMPLE_TABLE, strategy="summary")
        assert len(docs) == 1
        assert docs[0]["type"] == "table_summary"

    def test_rows_strategy(self):
        docs = self.enc.encode_for_retrieval(SAMPLE_TABLE, strategy="rows")
        assert len(docs) == 5  # 5 data rows
        assert all(d["type"] == "table_row" for d in docs)

    def test_empty_table(self):
        docs = self.enc.encode_for_retrieval([], strategy="multi")
        assert docs == []


class TestIndexerIntegration:
    def setup_method(self):
        self.retriever = HybridRetriever()
        self.indexer = FinancialDocumentIndexer(self.retriever, use_table_encoder=True)

    def test_table_encoder_initialized(self):
        assert self.indexer.table_encoder is not None

    def test_encode_table_produces_documents(self):
        docs, meta = self.indexer._encode_table(SAMPLE_TABLE, "test_id")
        assert len(docs) > 1
        assert all(m["type"] == "table" for m in meta)
        encodings = {m.get("encoding") for m in meta}
        assert "table_summary" in encodings

    def test_encode_table_fallback_without_encoder(self):
        indexer = FinancialDocumentIndexer(self.retriever, use_table_encoder=False)
        docs, meta = indexer._encode_table(SAMPLE_TABLE, "test_id")
        assert len(docs) == 1
        assert meta[0]["encoding"] == "pipe_delimited"
