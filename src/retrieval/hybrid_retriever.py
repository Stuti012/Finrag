"""Hybrid retrieval system combining dense embeddings + BM25 for tables and text."""

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class BM25Retriever:
    """BM25 sparse retrieval for financial text."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0
        self.token_docs: List[List[str]] = []
        self.n_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\w+", text.lower())

    def fit(self, documents: List[str]):
        """Index documents for BM25."""
        self.documents = documents
        self.n_docs = len(documents)
        self.token_docs = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.token_docs]
        self.avg_dl = sum(self.doc_lengths) / max(self.n_docs, 1)

        # Compute document frequencies
        self.doc_freqs = defaultdict(int)
        for tokens in self.token_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        doc_tokens = self.token_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        tf_map = Counter(doc_tokens)
        score = 0.0

        for qt in query_tokens:
            if qt not in self.doc_freqs:
                continue
            df = self.doc_freqs[qt]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf = tf_map.get(qt, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for top-k documents matching query."""
        query_tokens = self._tokenize(query)
        scores = [(i, self._score(query_tokens, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DenseRetriever:
    """Dense retrieval using sentence embeddings + FAISS."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384
        self.index = None
        self.documents: List[str] = []

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.encoder = SentenceTransformer(model_name)
                self.dimension = self.encoder.get_sentence_embedding_dimension()
            except Exception:
                self.encoder = None
        else:
            self.encoder = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.encoder is not None:
            embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.astype("float32")
        else:
            # Fallback: TF-IDF style random projection
            np.random.seed(42)
            embeddings = np.random.randn(len(texts), self.dimension).astype("float32")
            # Make somewhat content-aware using character hashing
            for i, text in enumerate(texts):
                for j, ch in enumerate(text[:self.dimension]):
                    embeddings[i][j % self.dimension] += ord(ch) / 128.0
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            embeddings = embeddings / norms
            return embeddings

    def fit(self, documents: List[str]):
        """Index documents using FAISS."""
        self.documents = documents
        embeddings = self._encode(documents)

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dimension)
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            # Fallback: store embeddings for brute-force search
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index = embeddings / np.maximum(norms, 1e-10)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for top-k similar documents."""
        query_emb = self._encode([query])

        if HAS_FAISS and isinstance(self.index, faiss.Index):
            faiss.normalize_L2(query_emb)
            scores, indices = self.index.search(query_emb, min(top_k, len(self.documents)))
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((int(idx), float(score)))
            return results
        else:
            # Brute-force cosine similarity
            query_norm = query_emb / np.maximum(np.linalg.norm(query_emb, axis=1, keepdims=True), 1e-10)
            similarities = np.dot(self.index, query_norm.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(int(i), float(similarities[i])) for i in top_indices]


class HybridRetriever:
    """Combines dense and BM25 retrieval with dynamic weighting."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
    ):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.dense_retriever = DenseRetriever(embedding_model)
        self.bm25_retriever = BM25Retriever()
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []

    def index_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Index documents in both dense and BM25 retrievers."""
        self.documents = documents
        self.doc_metadata = metadata or [{} for _ in documents]
        self.dense_retriever.fit(documents)
        self.bm25_retriever.fit(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and BM25 scores.

        Args:
            query: Search query.
            top_k: Number of results to return.
            filter_type: Optional filter for document type ('table' or 'text').

        Returns:
            List of dicts with 'document', 'score', 'metadata', 'index'.
        """
        n_candidates = min(top_k * 3, len(self.documents))

        dense_results = self.dense_retriever.search(query, n_candidates)
        bm25_results = self.bm25_retriever.search(query, n_candidates)

        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return {}
            max_score = max(s for _, s in results) if results else 1.0
            min_score = min(s for _, s in results) if results else 0.0
            rng = max_score - min_score if max_score != min_score else 1.0
            return {idx: (score - min_score) / rng for idx, score in results}

        dense_scores = normalize_scores(dense_results)
        bm25_scores = normalize_scores(bm25_results)

        # Combine scores
        all_indices = set(dense_scores.keys()) | set(bm25_scores.keys())
        combined = {}
        for idx in all_indices:
            d_score = dense_scores.get(idx, 0.0)
            b_score = bm25_scores.get(idx, 0.0)
            combined[idx] = self.dense_weight * d_score + self.bm25_weight * b_score

        # Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            if filter_type and self.doc_metadata[idx].get("type") != filter_type:
                continue
            results.append({
                "document": self.documents[idx],
                "score": score,
                "metadata": self.doc_metadata[idx],
                "index": idx,
            })
            if len(results) >= top_k:
                break

        return results


class FinancialDocumentIndexer:
    """Indexes FinQA examples for retrieval, separating tables and text."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def index_examples(self, examples: List[Any]):
        """Index all examples' tables and text passages."""
        documents = []
        metadata = []

        for ex in examples:
            # Index table as a document
            if ex.table:
                table_text = ex.table_text
                documents.append(table_text)
                metadata.append({
                    "type": "table",
                    "example_id": ex.id,
                    "source": "table",
                })

            # Index pre-text paragraphs
            for i, text in enumerate(ex.pre_text):
                if text.strip():
                    documents.append(text.strip())
                    metadata.append({
                        "type": "text",
                        "example_id": ex.id,
                        "source": "pre_text",
                        "index": i,
                    })

            # Index post-text paragraphs
            for i, text in enumerate(ex.post_text):
                if text.strip():
                    documents.append(text.strip())
                    metadata.append({
                        "type": "text",
                        "example_id": ex.id,
                        "source": "post_text",
                        "index": i,
                    })

        self.retriever.index_documents(documents, metadata)
        return len(documents)

    def retrieve_for_question(
        self,
        question: str,
        example: Any,
        top_k_table: int = 3,
        top_k_text: int = 5,
    ) -> Dict[str, List[Dict]]:
        """Retrieve relevant tables and text for a given question + example context."""
        # Build a focused index for this specific example
        docs = []
        meta = []

        if example.table:
            table_text = example.table_text
            docs.append(table_text)
            meta.append({"type": "table", "source": "table"})

        for i, text in enumerate(example.pre_text):
            if text.strip():
                docs.append(text.strip())
                meta.append({"type": "text", "source": "pre_text", "index": i})

        for i, text in enumerate(example.post_text):
            if text.strip():
                docs.append(text.strip())
                meta.append({"type": "text", "source": "post_text", "index": i})

        # Create a local retriever for this example
        local_retriever = HybridRetriever(
            embedding_model=self.retriever.dense_retriever.model_name,
            bm25_weight=self.retriever.bm25_weight,
            dense_weight=self.retriever.dense_weight,
        )
        # Share encoder to avoid reloading
        local_retriever.dense_retriever.encoder = self.retriever.dense_retriever.encoder
        local_retriever.dense_retriever.dimension = self.retriever.dense_retriever.dimension
        local_retriever.index_documents(docs, meta)

        table_results = local_retriever.search(question, top_k=top_k_table, filter_type="table")
        text_results = local_retriever.search(question, top_k=top_k_text, filter_type="text")

        return {
            "table_contexts": table_results,
            "text_contexts": text_results,
        }
