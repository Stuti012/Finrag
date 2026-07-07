from src.data import Chunk
from src.retrieval import HybridRetriever


def _make_retriever(texts):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.chunks = [
        Chunk(chunk_id=f"c{i}", doc_id=f"doc{i}", text=t, kind="text", fiscal_years=[]) for i, t in enumerate(texts)
    ]
    return retriever


def test_entity_match_boost_reorders_candidates():
    retriever = _make_retriever(
        [
            "entergy arkansas , inc . and subsidiaries results",
            "entergy corporation and subsidiaries results",
            "entergy new orleans , inc . results",
        ]
    )
    # doc 0 (wrong subsidiary) scores highest before the boost
    scored = [(0, 0.9), (1, 0.5), (2, 0.4)]
    boosted = retriever._boost_entity_matches(scored, "entergy corporation", increment=1.0)
    assert boosted[0][0] == 1  # the parent company chunk now ranks first


def test_no_entity_phrase_is_a_no_op():
    retriever = _make_retriever(["a", "b"])
    scored = [(0, 0.9), (1, 0.5)]
    assert retriever._boost_entity_matches(scored, None, increment=1.0) == scored


def test_boost_respects_cap():
    retriever = _make_retriever(["entergy corporation report"])
    scored = [(0, 0.95)]
    boosted = retriever._boost_entity_matches(scored, "entergy corporation", increment=0.5, cap=1.0)
    assert boosted[0][1] == 1.0
