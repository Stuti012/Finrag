from src.retrieval import HybridRetriever


def test_reciprocal_rank_fusion_prefers_consensus():
    # doc 0 ranks well in both dense and sparse -> should come out on top
    dense = [(0, 0.9), (1, 0.5), (2, 0.1)]
    sparse = [(0, 5.0), (2, 4.0), (1, 1.0)]
    fused = HybridRetriever.reciprocal_rank_fusion(dense, sparse, k=60)
    fused_ids = [idx for idx, _ in fused]
    assert fused_ids[0] == 0  # ranked #1 in both lists


def test_reciprocal_rank_fusion_handles_disjoint_lists():
    dense = [(0, 0.9)]
    sparse = [(1, 5.0)]
    fused = HybridRetriever.reciprocal_rank_fusion(dense, sparse, k=60)
    assert {idx for idx, _ in fused} == {0, 1}
