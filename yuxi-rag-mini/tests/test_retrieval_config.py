import pytest
from app.rag.schemas import MilvusRetrievalConfig


def test_default_values():
    config = MilvusRetrievalConfig()
    assert config.search_mode == "hybrid"
    assert config.final_top_k == 10
    assert config.similarity_threshold == 0.0
    assert config.bm25_top_k == 20
    assert config.vector_weight == 0.7
    assert config.bm25_weight == 0.3
    assert config.bm25_drop_ratio_search == 0.2
    assert config.include_distances is True
    assert config.use_reranker is False
    assert config.reranker_model == ""
    assert config.recall_top_k == 20
    assert config.use_graph_retrieval is False
    assert config.graph_entity_top_k == 5
    assert config.graph_triple_top_k == 5
    assert config.graph_max_nodes == 100
    assert config.graph_top_k == 10
    assert config.graph_weight == 0.1
    assert config.ppr_damping == 0.85


def test_custom_values():
    config = MilvusRetrievalConfig(search_mode="vector", final_top_k=5, vector_weight=0.9)
    assert config.search_mode == "vector"
    assert config.final_top_k == 5
    assert config.vector_weight == 0.9


def test_from_dict():
    d = {"search_mode": "keyword", "bm25_top_k": 30}
    config = MilvusRetrievalConfig(**{k: v for k, v in d.items() if hasattr(MilvusRetrievalConfig, k)})
    assert config.search_mode == "keyword"
    assert config.bm25_top_k == 30


def test_all_fields_exist():
    config = MilvusRetrievalConfig()
    expected_fields = [
        "search_mode", "final_top_k", "similarity_threshold", "bm25_top_k",
        "vector_weight", "bm25_weight", "bm25_drop_ratio_search", "include_distances",
        "use_reranker", "reranker_model", "recall_top_k", "use_graph_retrieval",
        "graph_entity_top_k", "graph_triple_top_k", "graph_max_nodes", "graph_top_k",
        "graph_weight", "ppr_damping",
    ]
    for field in expected_fields:
        assert hasattr(config, field), f"Missing field: {field}"
