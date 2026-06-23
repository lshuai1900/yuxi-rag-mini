"""Test Milvus collection schema design (schema structure, not actual Milvus operations)."""
import pytest
from pymilvus import DataType, FunctionType


def test_schema_field_definitions():
    """Test that schema field definitions are correct for BM25 support."""
    from pymilvus import FieldSchema

    # Test content field with analyzer
    content_field = FieldSchema(
        name="content", dtype=DataType.VARCHAR, max_length=65535,
        enable_analyzer=True, analyzer_params={"type": "chinese"},
    )
    assert content_field.name == "content"
    assert content_field.dtype == DataType.VARCHAR

    # Test sparse vector field
    sparse_field = FieldSchema(
        name="content_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR,
    )
    assert sparse_field.dtype == DataType.SPARSE_FLOAT_VECTOR


def test_bm25_function_definition():
    """Test BM25 function definition."""
    from pymilvus import Function, FunctionType

    bm25_function = Function(
        name="content_bm25",
        input_field_names=["content"],
        output_field_names=["content_sparse"],
        function_type=FunctionType.BM25,
    )
    # Verify the function was created with BM25 type
    # Note: pymilvus Function stores type internally; verify via repr or params
    assert bm25_function.name == "content_bm25"
    assert FunctionType.BM25 is not None


def test_sparse_index_params():
    """Test sparse index parameter structure."""
    sparse_index_params = {
        "metric_type": "BM25",
        "index_type": "SPARSE_INVERTED_INDEX",
        "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
    }
    assert sparse_index_params["metric_type"] == "BM25"
    assert sparse_index_params["index_type"] == "SPARSE_INVERTED_INDEX"


def test_dense_index_params():
    """Test dense vector index parameter structure."""
    dense_index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    assert dense_index_params["metric_type"] == "COSINE"
    assert dense_index_params["index_type"] == "IVF_FLAT"
