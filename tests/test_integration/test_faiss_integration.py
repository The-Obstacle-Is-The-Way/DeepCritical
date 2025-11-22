"""Integration tests for FAISS vector store with embeddings."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from DeepResearch.src.datatypes.embeddings_factory import create_embeddings
from DeepResearch.src.datatypes.rag import (
    Document,
    EmbeddingModelType,
    EmbeddingsConfig,
    SearchType,
    VectorStoreType,
)
from DeepResearch.src.vector_stores import create_vector_store
from DeepResearch.src.vector_stores.faiss_config import FAISSVectorStoreConfig


@pytest.fixture
def mock_sentence_transformer():
    with patch(
        "DeepResearch.src.datatypes.sentence_transformer_embeddings.SentenceTransformer"
    ) as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        def side_effect(texts, **kwargs):
            # Return deterministic embeddings based on text length/content
            # This ensures "machine learning" query matches "machine learning" doc if we design it right
            # or at least ensures consistency.
            # For this test, we want "machine learning" query to match "machine learning embeddings" doc.

            results = []
            for text in texts:
                # Create a deterministic vector
                # We'll use a simple heuristic:
                # if "machine" or "learning" in text, use vector A
                # else use vector B
                vec = np.zeros(384, dtype=np.float32)
                if "machine" in text or "learning" in text:
                    vec[0] = 1.0  # High similarity to other "machine learning" texts
                else:
                    vec[1] = 1.0  # Orthogonal to "machine learning"

                # Add some noise based on text hash to avoid exact duplicates if needed
                # but keep the main signal strong
                results.append(vec)

            return np.array(results)

        mock_instance.encode.side_effect = side_effect
        yield MockModel


@pytest.fixture
def embeddings(mock_sentence_transformer):
    """Create embeddings for testing."""
    config = EmbeddingsConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        num_dimensions=384,
        batch_size=32,
        device="cpu",
    )
    return create_embeddings(config)


@pytest.fixture
def temp_faiss_paths():
    """Create temporary paths for FAISS index and data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test_faiss.index")
        data_path = os.path.join(tmpdir, "test_faiss_docs.pkl")
        yield index_path, data_path


@pytest.mark.asyncio
async def test_persistence(embeddings, temp_faiss_paths):
    """Test: Add docs, save, restart, search - data persists."""
    index_path, data_path = temp_faiss_paths

    # Phase 1: Create store, add documents, save
    config = FAISSVectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        embedding_dimension=384,
        index_path=index_path,
        data_path=data_path,
    )
    store = create_vector_store(config, embeddings)

    docs = [
        Document(id="doc1", content="The quick brown fox"),
        Document(id="doc2", content="jumps over the lazy dog"),
        Document(id="doc3", content="machine learning embeddings"),
    ]
    doc_ids = await store.add_documents(docs)
    assert len(doc_ids) == 3

    # Verify index and data files were created
    assert os.path.exists(index_path), "Index file should exist after save"
    assert os.path.exists(data_path), "Data file should exist after save"

    # Phase 2: Create NEW store instance (simulates restart), verify data persists
    # IMPORTANT: We must close the previous store or ensure resources are freed if needed
    # FAISS indices in memory are independent, but let's just create a new one.

    store2 = create_vector_store(config, embeddings)

    results = await store2.search(
        "machine learning", search_type=SearchType.SIMILARITY, top_k=1
    )
    assert len(results) == 1
    assert results[0].document.id == "doc3"
    assert "machine learning" in results[0].document.content


@pytest.mark.asyncio
async def test_determinism(embeddings, temp_faiss_paths):
    """Test: Same doc IDs -> Same internal IDs (deterministic hashing)."""
    index_path, data_path = temp_faiss_paths

    config = FAISSVectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        embedding_dimension=384,
        index_path=index_path,
        data_path=data_path,
    )

    # Add same documents twice, verify internal IDs are identical
    store = create_vector_store(config, embeddings)

    doc1 = Document(id="stable_id_123", content="test content")
    await store.add_documents([doc1])

    # Get the internal hash for this doc_id
    from DeepResearch.src.vector_stores.faiss_vector_store import _stable_hash

    hash1 = _stable_hash("stable_id_123")

    # Create new store with same config (pointing to same files)
    # We need to handle the index file. add_documents calls _save().
    # If we overwrite the index file with a new store, it might be tricky.
    # But here we are testing that if we add the same doc ID again (even to a new store),
    # the hash calculation is consistent.

    # Let's create a fresh store instance pointing to same paths
    store2 = create_vector_store(config, embeddings)
    doc2 = Document(id="stable_id_123", content="different content same id")
    await store2.add_documents([doc2])

    hash2 = _stable_hash("stable_id_123")

    # Hashes must be identical (deterministic)
    assert hash1 == hash2, "Same doc ID must produce same hash"

    # Verify id_map consistency
    # Casting to FAISSVectorStore to access id_map
    from DeepResearch.src.vector_stores.faiss_vector_store import FAISSVectorStore

    assert isinstance(store2, FAISSVectorStore)

    # NOTE: store2 just added doc2. The id_map should be populated.
    assert hash1 in store2.id_map
    assert store2.id_map[hash1] == "stable_id_123"
