import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from DeepResearch.src.datatypes.rag import EmbeddingModelType, EmbeddingsConfig
from DeepResearch.src.datatypes.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddings,
)


@pytest.fixture
def mock_sentence_transformer():
    with patch(
        "DeepResearch.src.datatypes.sentence_transformer_embeddings.SentenceTransformer"
    ) as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        def side_effect(texts, **kwargs):
            # Return dummy embeddings
            rng = np.random.default_rng()
            return rng.random((len(texts), 384)).astype(np.float32)

        mock_instance.encode.side_effect = side_effect
        yield MockModel


@pytest.fixture
def config():
    return EmbeddingsConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        num_dimensions=384,
        batch_size=32,
        device="cpu",
    )


@pytest.fixture
def embeddings(config, mock_sentence_transformer):
    return SentenceTransformerEmbeddings(config)


@pytest.mark.asyncio
async def test_dimensions(embeddings):
    """Test that embeddings have correct dimensions."""
    vector = await embeddings.vectorize_query("test")
    assert len(vector) == 384
    assert isinstance(vector, list)
    assert isinstance(vector[0], float)


@pytest.mark.asyncio
async def test_vectorize_documents(embeddings):
    """Test vectorizing a list of documents."""
    docs = ["doc1", "doc2", "doc3"]
    vectors = await embeddings.vectorize_documents(docs)
    assert len(vectors) == 3
    assert len(vectors[0]) == 384
    assert isinstance(vectors, list)
    assert isinstance(vectors[0], list)


def test_sync_methods(embeddings):
    """Test synchronous methods."""
    vector = embeddings.vectorize_query_sync("test")
    assert len(vector) == 384

    vectors = embeddings.vectorize_documents_sync(["doc1"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384


@pytest.mark.asyncio
async def test_query_instruction(mock_sentence_transformer):
    """Test that query instruction doesn't break execution."""
    config = EmbeddingsConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        num_dimensions=384,
        query_instruction="Represent: ",
    )
    emb = SentenceTransformerEmbeddings(config)

    # We can verification the instruction was used by checking the mock
    vector = await emb.vectorize_query("test")
    assert len(vector) == 384

    # Verify the mock was called with the prepended instruction
    mock_instance = mock_sentence_transformer.return_value
    mock_instance.encode.assert_called_with(
        ["Represent: test"],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


@pytest.mark.asyncio
async def test_empty_input(embeddings):
    """Test handling of empty inputs."""
    with pytest.raises(ValueError, match="Cannot vectorize empty query"):
        await embeddings.vectorize_query("")

    vectors = await embeddings.vectorize_documents([])
    assert vectors == []


def test_lazy_model_load_thread_safe(config, mock_sentence_transformer):
    """Ensure lazy model init happens exactly once under concurrency."""

    # Slow down instantiation to widen the race window
    def delayed_model(*args, **kwargs):
        time.sleep(0.01)
        return mock_sentence_transformer.return_value

    mock_sentence_transformer.side_effect = delayed_model
    embeddings = SentenceTransformerEmbeddings(config)

    def access_model():
        _ = embeddings.model

    threads = [threading.Thread(target=access_model) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive(), "Thread hung while acquiring model lock"

    assert mock_sentence_transformer.call_count == 1
