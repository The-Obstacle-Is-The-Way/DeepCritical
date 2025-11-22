from unittest.mock import MagicMock

import pytest

from DeepResearch.src.datatypes.rag import VectorStoreType
from DeepResearch.src.vector_stores.faiss_config import FAISSVectorStoreConfig
from DeepResearch.src.vector_stores.faiss_vector_store import FAISSVectorStore


@pytest.fixture
def embeddings_fixture():
    """Fixture for a mock embeddings provider."""
    mock = MagicMock()

    async def vectorize_documents(texts: list[str]) -> list[list[float]]:
        # Return dummy vectors of size 2
        return [[0.1, 0.1] for _ in texts]

    async def vectorize_query(text: str) -> list[float]:
        return [0.1, 0.1]

    mock.vectorize_documents = vectorize_documents
    mock.vectorize_query = vectorize_query
    return mock


@pytest.fixture
def vector_store_fixture(tmp_path, embeddings_fixture):
    """Fixture for a FAISSVectorStore instance."""
    index_path = str(tmp_path / "test.index")
    data_path = str(tmp_path / "test.data")
    config = FAISSVectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        index_path=index_path,
        data_path=data_path,
    )
    return FAISSVectorStore(config, embeddings_fixture)
