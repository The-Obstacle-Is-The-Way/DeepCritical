import pytest

from DeepResearch.src.datatypes.embeddings_factory import create_embeddings
from DeepResearch.src.datatypes.rag import EmbeddingModelType, EmbeddingsConfig


def test_openai_error_message():
    config = EmbeddingsConfig(
        model_type=EmbeddingModelType.OPENAI, model_name="text-embedding-3-small"
    )
    with pytest.raises(ValueError, match="OpenAI embeddings are not implemented"):
        create_embeddings(config)
