from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from DeepResearch.src.datatypes.rag import EmbeddingModelType, EmbeddingsConfig
from DeepResearch.src.datatypes.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddings,
)


@pytest.mark.asyncio
async def test_mixedbread_query_instruction_logic():
    """Test that query_instruction is properly prepended during encoding."""
    # Use small model for testing (not actual mixedbread to avoid download)
    config = EmbeddingsConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",  # Small model for testing
        num_dimensions=384,
        batch_size=32,
        device="cpu",
        query_instruction="Represent this sentence: ",
    )

    # Mock the SentenceTransformer class
    with patch(
        "DeepResearch.src.datatypes.sentence_transformer_embeddings.SentenceTransformer"
    ) as MockModel:
        # Setup the mock instance
        mock_instance = MockModel.return_value
        # Return dummy embeddings: list of numpy arrays
        rng = np.random.default_rng()
        dummy_embedding = rng.random(384).astype(np.float32)

        def side_effect(texts, **kwargs):
            return np.array([dummy_embedding] * len(texts))

        mock_instance.encode.side_effect = side_effect
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        embeddings = SentenceTransformerEmbeddings(config)

        # Verify instruction field is set
        assert embeddings.query_instruction == "Represent this sentence: "

        # Generate embeddings (instruction will be prepended internally)
        query_vector = await embeddings.vectorize_query("test query")
        doc_vectors = await embeddings.vectorize_documents(["test doc"])

        # Verify dimensions
        assert len(query_vector) == 384
        assert len(doc_vectors) == 1
        assert len(doc_vectors[0]) == 384

        # Verify that the mock was called with the correct text (prepended instruction)
        # Check query call
        mock_instance.encode.assert_any_call(
            ["Represent this sentence: test query"],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Check doc call (should NOT have instruction)
        mock_instance.encode.assert_any_call(
            ["test doc"],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Behavioral check: query and doc vectors should be valid floats
        assert isinstance(query_vector[0], (float, np.floating))
        assert isinstance(doc_vectors[0][0], (float, np.floating))
