from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..datatypes.rag import VectorStoreConfig, VectorStoreType


class FAISSVectorStoreConfig(VectorStoreConfig):
    """Configuration for the FAISS vector store."""

    store_type: Literal[VectorStoreType.FAISS] = Field(default=VectorStoreType.FAISS)
    index_path: str = Field(description="File path to save or load the FAISS index.")
    data_path: str = Field(description="File path to save or load the document data.")
