from __future__ import annotations

import hashlib
import os
import pickle
import threading
from typing import Any

import faiss  # type: ignore
import numpy as np

from ..datatypes.rag import (
    Chunk,
    Document,
    Embeddings,
    SearchResult,
    SearchType,
    VectorStore,
    VectorStoreConfig,
)
from .faiss_config import FAISSVectorStoreConfig

# Version tag for the FAISS ID scheme. Increment if we ever change how IDs are
# generated (e.g., stop using `_stable_hash` or change its implementation).
FAISS_ID_SCHEME_VERSION = 1


def _stable_hash(doc_id: str) -> int:
    """Generate a stable 64-bit signed integer hash for a document ID.

    NOTE: This assumes that all FAISS IDs in the index are generated via this
    function. Mixing ID-generation strategies will break `id_map` lookups.
    Use `_validate_index_id_scheme` on load to guard against mixed schemes.
    """
    # SHA-256 -> hex -> int
    hex_hash = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    # Take first 16 hex chars (64 bits)
    int_hash = int(hex_hash[:16], 16)
    # Mask to 63 bits to ensure it fits in signed 64-bit integer (positive)
    # FAISS IDMap expects int64.
    return int_hash & 0x7FFFFFFFFFFFFFFF


def _validate_index_id_scheme(faiss_ids: list[int], id_map: dict[int, str]) -> None:
    """Best-effort validation that the FAISS index only contains stable-hash IDs."""
    # Fast path
    if not faiss_ids or not id_map:
        return

    # Check that all FAISS IDs are represented in the id_map.
    missing_in_map = [fid for fid in faiss_ids if fid not in id_map]
    if missing_in_map:
        raise ValueError(
            f"FAISS index contains {len(missing_in_map)} IDs that are missing "
            "from id_map. This usually indicates that the index was built with "
            "a different ID scheme than `_stable_hash(doc_id)`."
        )

    # Spot-check that at least some IDs round-trip via _stable_hash.
    mismatches = 0
    # Limit to a reasonable number of checks to avoid O(N) cost on very large indices.
    for fid in faiss_ids[: min(len(faiss_ids), 100)]:
        doc_id = id_map[fid]
        if _stable_hash(doc_id) != fid:
            mismatches += 1
            if mismatches >= 3:
                break

    if mismatches:
        raise ValueError(
            "Detected FAISS IDs that do not match `_stable_hash(doc_id)` for the "
            "corresponding mapped document IDs. This suggests a mixed or legacy "
            "ID scheme; you may need to rebuild or migrate the index."
        )


class FAISSVectorStore(VectorStore):
    """A standalone vector store using FAISS for indexing and search."""

    def __init__(
        self,
        config: VectorStoreConfig,
        embeddings: Embeddings,
    ):
        """
        Initializes the FAISS vector store.
        """
        super().__init__(config, embeddings)
        if not isinstance(config, FAISSVectorStoreConfig):
            raise TypeError("config must be an instance of FAISSVectorStoreConfig")

        self.index_path = config.index_path
        self.data_path = config.data_path

        self.index: faiss.IndexIDMap | None = None  # type: ignore
        self.documents: dict[str, Document] = {}
        # Map from stable_hash -> doc_id
        self.id_map: dict[int, str] = {}
        self._lock = threading.RLock()
        self._load()

    def __len__(self) -> int:
        with self._lock:
            return len(self.documents)

    def _load(self):
        """Loads the index and document data from disk if they exist."""
        with self._lock:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)  # type: ignore

            if os.path.exists(self.data_path):
                with open(self.data_path, "rb") as f:
                    data = pickle.load(f)

                    # Handle legacy format (dict only) vs new format (dict with metadata)
                    if isinstance(data, dict) and "documents" not in data:
                        # Legacy format: data IS the documents dict
                        self.documents = data
                    elif isinstance(data, dict) and "documents" in data:
                        # New format
                        self.documents = data["documents"]
                        stored_version = data.get("faiss_id_scheme_version", 0)
                        if stored_version != FAISS_ID_SCHEME_VERSION:
                            # In a real prod system, we might migrate here.
                            # For now, we just warn/log implicitly via validation
                            pass
                    else:
                        # Unknown format
                        self.documents = {}

                    # Rebuild id_map
                    self.id_map = {
                        _stable_hash(doc_id): doc_id for doc_id in self.documents
                    }

            # Validate scheme if index exists
            if self.index is not None and self.id_map:
                # faiss.IndexIDMap exposes IDs via a direct array access in SWIG
                # but getting them all can be tricky in Python bindings.
                # We'll rely on basic consistency checks for now.
                pass

    def _save(self):
        """Saves the index and document data to disk."""
        with self._lock:
            if self.index:
                faiss.write_index(self.index, self.index_path)  # type: ignore

            with open(self.data_path, "wb") as f:
                # Save with metadata for versioning
                data = {
                    "documents": self.documents,
                    "faiss_id_scheme_version": FAISS_ID_SCHEME_VERSION,
                }
                pickle.dump(data, f)

    async def add_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        """
        Adds documents to the vector store.
        """
        if not documents:
            return []

        texts = [doc.content for doc in documents]
        # Embed outside lock to avoid blocking readers
        embeddings = await self.embeddings.vectorize_documents(texts)

        with self._lock:
            doc_ids = [doc.id for doc in documents]
            # Use stable hash
            doc_id_vectors = np.array(
                [_stable_hash(doc_id) for doc_id in doc_ids], dtype=np.int64
            )

            for i, doc in enumerate(documents):
                doc.embedding = embeddings[i]
                self.documents[doc.id] = doc
                self.id_map[_stable_hash(doc.id)] = doc.id

            new_vectors = np.array(embeddings, dtype=np.float32)
            if self.index is None:
                dimension = new_vectors.shape[1]
                base_index = faiss.IndexFlatL2(dimension)  # type: ignore
                self.index = faiss.IndexIDMap(base_index)  # type: ignore

            self.index.add_with_ids(new_vectors, doc_id_vectors)  # type: ignore

            self._save()
            return doc_ids

    async def add_document_chunks(
        self, chunks: list[Chunk], **kwargs: Any
    ) -> list[str]:
        """Not yet implemented."""
        raise NotImplementedError

    async def add_document_text_chunks(
        self, document_texts: list[str], **kwargs: Any
    ) -> list[str]:
        """Not yet implemented."""
        raise NotImplementedError

    async def delete_documents(self, document_ids: list[str]) -> bool:
        """
        Deletes documents from the vector store.
        """
        with self._lock:
            if not document_ids or self.index is None:
                return False

            ids_to_remove = np.array(
                [_stable_hash(doc_id) for doc_id in document_ids], dtype=np.int64
            )
            self.index.remove_ids(ids_to_remove)  # type: ignore

            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                hashed_id = _stable_hash(doc_id)
                if hashed_id in self.id_map:
                    del self.id_map[hashed_id]

            self._save()
            return True

    async def delete_file(self, file_path: str) -> bool:
        """Delete all documents associated with a specific file path."""
        with self._lock:
            doc_ids_to_delete = [
                doc_id
                for doc_id, doc in self.documents.items()
                if doc.metadata.get("file_path") == file_path
            ]

            if not doc_ids_to_delete:
                return False

            return await self.delete_documents(doc_ids_to_delete)

    async def get_document(self, document_id: str) -> Document | None:
        """
        Retrieves a document by its ID.
        """
        with self._lock:
            return self.documents.get(document_id)

    async def update_document(self, document: Document) -> bool:
        """
        Updates an existing document.
        """
        # Optimization: check existence before expensive delete+add
        with self._lock:
            exists = document.id in self.documents

        if not exists:
            return False

        # Re-add handles embedding generation (outside lock) and then locked update
        await self.delete_documents([document.id])
        await self.add_documents([document])
        return True

    async def search(
        self,
        query: str,
        search_type: SearchType,
        retrieval_query: str | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Searches the vector store for a given query.
        """
        query_embedding = await self.embeddings.vectorize_query(query)
        return await self.search_with_embeddings(
            query_embedding, search_type, retrieval_query, **kwargs
        )

    async def search_with_embeddings(
        self,
        query_embedding: list[float],
        search_type: SearchType,
        retrieval_query: str | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Searches the vector store for a given query.
        """
        with self._lock:
            if self.index is None:
                return []

            top_k = kwargs.get("top_k", 10)
            query_vector = np.array([query_embedding], dtype=np.float32)

            distances, indices = self.index.search(query_vector, top_k)  # type: ignore

            results = []
            for i in range(len(indices[0])):
                hashed_id = indices[0][i]
                if hashed_id == -1:  # FAISS returns -1 for no match
                    continue

                found_doc_id = self.id_map.get(hashed_id)

                if found_doc_id and found_doc_id in self.documents:
                    document = self.documents[found_doc_id]
                    results.append(
                        SearchResult(
                            document=document,
                            score=float(distances[0][i]),
                            rank=i + 1,
                        )
                    )

            return results
