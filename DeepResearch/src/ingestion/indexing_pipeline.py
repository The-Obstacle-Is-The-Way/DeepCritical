"""Indexing pipeline orchestrator."""

import asyncio
import threading
from queue import Empty, Queue
from typing import Any

from DeepResearch.src.datatypes.rag import Document, Embeddings, VectorStore
from DeepResearch.src.ingestion.document_parser import ParserFactory


class IndexingPipeline:
    """Orchestrates file parsing, embedding, and indexing."""

    def __init__(
        self, embeddings: Embeddings, vector_store: VectorStore, batch_size: int = 50
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.queue: Queue[str] = Queue()
        self.running = False
        self.worker_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background worker."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        """Stop background worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def enqueue_file(self, file_path: str) -> None:
        """Add file to indexing queue."""
        self.queue.put(file_path)

    async def remove_file(self, file_path: str) -> None:
        """Remove file from index (deleted file)."""
        # Find all document IDs for this file
        # Note: This assumes vector_store has a 'documents' dict which FAISSVectorStore does
        if hasattr(self.vector_store, "documents"):
            # Cast for type safety if needed, but python runtime is duck typed
            from typing import cast

            docs_dict = cast("dict[str, Document]", self.vector_store.documents)
            doc_ids = [
                doc_id
                for doc_id, doc in docs_dict.items()
                if doc.metadata.get("file_path") == file_path
            ]
            if doc_ids:
                await self.vector_store.delete_documents(doc_ids)

    def _process_queue(self) -> None:
        """Background worker that processes indexing queue."""
        batch: list[Document] = []

        while self.running:
            try:
                # Wait for 1 second, then check if running again
                file_path = self.queue.get(timeout=1)

                parser = ParserFactory.get_parser(file_path)
                documents = parser.parse(file_path)
                batch.extend(documents)

                if len(batch) >= self.batch_size:
                    self._index_batch(batch)
                    batch = []
            except Empty:
                # Queue empty, verify if we still run or if we have remaining batch
                continue
            except Exception as e:
                print(f"Error indexing: {e}")

        # Process remaining
        if batch:
            self._index_batch(batch)

    def _index_batch(self, documents: list[Document]) -> None:
        """Embed and index a batch of documents."""
        # Run async method in a new event loop since we are in a thread
        asyncio.run(self.vector_store.add_documents(documents))
