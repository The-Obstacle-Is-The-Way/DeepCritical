"""Indexing pipeline orchestrator."""

import asyncio
import logging
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from typing import Any

from DeepResearch.src.datatypes.rag import Document, Embeddings, VectorStore
from DeepResearch.src.ingestion.document_parser import ParserFactory

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Orchestrates file parsing, embedding, and indexing."""

    def __init__(
        self, embeddings: Embeddings, vector_store: VectorStore, batch_size: int = 50
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.batch_size = batch_size
        # Queue holds paths to index (str) or paths to delete (tuple("DELETE", path))
        self.queue: Queue[str | tuple[str, str]] = Queue()
        self.running = False
        self.worker_thread: threading.Thread | None = None

        # Threading and async management
        self._loop: asyncio.AbstractEventLoop | None = None

        # Stats
        self.indexed_files: set[str] = set()
        self.last_index_time: datetime | None = None
        self._total_documents_indexed: int = 0
        self._total_batches_processed: int = 0

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

    def enqueue_deletion(self, file_path: str) -> None:
        """Add file deletion to queue."""
        self.queue.put(("DELETE", file_path))

    async def _remove_file_internal(self, file_path: str) -> None:
        """Remove file from index (internal async)."""
        # Find all document IDs for this file
        if hasattr(self.vector_store, "documents"):
            # Cast for type safety if needed
            from typing import cast

            docs_dict = cast("dict[str, Document]", self.vector_store.documents)
            doc_ids = [
                doc_id
                for doc_id, doc in docs_dict.items()
                if doc.metadata.get("file_path") == file_path
            ]
            if doc_ids:
                await self.vector_store.delete_documents(doc_ids)

                # Remove from stats if present
                if file_path in self.indexed_files:
                    self.indexed_files.remove(file_path)

                logger.info(f"Removed file {file_path} and {len(doc_ids)} chunks")

    def _process_queue(self) -> None:
        """Background worker that processes indexing queue."""
        # Create event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        batch: list[Document] = []
        last_flush_time = time.time()
        flush_interval = 2.0  # Seconds

        try:
            while self.running:
                try:
                    # Wait for 1 second, then check if running again
                    try:
                        item = self.queue.get(timeout=1)

                        if isinstance(item, tuple) and item[0] == "DELETE":
                            # Handle deletion immediately (high priority, no batching)
                            # Flush current batch first to maintain order consistency
                            if batch:
                                self._loop.run_until_complete(self._index_batch(batch))
                                batch = []
                                last_flush_time = time.time()

                            self._loop.run_until_complete(
                                self._remove_file_internal(item[1])
                            )

                        else:
                            # Handle indexing
                            file_path = str(item)
                            parser = ParserFactory.get_parser(file_path)
                            documents = parser.parse(file_path)
                            batch.extend(documents)

                    except Empty:
                        pass

                    current_time = time.time()
                    is_full_batch = len(batch) >= self.batch_size
                    is_time_to_flush = (
                        batch and (current_time - last_flush_time) >= flush_interval
                    )

                    if is_full_batch or is_time_to_flush:
                        self._loop.run_until_complete(self._index_batch(batch))
                        batch = []
                        last_flush_time = time.time()

                except Exception as e:
                    logger.error(f"Error processing queue item: {e}")

            # Process remaining
            if batch:
                self._loop.run_until_complete(self._index_batch(batch))
        finally:
            self._loop.close()

    async def _index_batch(self, documents: list[Document]) -> None:
        """Embed and index a batch of documents."""
        if not documents:
            return
        try:
            await self.vector_store.add_documents(documents)

            # Update stats
            self._total_documents_indexed += len(documents)
            self._total_batches_processed += 1
            self.last_index_time = datetime.now()

            # Track indexed files
            for doc in documents:
                if file_path := doc.metadata.get("file_path"):
                    self.indexed_files.add(file_path)

            logger.info(f"Indexed batch of {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error indexing batch: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get indexing statistics."""
        return {
            "indexed_files": sorted(self.indexed_files),
            "total_files": len(self.indexed_files),
            "total_documents": self._total_documents_indexed,
            "total_batches": self._total_batches_processed,
            "last_index_time": (
                self.last_index_time.isoformat() if self.last_index_time else None
            ),
            "queue_depth": self.queue.qsize(),
            "running": self.running,
        }
