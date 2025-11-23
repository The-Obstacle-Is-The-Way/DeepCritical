"""Mgrep Server - Semantic Code Search Service."""

# CRITICAL: Set thread limits BEFORE importing numpy/FAISS to prevent N² thread explosion
# See: https://github.com/facebookresearch/faiss/issues/3700
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, cast

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from DeepResearch.src.datatypes.embeddings_factory import EmbeddingsConfig
from DeepResearch.src.datatypes.rag import SearchResult, SearchType
from DeepResearch.src.datatypes.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddings,
)
from DeepResearch.src.ingestion.file_filter import FileFilter
from DeepResearch.src.ingestion.file_watcher import FileWatcher
from DeepResearch.src.ingestion.indexing_pipeline import IndexingPipeline
from DeepResearch.src.vector_stores.faiss_config import FAISSVectorStoreConfig
from DeepResearch.src.vector_stores.faiss_vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class MgrepServer:
    """Singleton server for semantic code search."""

    _instance: "MgrepServer | None" = None

    def __init__(self, config_path: str | Path | None = None):
        """Initialize Mgrep Server (use get_instance() instead)."""
        # Ensure .env variables (e.g., ANTHROPIC_API_KEY) are available before imports
        load_dotenv()
        self.config = self._load_config(config_path)
        self.running = False

        self.embeddings: SentenceTransformerEmbeddings | None = None
        self.vector_store: FAISSVectorStore | None = None
        self.pipeline: IndexingPipeline | None = None
        self.watcher: FileWatcher | None = None

        self._initialize_components()

    @classmethod
    def get_instance(cls, config_path: str | Path | None = None) -> "MgrepServer":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def _load_config(self, config_path: str | Path | None) -> DictConfig:
        """Load configuration."""
        if config_path:
            loaded = OmegaConf.load(config_path)
            return cast("DictConfig", loaded)

        default_path = Path("configs/tools/mgrep.yaml")
        if default_path.exists():
            loaded = OmegaConf.load(default_path)
            return cast("DictConfig", loaded)

        raise FileNotFoundError("No mgrep config found")

    def _initialize_components(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Mgrep Server components...")

        self._reset_data_if_needed()

        # 1. Embeddings
        emb_cfg = self.config.embeddings
        embeddings_config = EmbeddingsConfig(
            model_type=emb_cfg.model_type,
            model_name=emb_cfg.model_name,
            device=emb_cfg.device,
            batch_size=emb_cfg.batch_size,
            num_dimensions=self.config.vector_store.embedding_dimension,
        )
        self.embeddings = SentenceTransformerEmbeddings(embeddings_config)

        # CRITICAL: Pre-load model to prevent race conditions
        logger.info("Pre-loading embedding model...")
        _ = self.embeddings.model
        logger.info("Embedding model loaded.")

        # 2. Vector Store
        data_path = Path(self.config.data_path)
        data_path.mkdir(parents=True, exist_ok=True)

        vs_cfg = self.config.vector_store
        from DeepResearch.src.datatypes.rag import VectorStoreType

        faiss_config = FAISSVectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            embedding_dimension=vs_cfg.embedding_dimension,
            index_path=vs_cfg.index_path,
            data_path=vs_cfg.data_path,
        )
        self.vector_store = FAISSVectorStore(faiss_config, self.embeddings)

        # CRITICAL: Force FAISS to use single-threaded mode to prevent deadlocks
        # See: https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
        import faiss  # type: ignore

        faiss.omp_set_num_threads(1)  # type: ignore
        logger.info("FAISS threading set to 1 (single-threaded mode)")

        # 3. Indexing Pipeline
        self.pipeline = IndexingPipeline(
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            batch_size=self.config.indexing.batch_size,
        )

        # 4. File Watcher
        file_filter = FileFilter(
            allowed_extensions=list(self.config.allowed_extensions),
            exclude_patterns=list(self.config.exclude_patterns),
        )

        self.watcher = FileWatcher(
            watch_paths=list(self.config.watch_paths),
            file_filter=file_filter,
            indexing_pipeline=self.pipeline,
            recursive=True,
        )

        logger.info("Mgrep Server initialized.")

    def start(self) -> None:
        """Start background services."""
        if self.running:
            logger.warning("Mgrep Server already running.")
            return

        logger.info("Starting Mgrep Server...")
        if self.pipeline:
            self.pipeline.start()

        if self.watcher:
            self.watcher.start(initial_scan=self.config.indexing.initial_scan)

        self.running = True
        logger.info("Mgrep Server started.")

    def stop(self) -> None:
        """Stop background services."""
        if not self.running:
            return

        logger.info("Stopping Mgrep Server...")
        if self.watcher:
            self.watcher.stop()

        if self.pipeline:
            self.pipeline.stop()

        self.running = False
        logger.info("Mgrep Server stopped.")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Perform semantic search."""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        return await self.vector_store.search(
            query=query,
            search_type=SearchType.SIMILARITY,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def get_status(self) -> dict[str, Any]:
        """Get server status."""
        pipeline_stats = self.pipeline.get_stats() if self.pipeline else {}
        vector_store_stats = {
            "total_documents": len(self.vector_store) if self.vector_store else 0
        }
        return {
            "running": self.running,
            "pipeline": pipeline_stats,
            "vector_store": vector_store_stats,
            "config": {
                "watch_paths": self.config.watch_paths,
                "extensions": self.config.allowed_extensions,
            },
        }

    async def wait_until_ready(
        self,
        min_files: int = 1,
        idle_grace_seconds: float = 2.0,
        timeout_seconds: float = 300.0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Block until the indexing pipeline is idle and has processed files.

        Args:
            min_files: Minimum number of files that must be indexed before returning.
            idle_grace_seconds: How long the queue must remain empty before returning.
            timeout_seconds: Maximum time to wait before raising TimeoutError.

        Returns:
            The final server status dict.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")

        start = time.monotonic()
        idle_since: float | None = None

        while True:
            status = self.get_status()
            pipeline_stats = status.get("pipeline", {})
            vector_store_stats = status.get("vector_store", {})

            if progress_callback:
                progress_callback(status)

            queue_depth = pipeline_stats.get("queue_depth", 0)
            total_files = pipeline_stats.get("total_files", 0)
            last_index_time = pipeline_stats.get("last_index_time")
            total_documents = vector_store_stats.get("total_documents", 0)

            is_idle = (
                queue_depth == 0
                and last_index_time is not None
                and total_files >= min_files
                and total_documents > 0
            )

            now = time.monotonic()
            if is_idle:
                idle_since = idle_since or now
                if now - idle_since >= idle_grace_seconds:
                    return status
            else:
                idle_since = None

            if now - start > timeout_seconds:
                raise TimeoutError(
                    "Timeout waiting for indexing to become idle and ready."
                )

            await asyncio.sleep(0.5)

    def _reset_data_if_needed(self) -> None:
        """Optionally clear persisted index/doc store on startup (for demos/tests)."""
        reset = bool(getattr(self.config, "reset_on_start", False))
        if not reset:
            return

        data_dir = Path(self.config.data_path)
        index_path = Path(self.config.vector_store.index_path)
        store_path = Path(self.config.vector_store.data_path)

        try:
            if data_dir.exists():
                shutil.rmtree(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            # In case paths are outside data_dir, clean them explicitly.
            if index_path.exists():
                index_path.unlink()
            if store_path.exists():
                store_path.unlink()
            logger.info("Reset mgrep data directory for a clean start.")
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning(f"Failed to reset data directory '{data_dir}': {exc}")

    def reindex_all(self) -> None:
        """Trigger full re-scan."""
        if not self.watcher:
            raise RuntimeError("File watcher not initialized")

        logger.info("Triggering full re-index...")
        self.watcher.scan_all()
