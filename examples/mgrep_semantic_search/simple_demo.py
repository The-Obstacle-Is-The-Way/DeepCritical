"""Mgrep E2E Demo."""

# CRITICAL: Set thread limits BEFORE any imports to prevent N² thread explosion
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent fork deadlock warnings

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parents[2]))

from DeepResearch.src.tools.mgrep_server import MgrepServer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mgrep_demo")


def _print_status(status: dict[str, object]) -> None:
    """Render a single-line status update to stdout."""
    from typing import Any, cast

    pipeline = cast("dict[str, Any]", status.get("pipeline", {}))
    vector_store = cast("dict[str, Any]", status.get("vector_store", {}))

    doc_count = int(vector_store.get("total_documents", 0))
    total_files = int(pipeline.get("total_files", 0))
    queue_depth = int(pipeline.get("queue_depth", 0))

    sys.stdout.write(
        f"\rIndexed: {doc_count} docs from {total_files} files | Queue: {queue_depth}   "
    )
    sys.stdout.flush()


async def main() -> None:
    logger.info("🚀 Starting Mgrep E2E Demo...")

    # Initialize
    server = MgrepServer.get_instance()
    server.start()

    # Wait for indexing to become idle and ready
    logger.info("Waiting for indexing...")
    try:
        await server.wait_until_ready(
            min_files=1,
            idle_grace_seconds=2.0,
            timeout_seconds=180.0,
            progress_callback=_print_status,
        )
        logger.info("\n✅ Indexing ready!")
    except TimeoutError:
        logger.error("\n❌ Timeout waiting for indexing to complete.")
        server.stop()
        return

    # Search
    query = "How does the FAISS vector store handle thread safety?"
    logger.info(f"\n🔎 Searching: '{query}'")

    results = await server.search(query, top_k=3)

    if not results:
        logger.error("❌ No results!")
    else:
        logger.info(f"✅ Found {len(results)} results:\n")
        for i, res in enumerate(results, 1):
            file_path = res.document.metadata.get("file_path", "unknown")
            print(f"--- Result {i} (Score: {res.score:.4f}) ---")
            print(f"File: {file_path}")
            print(f"Snippet: {res.document.content[:200]}...")
            print()

    # Cleanup
    server.stop()
    logger.info("👋 Demo finished.")


if __name__ == "__main__":
    asyncio.run(main())
