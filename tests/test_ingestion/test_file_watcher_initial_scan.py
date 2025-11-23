import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from DeepResearch.src.datatypes.rag import SearchType
from DeepResearch.src.ingestion.file_filter import FileFilter
from DeepResearch.src.ingestion.file_watcher import FileWatcher
from DeepResearch.src.ingestion.indexing_pipeline import IndexingPipeline


async def wait_for_condition(
    condition_fn,
    timeout: float = 10.0,
    interval: float = 0.1,
    error_msg: str = "Condition not met",
) -> None:
    """Poll condition_fn until True or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        if await condition_fn():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"{error_msg} (timeout={timeout}s)")


@pytest.mark.asyncio
async def test_file_watcher_initial_scan_integration(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that FileWatcher indexes existing files on start (Integration)."""
    # Setup: Create file BEFORE starting watcher
    test_file = tmp_path / "existing.txt"
    test_file.write_text("Existing content")

    file_filter = FileFilter(allowed_extensions=[".txt"])
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=1
    )
    pipeline.start()

    watcher = FileWatcher(
        watch_paths=[str(tmp_path)], file_filter=file_filter, indexing_pipeline=pipeline
    )

    # Start watcher with initial_scan=True
    watcher.start(initial_scan=True)

    # Verify file is indexed
    async def _is_indexed():
        results = await vector_store_fixture.search(
            "Existing", SearchType.SIMILARITY, top_k=1
        )
        return len(results) > 0 and "Existing content" in results[0].document.content

    await wait_for_condition(
        _is_indexed, timeout=10.0, error_msg="Existing file not indexed on start"
    )

    watcher.stop()
    pipeline.stop()


def test_file_watcher_initial_scan_logic(
    tmp_path, embeddings_fixture, vector_store_fixture
):
    """Unit test to verify initial_scan flag controls _initial_crawl."""
    file_filter = FileFilter(allowed_extensions=[".txt"])
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=1
    )

    # Case 1: initial_scan=True
    watcher = FileWatcher(
        watch_paths=[str(tmp_path)], file_filter=file_filter, indexing_pipeline=pipeline
    )
    with patch.object(watcher, "_initial_crawl") as mock_crawl:
        with patch.object(watcher.observer, "start"):  # Don't actually start observer
            watcher.start(initial_scan=True)
            mock_crawl.assert_called_once()

    # Case 2: initial_scan=False
    watcher = FileWatcher(
        watch_paths=[str(tmp_path)], file_filter=file_filter, indexing_pipeline=pipeline
    )
    with patch.object(watcher, "_initial_crawl") as mock_crawl:
        with patch.object(watcher.observer, "start"):
            watcher.start(initial_scan=False)
            mock_crawl.assert_not_called()
