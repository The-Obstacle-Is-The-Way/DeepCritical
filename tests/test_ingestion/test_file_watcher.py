import asyncio
import time

import pytest

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
async def test_file_watcher_detects_new_file(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that FileWatcher detects new file creation."""
    # Setup
    file_filter = FileFilter(allowed_extensions=[".txt"])
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=1
    )
    pipeline.start()

    watcher = FileWatcher(
        watch_paths=[str(tmp_path)], file_filter=file_filter, indexing_pipeline=pipeline
    )

    # Start watching
    watcher.start()

    # Create file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello world")

    # Wait for indexing
    from DeepResearch.src.datatypes.rag import SearchType

    async def _is_indexed():
        results = await vector_store_fixture.search(
            "Hello", SearchType.SIMILARITY, top_k=1
        )
        return len(results) > 0 and "Hello world" in results[0].document.content

    await wait_for_condition(_is_indexed, error_msg="New file not indexed")

    # Cleanup
    watcher.stop()
    pipeline.stop()


@pytest.mark.asyncio
async def test_file_watcher_detects_modification(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that FileWatcher detects file modifications."""
    # Setup
    test_file = tmp_path / "test.txt"
    test_file.write_text("Original content")

    file_filter = FileFilter(allowed_extensions=[".txt"])
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=1
    )
    pipeline.start()

    watcher = FileWatcher(
        watch_paths=[str(tmp_path)], file_filter=file_filter, indexing_pipeline=pipeline
    )
    watcher.start()

    # Modify file
    await asyncio.sleep(0.1)  # Brief pause to ensure filesystem timestamp diff
    test_file.write_text("Modified content")

    # Verify modified content is searchable
    from DeepResearch.src.datatypes.rag import SearchType

    async def _is_modified():
        results = await vector_store_fixture.search(
            "Modified", SearchType.SIMILARITY, top_k=1
        )
        return len(results) > 0 and "Modified content" in results[0].document.content

    await wait_for_condition(_is_modified, error_msg="File modification not detected")

    # Cleanup
    watcher.stop()
    pipeline.stop()


@pytest.mark.asyncio
async def test_file_watcher_detects_deletion(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that FileWatcher detects file deletions."""
    # Setup pipeline and watcher FIRST
    file_filter = FileFilter(allowed_extensions=[".txt"])
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=1
    )
    pipeline.start()

    watcher = FileWatcher(
        watch_paths=[str(tmp_path)], file_filter=file_filter, indexing_pipeline=pipeline
    )
    watcher.start()

    # NOW create the file (watcher will see the creation event)
    test_file = tmp_path / "test.txt"
    test_file.write_text("Content to delete")

    # Wait for indexing
    from DeepResearch.src.datatypes.rag import SearchType

    async def _is_indexed():
        results = await vector_store_fixture.search(
            "Content", SearchType.SIMILARITY, top_k=1
        )
        return len(results) > 0

    await wait_for_condition(_is_indexed, error_msg="File to delete not indexed")

    # Delete file
    test_file.unlink()

    # Verify deletion from index
    async def _is_deleted():
        results_after = await vector_store_fixture.search(
            "Content", SearchType.SIMILARITY, top_k=1
        )
        return len(results_after) == 0

    await wait_for_condition(_is_deleted, error_msg="File deletion not detected")

    # Cleanup
    watcher.stop()
    pipeline.stop()
