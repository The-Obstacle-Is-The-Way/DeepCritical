import time

import pytest

from DeepResearch.src.ingestion.file_filter import FileFilter
from DeepResearch.src.ingestion.file_watcher import FileWatcher
from DeepResearch.src.ingestion.indexing_pipeline import IndexingPipeline


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

    # Wait for indexing (watchdog events are async)
    time.sleep(2)

    # Verify file was indexed
    # Note: search is async
    from DeepResearch.src.datatypes.rag import SearchType

    results = await vector_store_fixture.search("Hello", SearchType.SIMILARITY, top_k=1)
    assert len(results) > 0
    assert "Hello world" in results[0].document.content

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
    time.sleep(1)
    test_file.write_text("Modified content")
    time.sleep(2)

    # Verify modified content is searchable
    from DeepResearch.src.datatypes.rag import SearchType

    results = await vector_store_fixture.search(
        "Modified", SearchType.SIMILARITY, top_k=1
    )
    assert len(results) > 0
    assert "Modified content" in results[0].document.content

    # Cleanup
    watcher.stop()
    pipeline.stop()


@pytest.mark.asyncio
async def test_file_watcher_detects_deletion(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that FileWatcher detects file deletions."""
    # Setup
    test_file = tmp_path / "test.txt"
    test_file.write_text("Content to delete")

    file_filter = FileFilter(allowed_extensions=[".txt"])
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=1
    )
    pipeline.start()

    watcher = FileWatcher(
        watch_paths=[str(tmp_path)], file_filter=file_filter, indexing_pipeline=pipeline
    )
    watcher.start()

    # Ensure initial indexing
    time.sleep(2)
    from DeepResearch.src.datatypes.rag import SearchType

    results = await vector_store_fixture.search(
        "Content", SearchType.SIMILARITY, top_k=1
    )
    assert len(results) > 0

    # Delete file
    test_file.unlink()
    time.sleep(2)

    # Verify deletion from index
    results_after = await vector_store_fixture.search(
        "Content", SearchType.SIMILARITY, top_k=1
    )
    # Should be empty or significantly different (if using dummy embeddings)
    # With empty store, should be empty
    assert len(results_after) == 0

    # Cleanup
    watcher.stop()
    pipeline.stop()
