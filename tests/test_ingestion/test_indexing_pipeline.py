import asyncio
import time

import pytest

from DeepResearch.src.datatypes.rag import SearchType
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
async def test_indexing_pipeline_processes_files(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that IndexingPipeline parses and indexes files."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document")

    # Setup pipeline
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture,
        vector_store=vector_store_fixture,
        batch_size=1,  # Set to 1 to flush immediately
    )
    pipeline.start()

    # Enqueue file
    pipeline.enqueue_file(str(test_file))

    # Wait for processing
    async def _is_indexed():
        results = await vector_store_fixture.search(
            "test document", SearchType.SIMILARITY, top_k=1
        )
        return len(results) > 0 and "test document" in results[0].document.content

    await wait_for_condition(_is_indexed, error_msg="Document not indexed")

    # Cleanup
    pipeline.stop()


@pytest.mark.asyncio
async def test_indexing_pipeline_batch_processing(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that IndexingPipeline batches documents efficiently."""
    # Create multiple files
    for i in range(5):
        (tmp_path / f"test{i}.txt").write_text(f"Document {i}")

    # Setup pipeline with batch size 2
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=2
    )
    pipeline.start()

    # Enqueue all files
    for i in range(5):
        pipeline.enqueue_file(str(tmp_path / f"test{i}.txt"))

    # Wait for SOME processing (at least first 2 batches = 4 files)
    # We check if at least one file is indexed to verify it started
    async def _started_processing():
        stats = pipeline.get_stats()
        return stats["total_files"] >= 4

    await wait_for_condition(_started_processing, error_msg="Batch processing stalled")

    pipeline.stop()  # This flushes remaining batch

    for i in range(5):
        results = await vector_store_fixture.search(
            f"Document {i}", SearchType.SIMILARITY, top_k=1
        )
        assert len(results) > 0


@pytest.mark.asyncio
async def test_indexing_pipeline_remove_file(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that IndexingPipeline can remove files from the index."""
    # Setup pipeline
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=1
    )

    # Manually add documents to vector store to simulate indexed file
    from DeepResearch.src.datatypes.rag import Document

    file_path = str(tmp_path / "delete_me.txt")
    doc1 = Document(
        id="doc1", content="Chunk 1", metadata={"file_path": file_path, "type": "text"}
    )
    doc2 = Document(
        id="doc2", content="Chunk 2", metadata={"file_path": file_path, "type": "text"}
    )
    doc3 = Document(
        id="doc3",
        content="Keep me",
        metadata={"file_path": "other.txt", "type": "text"},
    )

    await vector_store_fixture.add_documents([doc1, doc2, doc3])

    # Verify presence
    assert len(await vector_store_fixture.search("Chunk 1", SearchType.SIMILARITY)) > 0
    assert len(await vector_store_fixture.search("Keep me", SearchType.SIMILARITY)) > 0

    # Start pipeline to process deletion
    pipeline.start()
    pipeline.enqueue_deletion(file_path)

    # Verify removal
    async def _is_deleted():
        results = await vector_store_fixture.search("Chunk 1", SearchType.SIMILARITY)
        found_ids = [r.document.id for r in results]
        return "doc1" not in found_ids and "doc2" not in found_ids

    await wait_for_condition(_is_deleted, error_msg="File not deleted")
    pipeline.stop()

    # Verify other file remains
    results_kept = await vector_store_fixture.search("Keep me", SearchType.SIMILARITY)
    assert len(results_kept) > 0
    assert "Keep me" in results_kept[0].document.content


@pytest.mark.asyncio
async def test_indexing_pipeline_stats(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test that pipeline tracks stats correctly."""
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture,
        vector_store=vector_store_fixture,
        batch_size=2,
    )
    pipeline.start()

    # Enqueue files
    for i in range(3):
        file_path = tmp_path / f"file{i}.txt"
        file_path.write_text(f"Content {i}")
        pipeline.enqueue_file(str(file_path))

    # Wait for first batch
    async def _first_batch_done():
        stats = pipeline.get_stats()
        return stats["total_files"] >= 2

    await wait_for_condition(_first_batch_done, error_msg="Stats not updating")

    pipeline.stop()

    stats = pipeline.get_stats()
    assert stats["total_files"] == 3
    assert stats["total_documents"] >= 3
    assert stats["total_batches"] >= 2
    assert stats["last_index_time"] is not None
    assert not stats["running"]
