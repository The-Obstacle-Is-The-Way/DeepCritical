import pytest

from DeepResearch.src.datatypes.rag import SearchType
from DeepResearch.src.ingestion.file_filter import FileFilter
from DeepResearch.src.ingestion.file_watcher import FileWatcher
from DeepResearch.src.ingestion.indexing_pipeline import IndexingPipeline
from tests.utils.async_utils import wait_for_condition


@pytest.mark.asyncio
async def test_python_file_indexing_e2e(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Test end-to-end indexing of a Python file."""
    # Setup
    file_filter = FileFilter(allowed_extensions=[".py"])
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture, vector_store=vector_store_fixture, batch_size=1
    )
    pipeline.start()

    watcher = FileWatcher(
        watch_paths=[str(tmp_path)], file_filter=file_filter, indexing_pipeline=pipeline
    )
    watcher.start()

    # Create Python file
    py_file = tmp_path / "test_e2e.py"
    py_file.write_text("def my_function():\n    return 'success'")

    # Wait for indexing
    async def check_indexed():
        results = await vector_store_fixture.search(
            "my_function", SearchType.SIMILARITY, top_k=1
        )
        return len(results) > 0

    await wait_for_condition(
        check_indexed, timeout=5.0, error_msg="File failed to index"
    )

    # Verify searchable
    results = await vector_store_fixture.search(
        "my_function", SearchType.SIMILARITY, top_k=1
    )
    assert len(results) > 0
    assert "def my_function" in results[0].document.content
    assert results[0].document.metadata["type"] == "FunctionDef"

    # Cleanup
    watcher.stop()
    pipeline.stop()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_codebase_indexing_simulation(
    tmp_path, vector_store_fixture, embeddings_fixture
):
    """Simulate indexing a larger codebase structure."""
    # Setup structure
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main(): pass")
    (src / "utils.py").write_text("class Helper: pass")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "readme.md").write_text("# Documentation\nInfo here.")

    # Setup pipeline
    file_filter = FileFilter(allowed_extensions=[".py", ".md"])
    pipeline = IndexingPipeline(
        embeddings=embeddings_fixture,
        vector_store=vector_store_fixture,
        batch_size=2,  # Batching
    )
    pipeline.start()

    watcher = FileWatcher(
        watch_paths=[str(tmp_path)],
        file_filter=file_filter,
        indexing_pipeline=pipeline,
        recursive=True,
    )
    watcher.start()

    # Create more files NOW (simulating active development)
    (src / "extra.py").write_text("def extra(): pass")
    (docs / "api.md").write_text("## API\nDetails.")

    # Wait for indexing
    async def check_results():
        r1 = await vector_store_fixture.search("extra", SearchType.SIMILARITY, top_k=1)
        r2 = await vector_store_fixture.search("API", SearchType.SIMILARITY, top_k=1)
        return len(r1) > 0 and len(r2) > 0

    await wait_for_condition(
        check_results, timeout=5.0, error_msg="Full codebase simulation failed"
    )

    # Stop pipeline to flush remaining batch (if any)
    pipeline.stop()
    watcher.stop()
