"""File watcher for automatic code indexing."""

import logging

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from DeepResearch.src.ingestion.file_filter import FileFilter
from DeepResearch.src.ingestion.indexing_pipeline import IndexingPipeline

logger = logging.getLogger(__name__)


class FileWatcher:
    """Watches directories for file changes and triggers indexing."""

    def __init__(
        self,
        watch_paths: list[str],
        file_filter: FileFilter,
        indexing_pipeline: IndexingPipeline,
        recursive: bool = True,
    ):
        self.watch_paths = watch_paths
        self.file_filter = file_filter
        self.indexing_pipeline = indexing_pipeline
        self.recursive = recursive
        self.observer = Observer()
        self.event_handler = CodebaseEventHandler(
            file_filter=file_filter, pipeline=indexing_pipeline
        )

    def start(self, initial_scan: bool = True) -> None:
        """Start watching file system.

        Args:
            initial_scan: If True, crawl and index existing files before watching.
        """
        if initial_scan:
            self._initial_crawl()

        for path in self.watch_paths:
            self.observer.schedule(
                self.event_handler, path=path, recursive=self.recursive
            )
        self.observer.start()

    def _initial_crawl(self) -> None:
        """Index all existing files in watch paths."""
        from pathlib import Path

        logger.info("Starting initial crawl of watch paths...")
        count = 0
        for watch_path in self.watch_paths:
            path_obj = Path(watch_path)
            if not path_obj.exists():
                logger.warning(f"Watch path does not exist: {watch_path}")
                continue

            # Choose glob pattern based on recursive flag
            iterator = path_obj.rglob("*") if self.recursive else path_obj.glob("*")

            for file_path in iterator:
                if file_path.is_file() and self.file_filter.should_index(
                    str(file_path)
                ):
                    self.indexing_pipeline.enqueue_file(str(file_path))
                    count += 1

        logger.info(f"Initial crawl complete. Enqueued {count} files.")

    def stop(self) -> None:
        """Stop watching file system."""
        self.observer.stop()
        self.observer.join()


class CodebaseEventHandler(FileSystemEventHandler):
    """Handles file system events for indexing."""

    def __init__(self, file_filter: FileFilter, pipeline: IndexingPipeline):
        self.file_filter = file_filter
        self.pipeline = pipeline

    def on_created(self, event: FileSystemEvent) -> None:
        src_path = str(event.src_path)
        logger.info(f"on_created event for: {src_path}")
        if not event.is_directory and self.file_filter.should_index(src_path):
            self.pipeline.enqueue_file(src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        src_path = str(event.src_path)
        if not event.is_directory and self.file_filter.should_index(src_path):
            self.pipeline.enqueue_file(src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            try:
                src_path = str(event.src_path)
                self.pipeline.enqueue_deletion(src_path)
            except Exception as e:
                logger.error(f"Error removing file {event.src_path}: {e}")
