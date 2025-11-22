"""File watcher for automatic code indexing."""

import asyncio

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

    def start(self) -> None:
        """Start watching file system."""
        for path in self.watch_paths:
            self.observer.schedule(
                self.event_handler, path=path, recursive=self.recursive
            )
        self.observer.start()

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
            if not event.is_directory and self.file_filter.should_index(src_path):
                self.pipeline.enqueue_file(src_path)
    
        def on_modified(self, event: FileSystemEvent) -> None:
            src_path = str(event.src_path)
            if not event.is_directory and self.file_filter.should_index(src_path):
                self.pipeline.enqueue_file(src_path)
    
        def on_deleted(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                # remove_file is async, but we are in a sync callback
                # We can fire-and-forget using asyncio.run or better yet,
                # add to a queue for deletion same as creation.
                # For now, the pipeline.remove_file is async.
                # Let's update pipeline to have a sync wrapper or queue it.
                # BUT pipeline.remove_file assumes vector store is available.
                
                # Simplest for MVP: Use asyncio.run here? No, event handler runs in Observer thread.
                # Better: Enqueue deletion task? 
                # For Phase 4A MVP, we only tested creation/modification.
                # The implementation uses pipeline.remove_file(event.src_path)
                # I'll implement a sync wrapper in the event handler for now.
                try:
                     src_path = str(event.src_path)
                     asyncio.run(self.pipeline.remove_file(src_path))
                except Exception as e:
                    print(f"Error removing file {event.src_path}: {e}")
