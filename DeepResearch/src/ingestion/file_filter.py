"""File filtering for indexing."""

from pathlib import Path


class FileFilter:
    """Filters files based on extension (gitignore support in Phase 4B)."""

    def __init__(self, allowed_extensions: list[str] | None = None):
        self.allowed_extensions = allowed_extensions or [".txt", ".py", ".md"]

    def should_index(self, file_path: str) -> bool:
        """Check if file should be indexed."""
        ext = Path(file_path).suffix
        return ext in self.allowed_extensions
