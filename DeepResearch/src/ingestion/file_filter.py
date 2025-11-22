"""File filtering for indexing."""

from pathlib import Path

from gitignore_parser import parse_gitignore


class FileFilter:
    """Filters files based on gitignore patterns and file extensions."""

    def __init__(
        self,
        gitignore_path: str | None = None,
        allowed_extensions: list[str] | None = None,
    ):
        self.allowed_extensions = allowed_extensions or [".txt", ".py", ".md"]

        if gitignore_path and Path(gitignore_path).exists():
            self.gitignore_matcher = parse_gitignore(gitignore_path)
        else:
            self.gitignore_matcher = lambda x: False

    def should_index(self, file_path: str) -> bool:
        """Check if file should be indexed."""
        # Check gitignore
        if self.gitignore_matcher(file_path):
            return False

        # Check extension
        ext = Path(file_path).suffix
        if ext not in self.allowed_extensions:
            return False

        # Check binary
        if not self._is_text_file(file_path):
            return False

        return True

    @staticmethod
    def _is_text_file(file_path: str) -> bool:
        """Check if file is text (not binary)."""
        try:
            # Try reading start of file as UTF-8
            with open(file_path, encoding="utf-8") as f:
                f.read(512)
            return True
        except (UnicodeDecodeError, FileNotFoundError):
            return False
