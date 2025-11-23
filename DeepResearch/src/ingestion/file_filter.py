"""File filtering for indexing."""

import logging
from pathlib import Path

from gitignore_parser import parse_gitignore

logger = logging.getLogger(__name__)


class FileFilter:
    """Filters files based on gitignore patterns and file extensions."""

    def __init__(
        self,
        gitignore_path: str | None = None,
        mgrepignore_path: str | None = None,
        allowed_extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ):
        self.allowed_extensions = allowed_extensions or [".txt", ".py", ".md"]
        self.exclude_patterns = exclude_patterns or []

        # Parse .gitignore
        if gitignore_path:
            path = Path(gitignore_path)
            if path.exists():
                self.gitignore_matcher = parse_gitignore(gitignore_path)
            else:
                logger.warning(
                    f"Gitignore file not found at {gitignore_path}, patterns will be ignored."
                )
                self.gitignore_matcher = lambda x: False
        else:
            self.gitignore_matcher = lambda x: False

        # Parse .mgrepignore
        if mgrepignore_path:
            path = Path(mgrepignore_path)
            if path.exists():
                self.mgrepignore_matcher = parse_gitignore(mgrepignore_path)
            else:
                logger.warning(
                    f"Mgrepignore file not found at {mgrepignore_path}, patterns will be ignored."
                )
                self.mgrepignore_matcher = lambda x: False
        else:
            self.mgrepignore_matcher = lambda x: False

    def should_index(self, file_path: str) -> bool:
        """Check if file should be indexed."""
        path = Path(file_path)

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in str(path):
                return False

        # Check gitignore
        if self.gitignore_matcher(file_path):
            return False

        # Check mgrepignore
        if self.mgrepignore_matcher(file_path):
            return False

        # Check extension
        ext = path.suffix
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
