from unittest.mock import patch

import pytest

from DeepResearch.src.ingestion.file_filter import FileFilter


def test_file_filter_exclude_patterns():
    """Test that exclude_patterns are respected."""
    file_filter = FileFilter(exclude_patterns=["node_modules", "venv", ".git"])

    with patch.object(FileFilter, "_is_text_file", return_value=True):
        assert not file_filter.should_index("/project/node_modules/package.json")
        assert not file_filter.should_index("/project/venv/bin/python")
        assert not file_filter.should_index("/project/.git/HEAD")
        assert file_filter.should_index("/project/src/main.py")
