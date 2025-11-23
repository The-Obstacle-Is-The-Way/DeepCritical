from DeepResearch.src.ingestion.file_filter import FileFilter


def test_gitignore_excludes_patterns(tmp_path):
    """Test that FileFilter respects .gitignore patterns."""
    # Create .gitignore
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__pycache__/\n*.pyc\n.venv/\n")

    # Create FileFilter
    file_filter = FileFilter(gitignore_path=str(gitignore))

    # Test exclusions
    assert not file_filter.should_index(str(tmp_path / "__pycache__/foo.pyc"))
    assert not file_filter.should_index(str(tmp_path / "test.pyc"))
    assert not file_filter.should_index(str(tmp_path / ".venv/lib/foo.py"))

    # Test allowed
    # Need to match allowed extensions (.txt default in filter?)
    # The default in Phase 4A was ['.txt', '.py', '.md']
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("print('hello')")
    assert file_filter.should_index(str(tmp_path / "src/main.py"))


def test_binary_file_detection(tmp_path):
    """Test that binary files are excluded."""
    # Create binary file
    binary_file = tmp_path / "test.bin"
    binary_file.write_bytes(b"\xff\xff\xff\xff")  # Invalid UTF-8

    # Create text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("Hello")

    file_filter = FileFilter(allowed_extensions=[".bin", ".txt"])

    assert not file_filter.should_index(str(binary_file))
    assert file_filter.should_index(str(text_file))


def test_mgrepignore_filtering(tmp_path):
    """Test that .mgrepignore patterns are respected."""
    mgrepignore = tmp_path / ".mgrepignore"
    mgrepignore.write_text("*.log\ntemp/\n")

    # Create test files
    (tmp_path / "debug.log").write_text("log")
    (tmp_path / "temp").mkdir()
    (tmp_path / "temp/data.txt").write_text("data")
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src/main.py").write_text("code")

    file_filter = FileFilter(mgrepignore_path=str(mgrepignore))

    assert not file_filter.should_index(str(tmp_path / "debug.log"))
    assert not file_filter.should_index(str(tmp_path / "temp/data.txt"))
    assert file_filter.should_index(str(tmp_path / "src/main.py"))
