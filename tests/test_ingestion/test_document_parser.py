from pathlib import Path

import pytest

from DeepResearch.src.ingestion.document_parser import ParserFactory, PythonParser


def test_python_parser_extracts_functions(tmp_path):
    """Test that PythonParser extracts function definitions."""
    # Create Python file
    py_file = tmp_path / "test.py"
    py_file.write_text('''
def foo():
    """Docstring for foo."""
    return 42

def bar():
    """Docstring for bar."""
    return 100
''')

    parser = PythonParser()
    documents = parser.parse(str(py_file))

    assert len(documents) == 2
    docs_by_name = {d.metadata["name"]: d for d in documents}
    assert "foo" in docs_by_name
    assert "bar" in docs_by_name
    assert "return 42" in docs_by_name["foo"].content
    assert "return 100" in docs_by_name["bar"].content

    # Check line ranges
    assert docs_by_name["foo"].metadata["line_range"] == (2, 4)
    assert docs_by_name["bar"].metadata["line_range"] == (6, 8)


def test_python_parser_extracts_classes(tmp_path):
    """Test that PythonParser extracts class definitions."""
    py_file = tmp_path / "test.py"
    py_file.write_text('''
class MyClass:
    """Docstring for MyClass."""

    def method(self):
        return "hello"
''')

    parser = PythonParser()
    documents = parser.parse(str(py_file))

    # Should extract both class and method
    # Implementation detail: Does it extract method separately?
    # The plan says "for node in ast.walk(tree): if isinstance(node, (ast.FunctionDef, ast.ClassDef))"
    # So yes, it will extract MyClass (full) AND method (inside).

    types = [doc.metadata["type"] for doc in documents]
    assert "ClassDef" in types
    assert "FunctionDef" in types

    # Check class content contains method
    class_doc = next(d for d in documents if d.metadata["type"] == "ClassDef")
    assert "class MyClass:" in class_doc.content
    assert "def method(self):" in class_doc.content


def test_python_parser_invalid_syntax(tmp_path):
    """Test that PythonParser falls back to plain text for invalid syntax."""
    py_file = tmp_path / "test.py"
    py_file.write_text("def foo(:\n    invalid syntax")

    parser = PythonParser()
    documents = parser.parse(str(py_file))

    # Should fall back to plain text
    assert len(documents) == 1
    assert documents[0].metadata["type"] == "python_invalid"
    assert documents[0].content == "def foo(:\n    invalid syntax"


def test_factory_selects_python_parser(tmp_path):
    """Test that factory returns PythonParser for .py files."""
    py_file = tmp_path / "test.py"
    parser = ParserFactory.get_parser(str(py_file))
    assert isinstance(parser, PythonParser)


def test_markdown_parser_splits_by_headers(tmp_path):
    """Test that MarkdownParser splits by section headers."""
    md_file = tmp_path / "test.md"
    md_file.write_text("""
# Introduction
This is the intro.

## Section 1
Content for section 1.

## Section 2
Content for section 2.
""")

    from DeepResearch.src.ingestion.document_parser import MarkdownParser

    parser = MarkdownParser()
    documents = parser.parse(str(md_file))

    assert len(documents) == 3
    assert "# Introduction" in documents[0].content
    assert "## Section 1" in documents[1].content
    assert "## Section 2" in documents[2].content
    assert documents[0].metadata["type"] == "markdown_section"
    assert documents[1].metadata["section_index"] == 1


def test_factory_selects_markdown_parser(tmp_path):
    """Test that factory returns MarkdownParser for .md files."""
    md_file = tmp_path / "test.md"

    from DeepResearch.src.ingestion.document_parser import MarkdownParser

    parser = ParserFactory.get_parser(str(md_file))
    assert isinstance(parser, MarkdownParser)


def test_plain_text_parser_skips_empty_file(tmp_path):
    """PlainTextParser.parse should return no documents for an empty file."""
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("")

    from DeepResearch.src.ingestion.document_parser import PlainTextParser

    parser = PlainTextParser()
    documents = parser.parse(str(empty_txt))

    assert isinstance(documents, list)
    assert len(documents) == 0


def test_plain_text_parser_parses_text_file(tmp_path):
    """PlainTextParser.parse should return a single document with correct metadata."""
    txt_file = tmp_path / "sample.txt"
    content = "First line\nSecond line\nThird line"
    txt_file.write_text(content)

    from DeepResearch.src.ingestion.document_parser import PlainTextParser

    parser = PlainTextParser()
    documents = parser.parse(str(txt_file))

    assert len(documents) == 1
    doc = documents[0]

    # Content should include the original text
    assert "First line" in doc.content
    assert "Second line" in doc.content
    assert "Third line" in doc.content

    # Basic metadata expectations; adjust if PlainTextParser uses different keys
    assert doc.metadata.get("type") == "text"
    assert doc.metadata.get("file_path") == str(txt_file)


def test_factory_selects_plain_text_parser_for_non_code_markdown_extensions(tmp_path):
    """ParserFactory should return PlainTextParser for non-.py/.md files (e.g., .txt)."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Some plain text content")

    from DeepResearch.src.ingestion.document_parser import PlainTextParser

    parser = ParserFactory.get_parser(str(txt_file))
    assert isinstance(parser, PlainTextParser)
