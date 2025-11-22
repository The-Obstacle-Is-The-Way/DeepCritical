import pytest
from pathlib import Path
from DeepResearch.src.ingestion.document_parser import PythonParser, ParserFactory

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
    assert documents[0].metadata['name'] == 'foo'
    assert documents[1].metadata['name'] == 'bar'
    assert 'return 42' in documents[0].content
    assert 'return 100' in documents[1].content


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
    
    types = [doc.metadata['type'] for doc in documents]
    assert 'ClassDef' in types
    assert 'FunctionDef' in types
    
    # Check class content contains method
    class_doc = next(d for d in documents if d.metadata['type'] == 'ClassDef')
    assert 'class MyClass:' in class_doc.content
    assert 'def method(self):' in class_doc.content


def test_python_parser_invalid_syntax(tmp_path):
    """Test that PythonParser falls back to plain text for invalid syntax."""
    py_file = tmp_path / "test.py"
    py_file.write_text('def foo(:\n    invalid syntax')

    parser = PythonParser()
    documents = parser.parse(str(py_file))

    # Should fall back to plain text
    assert len(documents) == 1
    assert documents[0].metadata['type'] == 'python_invalid'
    assert documents[0].content == 'def foo(:\n    invalid syntax'

def test_factory_selects_python_parser(tmp_path):
    """Test that factory returns PythonParser for .py files."""
    py_file = tmp_path / "test.py"
    parser = ParserFactory.get_parser(str(py_file))
    assert isinstance(parser, PythonParser)
