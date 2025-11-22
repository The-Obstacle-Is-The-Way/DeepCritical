"""Document parsing."""
import ast
from pathlib import Path
from typing import Any
from DeepResearch.src.datatypes.rag import Document


class DocumentParser:
    """Abstract base class for parsers (implied interface)."""

    def parse(self, file_path: str) -> list[Document]:
        raise NotImplementedError


class PythonParser(DocumentParser):
    """Parse Python files using AST."""

    def parse(self, file_path: str) -> list[Document]:
        """Parse Python file into semantic chunks (functions, classes)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fall back to treating as plain text
            return [
                Document(
                    id=str(Path(file_path).resolve()),
                    content=source,
                    metadata={"file_path": file_path, "type": "python_invalid"},
                )
            ]

        documents = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Extract source for this node
                chunk = ast.get_source_segment(source, node)
                if chunk:
                    doc_id = f"{file_path}::{node.name}::{node.lineno}"
                    documents.append(
                        Document(
                            id=doc_id,
                            content=chunk,
                            metadata={
                                "file_path": file_path,
                                "type": node.__class__.__name__,
                                "name": node.name,
                                "line": node.lineno,
                            },
                        )
                    )

        # If no semantic units found but file has content, index as module
        if not documents and source.strip():
             # Could index whole file as module
             # But for now, let's return empty list or whole file?
             # Plan implies chunking. If no functions/classes, maybe just file?
             pass
             
        return documents


class PlainTextParser(DocumentParser):
    """Parse plain text files."""

    def parse(self, file_path: str) -> list[Document]:
        """Parse file as plain text."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple check to avoid indexing empty files
            if not content.strip():
                return []

            return [
                Document(
                    id=file_path,
                    content=content,
                    metadata={"file_path": file_path, "type": "text"},
                )
            ]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []


class ParserFactory:
    """Factory for selecting parser."""

    @staticmethod
    def get_parser(file_path: str) -> DocumentParser:
        """Get parser for file based on extension."""
        ext = Path(file_path).suffix

        if ext == ".py":
            return PythonParser()
        else:
            return PlainTextParser()

