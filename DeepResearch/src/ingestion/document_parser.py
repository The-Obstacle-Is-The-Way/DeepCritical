"""Document parsing."""

import ast
import logging
import re
from pathlib import Path
from typing import Any

from DeepResearch.src.datatypes.rag import Document

logger = logging.getLogger(__name__)


class DocumentParser:
    """Abstract base class for parsers (implied interface)."""

    def parse(self, file_path: str) -> list[Document]:
        raise NotImplementedError


class PythonParser(DocumentParser):
    """Parse Python files using AST."""

    def parse(self, file_path: str) -> list[Document]:
        """Parse Python file into semantic chunks (functions, classes)."""
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            logger.error(f"Error reading Python file {file_path}: {e}")
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

        # Collect relevant nodes first
        nodes = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)

        # Sort by line number to ensure deterministic order
        nodes.sort(key=lambda x: x.lineno)

        for node in nodes:
            # Extract source for this node
            if chunk := ast.get_source_segment(source, node):
                start_line = node.lineno
                end_line = node.end_lineno or start_line

                doc_id = f"{file_path}::{node.name}::{start_line}"
                documents.append(
                    Document(
                        id=doc_id,
                        content=chunk,
                        metadata={
                            "file_path": file_path,
                            "type": node.__class__.__name__,
                            "name": node.name,
                            "line": node.lineno,
                            "line_range": (start_line, end_line),
                            "start_line": start_line,
                            "end_line": end_line,
                        },
                    )
                )

        return documents


class MarkdownParser(DocumentParser):
    """Parse Markdown files by sections (headers)."""

    def parse(self, file_path: str) -> list[Document]:
        """Parse markdown file into sections."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading Markdown file {file_path}: {e}")
            return []

        sections = self._split_by_headers(content)
        documents = []

        for i, section in enumerate(sections):
            doc_id = f"{file_path}::section{i}"
            documents.append(
                Document(
                    id=doc_id,
                    content=section,
                    metadata={
                        "file_path": file_path,
                        "type": "markdown_section",
                        "section_index": i,
                    },
                )
            )

        return documents

    @staticmethod
    def _split_by_headers(text: str) -> list[str]:
        """Split markdown by headers (##, ###, etc.)."""
        sections = []
        current_section = []

        for line in text.split("\n"):
            if re.match(r"^#{1,6} ", line):  # Header line
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        # Filter empty sections
        return [s for s in sections if s.strip()]


class PlainTextParser(DocumentParser):
    """Parse plain text files."""

    def parse(self, file_path: str) -> list[Document]:
        """Parse file as plain text."""
        try:
            with open(file_path, encoding="utf-8") as f:
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
            logger.error(f"Error reading text file {file_path}: {e}")
            return []


class ParserFactory:
    """Factory for selecting parser."""

    @staticmethod
    def get_parser(file_path: str) -> DocumentParser:
        """Get parser for file based on extension."""
        ext = Path(file_path).suffix

        if ext == ".py":
            return PythonParser()
        return MarkdownParser() if ext == ".md" else PlainTextParser()
