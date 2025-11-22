"""Document parsing (plain text only for Phase 4A)."""

from pathlib import Path

from DeepResearch.src.datatypes.rag import Document


class PlainTextParser:
    """Parse plain text files."""

    @staticmethod
    def parse(file_path: str) -> list[Document]:
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
            print(f"Error reading {file_path}: {e}")
            return []


class ParserFactory:
    """Factory for selecting parser (always plain text in Phase 4A)."""

    @staticmethod
    def get_parser(file_path: str) -> PlainTextParser:
        """Get parser for file based on extension."""
        _ = file_path  # Unused for now
        return PlainTextParser()
