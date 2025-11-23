"""Mgrep tool wrappers for PRIME and Pydantic AI."""

import asyncio
import json
import logging
from typing import Any

from pydantic_ai import RunContext

from DeepResearch.src.tools.base import ExecutionResult, ToolRunner, ToolSpec, registry
from DeepResearch.src.tools.mgrep_server import MgrepServer

logger = logging.getLogger(__name__)

_server_instance: MgrepServer | None = None


def get_mgrep_server() -> MgrepServer:
    """Get or initialize Mgrep server singleton."""
    global _server_instance  # noqa: PLW0603
    if _server_instance is None:
        _server_instance = MgrepServer.get_instance()
        _server_instance.start()
    return _server_instance


class MgrepSearchTool(ToolRunner):
    """Sync tool for PRIME system."""

    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="mgrep_search",
                description="Search codebase using natural language.",
                inputs={"query": "TEXT", "top_k": "INTEGER"},
                outputs={"results": "JSON", "count": "INTEGER", "success": "BOOLEAN"},
            )
        )

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        """Execute search."""
        query = params.get("query")
        if not query:
            return ExecutionResult(success=False, error="Query required")

        top_k = params.get("top_k", 5)

        try:
            server = get_mgrep_server()

            # Safely execute async search from sync context
            # If we are in a running event loop (e.g. inside an async agent calling this sync tool),
            # we must not block the loop. We offload to a separate thread.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            coro = server.search(query, top_k=top_k)

            if loop and loop.is_running():
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=1) as executor:
                    results = executor.submit(asyncio.run, coro).result()
            else:
                results = asyncio.run(coro)

            data = [
                {
                    "file_path": r.document.metadata.get("file_path", "unknown"),
                    "content": r.document.content,
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in results
            ]

            return ExecutionResult(
                success=True,
                data={"results": data, "count": len(data), "success": True},
            )

        except Exception as e:
            logger.error(f"Mgrep search failed: {e}")
            return ExecutionResult(success=False, error=str(e))


async def mgrep_search(ctx: RunContext[Any], query: str, top_k: int = 5) -> str:
    """
    Search codebase semantically (Pydantic AI compatible).

    Args:
        ctx: Context
        query: Natural language query
        top_k: Number of results

    Returns:
        JSON string with results
    """
    try:
        server = get_mgrep_server()
        results = await server.search(query, top_k=top_k)

        output = [
            {
                "file": r.document.metadata.get("file_path", "unknown"),
                "score": round(r.score, 3),
                "content_snippet": (
                    f"{r.document.content[:200]}..."
                    if len(r.document.content) > 200
                    else r.document.content
                ),
            }
            for r in results
        ]

        return json.dumps(output, indent=2)
    except Exception as e:
        return f"Error searching codebase: {e}"


def register_mgrep_tools() -> None:
    """Register with global registry."""
    registry.register("mgrep_search", MgrepSearchTool)


register_mgrep_tools()
