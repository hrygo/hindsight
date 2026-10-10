"""Client-selected MCP views compose with the real knowledge-page lifecycle.

The REST SDK prepares an isolated empty-bank page; the official MCP SDK exercises
its tool catalogue and CallToolResult protocol. Hindsight's REST wrapper cannot
observe tools/list, HTTP MCP session context, or isError. No engine/SQL access.

NOT_RUN in implementation verification. Page creation schedules a refresh even
for an empty bank; execution must use the isolated database and model/embedding
stubs provided by the system-test fixtures, not a production service.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from mcp.client import streamable_http as sdk_http
from mcp.client.session import ClientSession

pytestmark = pytest.mark.asyncio

READER = {
    "recall",
    "reflect",
    "search_knowledge_base",
    "get_knowledge_page",
    "get_knowledge_base_tree",
    "list_banks",
}
CURATOR = READER | {
    "list_mental_models",
    "get_mental_model",
    "create_mental_model",
    "update_mental_model",
    "refresh_mental_model",
    "list_memories",
    "get_memory",
    "update_memory",
    "invalidate_memory",
    "list_documents",
    "get_document",
    "list_operations",
    "get_operation",
    "list_tags",
    "get_bank",
    "create_knowledge_folder",
    "create_knowledge_page",
    "update_knowledge_node",
    "list_directives",
}
SOURCE_QUERY = "  Operations\n\t" + "部署🙂" * 90 + "  wiki  "


async def test_http_profiles_preserve_full_page_reads(client, hindsight_server, bank_id, settled):
    created = await client.knowledge_base.create_knowledge_page(
        bank_id, {"name": "Operations wiki", "source_query": SOURCE_QUERY, "tags": ["operations"]}
    )
    await settled(bank_id)
    full_rest = await client.knowledge_base.get_knowledge_page(bank_id, created.page_id)
    assert full_rest.description == SOURCE_QUERY
    search_rest = await client.knowledge_base.search_knowledge_base(bank_id, q="Operations wiki")
    assert [hit.id for hit in search_rest.results] == [created.page_id]

    client_module = getattr(sdk_http, "httpx2", httpx)

    async def browse(profile: str | None) -> None:
        headers = {"X-Bank-Id": bank_id}
        if profile is not None:
            headers["X-Hindsight-Tool-Profile"] = profile
        async with client_module.AsyncClient(headers=headers) as http_client:
            async with sdk_http.streamable_http_client(
                f"{hindsight_server.url}/mcp/", http_client=http_client
            ) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    names = {tool.name for tool in (await session.list_tools()).tools}
                    if profile == "reader":
                        assert names == READER
                    elif profile == "curator":
                        assert names == CURATOR
                    else:
                        assert len(names) == 39
                        assert CURATOR <= names
                        assert "delete_bank" in names

                    tree_result = await session.call_tool("get_knowledge_base_tree", {})
                    tree = json.loads(tree_result.content[0].text)
                    assert len(tree["roots"]) == 1
                    page = tree["roots"][0]
                    assert page["id"] == created.page_id
                    assert page["name"] == "Operations wiki"
                    assert page["tags"] == ["operations"]
                    assert "timestamp" in page
                    assert "is_stale" in page

                    search_result = await session.call_tool("search_knowledge_base", {"query": "Operations wiki"})
                    search = json.loads(search_result.content[0].text)
                    assert search["total"] == 1
                    assert [hit["id"] for hit in search["results"]] == [created.page_id]
                    hit = search["results"][0]
                    assert hit["name"] == "Operations wiki"
                    assert hit["snippet"] == search_rest.results[0].snippet
                    assert hit["score"] == search_rest.results[0].score == 1.0

                    if profile in {"reader", "curator"}:
                        expected = " ".join(SOURCE_QUERY.split())[:199] + "…"
                        assert page["description"] == expected
                        assert page["description_truncated"] is True
                        assert "get_knowledge_page" in page["read_hint"]
                        assert created.page_id in page["read_hint"]
                        assert hit["source_query"] == expected
                        assert hit["description_truncated"] is True
                    else:
                        assert page["description"] == SOURCE_QUERY
                        assert "description_truncated" not in page
                        assert hit["source_query"] == SOURCE_QUERY

                    full_result = await session.call_tool("get_knowledge_page", {"page_id": created.page_id})
                    full = json.loads(full_result.content[0].text)
                    assert full["id"] == created.page_id
                    assert full["description"] == SOURCE_QUERY
                    assert full["markdown"] == full_rest.markdown
                    if profile == "reader":
                        denied = await session.call_tool("delete_bank", {})
                        assert (denied.is_error if hasattr(denied, "is_error") else denied.isError) is True

    # Separate connections/session IDs, never a hot-switch of one session's header.
    await asyncio.gather(*(browse(profile) for profile in ("reader", "curator", "admin", None)))
    unchanged = await client.knowledge_base.get_knowledge_page(bank_id, created.page_id)
    assert unchanged.description == SOURCE_QUERY
    assert unchanged.markdown == full_rest.markdown
