"""HTTP tool profiles: ceilings, bounded navigation and real SDK session isolation.

No real engine, database or model is used. The loopback tests exercise the installed
MCP SDK's stateful task dispatch rather than assuming request context survives it.
"""

import asyncio
import copy
import json
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import uvicorn
from fastmcp import FastMCP
from mcp.client import streamable_http as sdk_http
from mcp.client.session import ClientSession
from starlette.applications import Starlette

from hindsight_api.api.mcp import (
    MCPMiddleware,
    _current_bank_id,
    _current_tool_profile,
    _get_mcp_tools,
    _get_session_manager,
    create_mcp_server,
    get_current_tool_profile,
)
from hindsight_api.mcp_profiles import TOOL_PROFILES
from hindsight_api.mcp_tools import (
    _ALL_TOOLS,
    MCPToolsConfig,
    _apply_bank_tool_filtering,
    register_mcp_tools,
)

READER = frozenset(
    {
        "recall",
        "reflect",
        "search_knowledge_base",
        "get_knowledge_page",
        "get_knowledge_base_tree",
        "list_banks",
    }
)
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
LONG_QUERY = "\n  中文🙂\t" + "知" * 230 + "  tail\n"
PAGE = {
    "id": "kp-1",
    "kind": "page",
    "name": "Deploys",
    "parent_id": "kf-1",
    "managed": False,
    "mental_model_id": "mm-1",
    "source_query": LONG_QUERY,
    "tags": ["infra"],
    "last_refreshed_at": "2026-10-01T00:00:00Z",
    "is_stale": True,
    "content": "# Complete document\n\n" + "Body " * 100,
}


@pytest.fixture
def memory() -> MagicMock:
    engine = MagicMock()
    engine.resolve_bank_alias = AsyncMock(side_effect=lambda bank_id, **_: bank_id)
    engine._tenant_extension.authenticate_mcp = AsyncMock(return_value=None)
    engine._operation_validator = None
    engine.audit_logger.action_allowed.return_value = False
    engine._config_resolver.get_bank_config = AsyncMock(return_value={"mcp_enabled_tools": None})
    engine.list_knowledge_nodes = AsyncMock(
        return_value=[
            {"id": "kf-1", "kind": "folder", "name": "Operations", "parent_id": None},
            copy.deepcopy(PAGE),
        ]
    )
    engine.search_knowledge_pages = AsyncMock(
        return_value=[
            {**copy.deepcopy(PAGE), "snippet": "relevant snippet", "score": 0.9},
            {**copy.deepcopy(PAGE), "id": "kp-2", "snippet": "second snippet", "score": 0.8},
        ]
    )
    engine.get_knowledge_page = AsyncMock(return_value=copy.deepcopy(PAGE))
    engine.delete_bank = AsyncMock(return_value={"deleted_memories": 0})
    return engine


@asynccontextmanager
async def profile_context(profile: str | None, bank: str | None = "session-bank") -> AsyncIterator[None]:
    profile_token = _current_tool_profile.set(profile)
    bank_token = _current_bank_id.set(bank)
    try:
        yield
    finally:
        _current_tool_profile.reset(profile_token)
        _current_bank_id.reset(bank_token)


def test_profile_sets_are_exact_and_immutable() -> None:
    assert TOOL_PROFILES["reader"] == READER
    assert TOOL_PROFILES["curator"] == CURATOR
    assert len(READER) == 6
    assert len(CURATOR) == 25
    assert len(_ALL_TOOLS) == 39
    assert CURATOR <= _ALL_TOOLS
    assert TOOL_PROFILES["admin"] is None
    with pytest.raises(TypeError):
        TOOL_PROFILES["reader"] = frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile,expected", [(None, _ALL_TOOLS), ("admin", _ALL_TOOLS), ("reader", READER), ("curator", CURATOR)]
)
async def test_tools_list_exact_profiles(memory: MagicMock, profile: str | None, expected: frozenset[str]) -> None:
    mcp = create_mcp_server(memory)
    async with profile_context(profile):
        assert {t.name for t in await mcp.list_tools()} == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["reader", "curator"])
@pytest.mark.parametrize("bank_tools", [None, [], ["retain"], ["recall", "retain"]])
async def test_profile_intersects_bank_and_no_bank_early_return(memory: MagicMock, profile: str, bank_tools) -> None:
    memory._config_resolver.get_bank_config.return_value = {"mcp_enabled_tools": bank_tools}
    mcp = create_mcp_server(memory)
    ceiling = TOOL_PROFILES[profile]
    async with profile_context(profile):
        expected = ceiling if bank_tools is None else ceiling & set(bank_tools)
        assert {t.name for t in await mcp.list_tools()} == expected
        assert await mcp.get_tool("retain") is None
    async with profile_context(profile, bank=None):
        assert {t.name for t in await mcp.list_tools()} == ceiling


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["expand", "raise", "deny"])
async def test_validator_never_loses_profile_ceiling(memory: MagicMock, behavior: str) -> None:
    validator = MagicMock()
    if behavior == "raise":
        validator.filter_mcp_tools = AsyncMock(side_effect=RuntimeError("offline policy backend"))
    elif behavior == "deny":
        validator.filter_mcp_tools = AsyncMock(return_value=frozenset())
    else:
        validator.filter_mcp_tools = AsyncMock(side_effect=lambda bank, context, tools: tools | _ALL_TOOLS)
    memory._operation_validator = validator
    mcp = create_mcp_server(memory)
    async with profile_context("reader"):
        assert {t.name for t in await mcp.list_tools()} == (set() if behavior == "deny" else READER)
        assert await mcp.get_tool("retain") is None


class Extension:
    def register_tools(self, mcp: FastMCP, memory: MagicMock) -> None:
        @mcp.tool()
        async def extension_write() -> str:
            return "extension write executed"


@pytest.mark.asyncio
async def test_final_extension_registration_filtered_v3(memory: MagicMock) -> None:
    with patch("hindsight_api.api.mcp.load_extension", return_value=Extension()):
        mcp = create_mcp_server(memory)
    async with profile_context("reader"):
        assert {t.name for t in await mcp.list_tools()} == READER
        assert await mcp.get_tool("extension_write") is None
    async with profile_context("admin"):
        assert "extension_write" in {t.name for t in await mcp.list_tools()}
        assert await mcp.get_tool("extension_write") is not None


@pytest.mark.asyncio
async def test_final_extension_registration_filtered_v2(memory: MagicMock) -> None:
    # Exercise the v2 manager surface without installing/changing dependencies.
    extension_run = AsyncMock(return_value="write")
    extension = SimpleNamespace(name="extension_write", run=extension_run)
    recall = SimpleNamespace(name="recall", run=AsyncMock(return_value="read"))
    manager = SimpleNamespace(_tools={"recall": recall, "extension_write": extension})
    manager.get_tools = AsyncMock(side_effect=lambda: dict(manager._tools))
    mcp = SimpleNamespace(_tool_manager=manager)
    config = MCPToolsConfig(bank_id_resolver=lambda: "bank", tool_profile_resolver=get_current_tool_profile)
    _apply_bank_tool_filtering(mcp, memory, config)
    async with profile_context("reader"):
        assert set(await manager.get_tools()) == {"recall"}
        with pytest.raises(ValueError, match="not enabled"):
            await extension.run({})
        assert await recall.run({}) == "read"
    async with profile_context("admin"):
        assert set(await manager.get_tools()) == {"recall", "extension_write"}
        assert await extension.run({}) == "write"
    extension_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_config_empty_tools_keeps_existing_registration_semantics(memory: MagicMock) -> None:
    mcp = FastMCP("empty-config-legacy")
    register_mcp_tools(mcp, memory, MCPToolsConfig(bank_id_resolver=lambda: None, tools=set()))
    assert {t.name for t in await mcp.list_tools()} == _ALL_TOOLS


async def downstream(scope, receive, send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [("X-Hindsight-Tool-Profile", "unknown")],
        [("X-Hindsight-Tool-Profile", "")],
        [("X-Hindsight-Tool-Profile", " \t")],
        [("X-Hindsight-Tool-Profile", "reader"), ("x-hindsight-tool-profile", "admin")],
        [("X-Hindsight-Tool-Profile", "reader,admin")],
    ],
)
@pytest.mark.parametrize("method", ["POST", "GET"])
async def test_invalid_headers_error_before_dispatch(memory: MagicMock, headers, method: str) -> None:
    middleware = MCPMiddleware(downstream, memory, multi_bank_app=downstream, single_bank_app=downstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as client:
        response = await client.request(method, "/mcp/", headers=headers)
    assert response.status_code == 400
    assert "X-Hindsight-Tool-Profile" in response.json()["error"]


@pytest.mark.asyncio
async def test_profile_reset_on_exception_and_non_mcp_unchanged(memory: MagicMock) -> None:
    async def broken(scope, receive, send) -> None:
        assert get_current_tool_profile() == "reader"
        raise RuntimeError("downstream failure")

    middleware = MCPMiddleware(downstream, memory, multi_bank_app=broken, single_bank_app=broken)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as client:
        with pytest.raises(RuntimeError, match="downstream failure"):
            await client.post("/mcp/", headers={"X-Bank-Id": "bank", "X-Hindsight-Tool-Profile": "reader"})
        assert get_current_tool_profile() is None
        response = await client.get("/elsewhere", headers={"X-Hindsight-Tool-Profile": "bad"})
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "admin", "reader", "curator"])
@pytest.mark.parametrize("include_bank_id", [True, False])
async def test_navigation_bounded_only_for_compact_profiles(
    memory: MagicMock, profile: str | None, include_bank_id: bool
) -> None:
    mcp = create_mcp_server(memory, multi_bank=include_bank_id)
    tools = _get_mcp_tools(mcp)
    originals = copy.deepcopy(memory.list_knowledge_nodes.return_value)
    search_originals = copy.deepcopy(memory.search_knowledge_pages.return_value)
    async with profile_context(profile):
        tree = await tools["get_knowledge_base_tree"].fn()
        hits = await tools["search_knowledge_base"].fn(query="deploy")
        full = await tools["get_knowledge_page"].fn(page_id="kp-1")
    if include_bank_id:
        tree, hits, full = (json.loads(value) for value in (tree, hits, full))
    page = tree["roots"][0]["children"][0]
    if profile in {"reader", "curator"}:
        assert len(page["description"]) == 200
        assert page["description"].endswith("…")
        assert not any(char in page["description"] for char in "\n\t")
        assert page["description_truncated"] is True
        assert "get_knowledge_page" in page["read_hint"]
        assert "kp-1" in page["read_hint"]
        for hit in hits["results"]:
            assert len(hit["source_query"]) == 200
            assert hit["description_truncated"] is True
            assert "get_knowledge_page" in hit["read_hint"]
    else:
        assert page["description"] == LONG_QUERY
        assert "description_truncated" not in page
        assert hits["results"] == search_originals
    assert page["id"] == "kp-1"
    assert page["name"] == "Deploys"
    assert page["tags"] == ["infra"]
    assert page["timestamp"] == PAGE["last_refreshed_at"]
    assert page["is_stale"] is True
    assert hits["total"] == 2
    assert [hit["id"] for hit in hits["results"]] == ["kp-1", "kp-2"]
    assert [hit["snippet"] for hit in hits["results"]] == ["relevant snippet", "second snippet"]
    assert full["description"] == LONG_QUERY
    assert PAGE["content"].rstrip() in full["markdown"]
    assert memory.list_knowledge_nodes.return_value == originals
    assert memory.search_knowledge_pages.return_value == search_originals


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_query,expected,truncated",
    [
        ("  alpha\n\t beta  ", "alpha beta", False),
        ("界" * 200, "界" * 200, False),
        ("🙂" * 201, "🙂" * 199 + "…", True),
        ("", "", False),
        (None, None, False),
    ],
)
async def test_unicode_and_whitespace_boundaries(memory: MagicMock, source_query, expected, truncated: bool) -> None:
    memory.list_knowledge_nodes.return_value = [{**PAGE, "parent_id": None, "source_query": source_query}]
    mcp = create_mcp_server(memory)
    async with profile_context("reader"):
        tree = json.loads(await _get_mcp_tools(mcp)["get_knowledge_base_tree"].fn())
    node = tree["roots"][0]
    assert node["description"] == expected
    assert node["description_truncated"] is truncated


@dataclass
class ProfileTestServer:
    url: str
    middleware: MCPMiddleware
    http_app: Starlette


@asynccontextmanager
async def loopback_server(memory: MagicMock, stateless: bool) -> AsyncIterator[ProfileTestServer]:
    """A real HTTP server, with only mocked engine methods and an ephemeral port."""
    mcp = create_mcp_server(memory)
    http_app = mcp.http_app(path="/", stateless_http=stateless)

    @asynccontextmanager
    async def lifespan(app) -> AsyncIterator[None]:
        async with http_app.router.lifespan_context(http_app):
            yield

    app = Starlette(lifespan=lifespan)
    middleware = MCPMiddleware(app, memory=memory, multi_bank_app=http_app, single_bank_app=http_app)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(middleware, log_level="error", lifespan="on", ws="none"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    task.result()
                    raise AssertionError("test HTTP server failed to start")
                await asyncio.sleep(0.01)
        yield ProfileTestServer(f"http://127.0.0.1:{port}/mcp/", middleware, http_app)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        sock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stateless", [False, True])
async def test_real_sdk_http_concurrent_profiles_and_session_inheritance(memory: MagicMock, stateless: bool) -> None:
    # MCP 2 uses httpx2 and two transport streams; MCP 1 uses httpx and three.
    # Use the SDK's own client implementation, without changing either install.
    client_module = getattr(sdk_http, "httpx2", httpx)
    async with loopback_server(memory, stateless) as test_server:
        url = test_server.url
        barrier = asyncio.Barrier(2)

        async def exercise(profile: str) -> None:
            async with client_module.AsyncClient(
                headers={
                    "X-Bank-Id": "session-bank",
                    "X-Hindsight-Tool-Profile": profile,
                }
            ) as client:
                async with sdk_http.streamable_http_client(url, http_client=client) as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await session.initialize()
                        await barrier.wait()
                        expected = READER if profile == "reader" else _ALL_TOOLS
                        for _ in range(3):
                            assert {t.name for t in (await session.list_tools()).tools} == expected
                            result = await session.call_tool("get_knowledge_base_tree", {})
                            assert not result.isError
                            body = json.loads(result.content[0].text)
                            page = body["roots"][0]["children"][0]
                            assert ("description_truncated" in page) == (profile == "reader")
                        denied = await session.call_tool("delete_bank", {})
                        if profile == "reader":
                            assert denied.isError is True
                        # No-header requests restore the initialization view even
                        # when the SDK propagates context per message (MCP 2).
                        if not stateless:
                            del client.headers["X-Hindsight-Tool-Profile"]
                            assert {t.name for t in (await session.list_tools()).tools} == expected
                            if profile == "reader":
                                assert (await session.call_tool("delete_bank", {})).isError is True

        async with asyncio.timeout(30):
            await asyncio.gather(exercise("reader"), exercise("admin"))
    assert get_current_tool_profile() is None
    assert memory._config_resolver.get_bank_config.await_count > 0
    assert memory.delete_bank.await_count == 1
    assert len(test_server.middleware._profile_sessions) == 0


async def initialize_request(client: httpx.AsyncClient, url: str, profile: str | None) -> httpx.Response:
    headers = {"X-Bank-Id": "session-bank"}
    if profile is not None:
        headers["X-Hindsight-Tool-Profile"] = profile
    return await client.post(
        url,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "profile-test", "version": "1"},
            },
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_profile", [None, "reader", "admin"])
async def test_stateful_conflicts_unknown_binding_and_cleanup(memory: MagicMock, initial_profile: str | None) -> None:
    async with loopback_server(memory, False) as test_server:
        async with httpx.AsyncClient() as client:
            response = await initialize_request(client, test_server.url, initial_profile)
            assert response.status_code == 200
            sid = response.headers["mcp-session-id"]
            manager = _get_session_manager(test_server.http_app)
            transport = manager._server_instances[sid]
            binding = transport._hindsight_tool_profile
            assert binding.profile == initial_profile
            conflict = "admin" if initial_profile == "reader" else "reader"
            headers = {"X-Bank-Id": "session-bank", "Mcp-Session-Id": sid, "X-Hindsight-Tool-Profile": conflict}
            response = await client.post(
                test_server.url, headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            )
            assert response.status_code == 400
            assert "fixed for this session" in response.json()["error"]
            headers["X-Hindsight-Tool-Profile"] = ""
            response = await client.post(
                test_server.url, headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            )
            assert response.status_code == 400
            del headers["X-Hindsight-Tool-Profile"]
            response = await client.delete(test_server.url, headers=headers)
            assert response.status_code == 200
            assert not hasattr(transport, "_hindsight_tool_profile")
            assert len(test_server.middleware._profile_sessions) == 0
            response = await client.post(
                test_server.url, headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            )
            assert response.status_code == 404

            response = await initialize_request(client, test_server.url, "reader")
            new_sid = response.headers["mcp-session-id"]
            transport = manager._server_instances[new_sid]
            delattr(transport, "_hindsight_tool_profile")
            response = await client.post(
                test_server.url,
                headers={"X-Bank-Id": "session-bank", "Mcp-Session-Id": new_sid},
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            assert response.status_code == 409
            assert "initialize a new session" in response.json()["error"]


@pytest.mark.asyncio
async def test_stateful_capacity_no_eviction_and_default_unchanged(memory: MagicMock) -> None:
    async with loopback_server(memory, False) as test_server:
        test_server.middleware._MAX_PROFILE_SESSIONS = 1
        async with httpx.AsyncClient() as client:
            first = await initialize_request(client, test_server.url, "reader")
            sid = first.headers["mcp-session-id"]
            assert len(test_server.middleware._profile_sessions) == 1
            over_capacity = await initialize_request(client, test_server.url, "admin")
            assert over_capacity.status_code == 503
            assert len(test_server.middleware._profile_sessions) == 1
            assert test_server.middleware._pending_profile_sessions == 0
            # No explicit profile adds no view quota to the SDK's legacy behavior.
            default = await initialize_request(client, test_server.url, None)
            assert default.status_code == 200
            manager = _get_session_manager(test_server.http_app)
            assert manager._server_instances[sid]._hindsight_tool_profile.profile == "reader"
            deleted = await client.delete(test_server.url, headers={"X-Bank-Id": "session-bank", "Mcp-Session-Id": sid})
            assert deleted.status_code == 200
            assert len(test_server.middleware._profile_sessions) == 0
            later = await initialize_request(client, test_server.url, "curator")
            assert later.status_code == 200
            assert len(test_server.middleware._profile_sessions) == 1


@pytest.mark.asyncio
async def test_failed_initialization_and_auth_never_bind(memory: MagicMock) -> None:
    from hindsight_api.extensions.tenant import AuthenticationError

    async with loopback_server(memory, False) as test_server:
        async with httpx.AsyncClient() as client:
            malformed = await client.post(
                test_server.url,
                headers={"X-Bank-Id": "session-bank", "X-Hindsight-Tool-Profile": "reader"},
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
            assert "error" in malformed.text
            manager = _get_session_manager(test_server.http_app)
            assert not any(
                hasattr(transport, "_hindsight_tool_profile") for transport in manager._server_instances.values()
            )
            assert len(test_server.middleware._profile_sessions) == 0
            # The SDK may finish its server-side stream cleanup just after the
            # client sees the final response bytes; wait for dispatch to unwind.
            async with asyncio.timeout(2):
                while test_server.middleware._pending_profile_sessions:
                    await asyncio.sleep(0.01)
            prior_count = len(manager._server_instances)
            memory._tenant_extension.authenticate_mcp.side_effect = AuthenticationError("rejected test credential")
            rejected = await initialize_request(client, test_server.url, "reader")
            assert rejected.status_code == 401
            assert len(manager._server_instances) == prior_count
            assert len(test_server.middleware._profile_sessions) == 0


@pytest.mark.asyncio
async def test_pending_reservation_and_context_reset_on_downstream_failure(memory: MagicMock) -> None:
    async def broken(scope, receive, send) -> None:
        assert get_current_tool_profile() == "reader"
        raise RuntimeError("failed SDK dispatch")

    mcp = create_mcp_server(memory)
    http_app = mcp.http_app(path="/", stateless_http=False)
    async with http_app.router.lifespan_context(http_app):
        middleware = MCPMiddleware(downstream, memory, multi_bank_app=broken, single_bank_app=broken)
        middleware.multi_bank_session_manager = _get_session_manager(http_app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as client:
            with pytest.raises(RuntimeError, match="failed SDK dispatch"):
                await initialize_request(client, "http://test/mcp/", "reader")
        assert middleware._pending_profile_sessions == 0
        assert len(middleware._profile_sessions) == 0
        assert get_current_tool_profile() is None


@pytest.mark.asyncio
async def test_stateful_capacity_reservation_is_concurrent_safe(memory: MagicMock) -> None:
    barrier = asyncio.Barrier(2)

    async def resolve_alias(bank_id: str, **kwargs) -> str:
        await barrier.wait()
        return bank_id

    memory.resolve_bank_alias.side_effect = resolve_alias
    async with loopback_server(memory, False) as test_server:
        test_server.middleware._MAX_PROFILE_SESSIONS = 1
        async with httpx.AsyncClient() as client:
            responses = await asyncio.gather(
                initialize_request(client, test_server.url, "reader"),
                initialize_request(client, test_server.url, "admin"),
            )
            assert sorted(response.status_code for response in responses) == [200, 503]
            assert len(test_server.middleware._profile_sessions) == 1
            async with asyncio.timeout(2):
                while test_server.middleware._pending_profile_sessions:
                    await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,sid", [("reader", None), ("admin", None), (None, "unknown-session")])
async def test_unknown_sdk_lifecycle_explicitly_errors(memory: MagicMock, profile: str | None, sid: str | None) -> None:
    mcp = create_mcp_server(memory)
    http_app = mcp.http_app(path="/", stateless_http=False)
    with patch("hindsight_api.api.mcp._get_session_manager", return_value=None):
        middleware = MCPMiddleware(downstream, memory, multi_bank_app=http_app, single_bank_app=http_app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as client:
            if sid is None:
                response = await initialize_request(client, "http://test/mcp/", profile)
            else:
                response = await client.post("/mcp/", headers={"X-Bank-Id": "session-bank", "Mcp-Session-Id": sid})
            assert response.status_code == 503
            assert "session manager unavailable" in response.json()["error"]
            assert middleware._pending_profile_sessions == 0
            assert get_current_tool_profile() is None
