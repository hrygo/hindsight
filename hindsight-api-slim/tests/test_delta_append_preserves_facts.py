"""Growing a source must retain the identities of its already-extracted facts.

Re-chunking an append can change the last historical chunk without changing any
historical source text. Re-extraction is not guaranteed to reproduce every fact,
and deleting that chunk also destroys the source IDs used by observations.
These tests exercise the public retain API and real PostgreSQL writes. Only the
LLM and embeddings are deterministic; chunking, delta, transactions and cascades
are the production implementations.
"""

import hashlib
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer
from hindsight_api.engine.task_backend import SyncTaskBackend


class _Embeddings(Embeddings):
    @property
    def provider_name(self):
        return "stub"

    @property
    def dimension(self):
        return 384

    async def initialize(self):
        pass

    async def encode(self, texts):
        return [[byte / 255 for byte in hashlib.sha256(text.encode()).digest()] * 12 for text in texts]


@pytest_asyncio.fixture
async def append_memory(pg0_db_url, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "100000")
    clear_config_cache()
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_Embeddings(),
        cross_encoder=AsyncMock(),
        query_analyzer=DateparserQueryAnalyzer(),
        pool_min_size=1,
        pool_max_size=15,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
    )
    await memory.initialize()
    try:
        yield memory
    finally:
        await memory.close()


def _body(long):
    if not long:
        return "Alice shipped Alpha at Google. Bob reviewed Beta at Microsoft."
    prefix = " ".join(f"Alice shipped Alpha{i} at Google in the infrastructure group." for i in range(90))
    tail = " ".join(f"Bob reviewed Beta{i} at Microsoft in the networking group." for i in range(40))
    return f"{prefix}\n\n{tail}"


async def _state(memory, bank, document):
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        facts = await conn.fetch(
            "SELECT id::text, text, chunk_id FROM memory_units "
            "WHERE bank_id=$1 AND document_id=$2 AND fact_type IN ('world', 'experience')",
            bank,
            document,
        )
        chunks = await conn.fetch(
            "SELECT chunk_id, content_hash FROM chunks WHERE bank_id=$1 AND document_id=$2",
            bank,
            document,
        )
    return {f["id"]: (f["text"], f["chunk_id"]) for f in facts}, {c["chunk_id"]: c["content_hash"] for c in chunks}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,long,oversized",
    [
        ("append", False, False),
        ("append", True, False),
        ("replace", False, False),
        ("replace", True, False),
        ("replace", True, True),
        ("append", True, True),
    ],
)
async def test_extension_preserves_old_fact_ids(append_memory, request_context, monkeypatch, mode, long, oversized):
    memory = append_memory
    bank = f"test-append-identities-{uuid.uuid4().hex}"
    document = "growing-session"
    before_body = _body(long)
    tail = "Carol shipped Gamma at Amazon. " * (150 if oversized and mode == "append" else 3)
    try:
        # Seed the historical body in two turns. Their independent chunk boundaries
        # differ from re-chunking the concatenated transcript on the next turn.
        midpoint = 5200 if long else len(before_body) // 2
        head, second_turn = before_body[:midpoint], before_body[midpoint:]
        await memory.retain_async(bank_id=bank, document_id=document, content=head, request_context=request_context)
        await memory.retain_batch_async(
            bank_id=bank,
            contents=[{"document_id": document, "content": second_turn, "update_mode": "append"}],
            request_context=request_context,
        )
        before_body = head + "\n" + second_turn
        before_facts, before_chunks = await _state(memory, bank, document)
        assert before_facts, "premise: the historical document must have facts"
        if oversized:
            monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "500")
            clear_config_cache()
        await memory.retain_batch_async(
            bank_id=bank,
            contents=[
                {
                    "document_id": document,
                    "content": tail if mode == "append" else before_body + "\n" + tail,
                    "update_mode": mode,
                }
            ],
            request_context=request_context,
        )
        after_facts, after_chunks = await _state(memory, bank, document)
        changed = [key for key, value in before_chunks.items() if after_chunks.get(key) != value]
        assert changed, "premise: extension must re-chunk historical content"
        if long:
            assert any(after_chunks.get(key) == value for key, value in before_chunks.items()), (
                "premise: delta has an unchanged prefix"
            )
        lost = [key for key, value in before_facts.items() if after_facts.get(key) != value]
        assert not lost, f"source only grew, but {len(lost)} historical facts were deleted or replaced"
        result = await memory.get_document(document, bank, request_context=request_context)
        assert result["original_text"] == before_body + "\n" + tail
        assert len(after_facts) > len(before_facts), "new tail must also be extracted"
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", ["change", "remove"])
async def test_real_edit_still_removes_superseded_facts(append_memory, request_context, edit):
    memory = append_memory
    bank = f"test-replace-identities-{uuid.uuid4().hex}"
    document = "edited-document"
    body = _body(True)
    prefix, tail = body.split("\n\n")
    replacement = prefix if edit == "remove" else prefix + "\n\n" + tail.replace("Microsoft", "Amazon")
    try:
        await memory.retain_async(bank_id=bank, document_id=document, content=body, request_context=request_context)
        before_facts, before_chunks = await _state(memory, bank, document)
        await memory.retain_async(
            bank_id=bank, document_id=document, content=replacement, request_context=request_context
        )
        after_facts, after_chunks = await _state(memory, bank, document)
        unchanged = {key for key, value in before_chunks.items() if after_chunks.get(key) == value}
        superseded = {key for key in before_chunks if key not in unchanged}
        assert unchanged and superseded, "premise: the replacement edits or deletes only the tail"
        surviving = {key for key, (_, chunk) in before_facts.items() if chunk in unchanged}
        deleted = {key for key, (_, chunk) in before_facts.items() if chunk in superseded}
        assert surviving and deleted
        assert surviving <= after_facts.keys()
        assert deleted.isdisjoint(after_facts)
        result = await memory.get_document(document, bank, request_context=request_context)
        assert result["original_text"] == replacement
    finally:
        await memory.delete_bank(bank, request_context=request_context)
