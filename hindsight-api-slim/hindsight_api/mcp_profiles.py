"""Client-selected HTTP MCP tool views, not authentication or authorization.

An omitted profile and ``admin`` add no restriction. The finite views deliberately
exclude future tools until they are explicitly reviewed into the allowlists.
"""

from types import MappingProxyType
from typing import Literal, Mapping

ToolProfile = Literal["reader", "curator", "admin"]

_READER: frozenset[str] = frozenset(
    {
        "recall",
        "reflect",
        "search_knowledge_base",
        "get_knowledge_page",
        "get_knowledge_base_tree",
        "list_banks",
    }
)
_CURATOR: frozenset[str] = _READER | frozenset(
    {
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
)

TOOL_PROFILES: Mapping[ToolProfile, frozenset[str] | None] = MappingProxyType(
    {"reader": _READER, "curator": _CURATOR, "admin": None}
)
