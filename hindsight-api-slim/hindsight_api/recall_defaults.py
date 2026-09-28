"""Deployment-level defaults for public recall requests.

The engine keeps its own ``prefer_observations=False`` default because internal
callers such as consolidation need the raw facts they fold into observations.
This module applies deployment policy only at public MCP and HTTP boundaries.
"""

from dataclasses import dataclass

VALID_RECALL_DEFAULT_TYPES = frozenset({"world", "experience", "observation"})


@dataclass(frozen=True)
class RecallDefaults:
    """Server policy applied when a public recall request omits a field."""

    types: tuple[str, ...] | None = None
    prefer_observations: bool = False


def parse_recall_default_types(raw: str | None) -> list[str] | None:
    """Parse the comma-separated deployment default without silent typos."""
    if raw is None or not raw.strip():
        return None

    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("HINDSIGHT_API_RECALL_DEFAULT_TYPES must not be empty when set")

    invalid = [value for value in values if value not in VALID_RECALL_DEFAULT_TYPES]
    if invalid:
        raise ValueError(
            "HINDSIGHT_API_RECALL_DEFAULT_TYPES contains invalid values: "
            f"{', '.join(invalid)}. Valid types: {', '.join(sorted(VALID_RECALL_DEFAULT_TYPES))}"
        )
    return values


def resolve_recall_defaults(
    types: list[str] | None,
    prefer_observations: bool | None,
    defaults: RecallDefaults,
) -> tuple[list[str] | None, bool]:
    """Resolve omitted request fields without overwriting explicit values."""
    resolved_types = list(defaults.types) if types is None and defaults.types is not None else types
    resolved_prefer = defaults.prefer_observations if prefer_observations is None else prefer_observations
    return resolved_types, resolved_prefer
