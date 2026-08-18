"""Which meta field a chunk is grouped by, and how (plan §3.1, generalised).

`sections.py` supplies the breadcrumb-specific mechanics (path segments, the
folder a document lives in, shared-prefix depth) for `title`. That was the
*only* grouping S1 could use. This module is the layer above it: it names
*which* field feeds the grouping — a builtin chunk attribute, or a key in
`Chunk.meta` that a corpus's metadata sidecar carries — and *how* to turn its
value into one scope key, so `subgraphs.py` never has to know whether it is
grouping ЦКР's `title` breadcrumb or zakupki's `region`.

Two strategies today, chosen by ``mining.scope_strategy``:

``"path"``
    The field is a ``/``-separated breadcrumb; the scope is its folder (see
    `sections.scope_of`). This is the original title-folder behaviour and
    stays the default, so an existing config keeps mining exactly as before.

``"exact"``
    The field's value *is* the scope, verbatim — the right shape for a flat
    categorical facet that is not a path at all: zakupki's `region`,
    `customer`, `okpd2_code`, `law`, `year`, `price_bucket`, … (see S0's
    `index_fields.yaml` → `meta_fields` for what a given corpus actually
    offers, coverage and cardinality included).

Adding a third strategy later (a numeric bucket, a date rounded to month, a
composite of two fields) means adding one branch here — `subgraphs.py` and
`compat.py` do not change.
"""
from __future__ import annotations

from ..schema import Chunk
from .sections import scope_of as _path_scope_of

STRATEGIES = ("path", "exact")

#: Fields that live directly on the ``Chunk`` dataclass rather than in its
#: ``meta`` sidecar dict.
BUILTIN_FIELDS = ("title", "document_id", "file_name")


def field_value(chunk: Chunk, field: str) -> str:
    """Raw value of ``field`` for ``chunk``, whichever side of it lives on.

    A list-valued facet is joined rather than stringified: a sidecar that
    carries ``datasets: ["eis", "contracts"]`` should read as text both when it
    labels a passage and when it keys a scope, not as ``['eis', 'contracts']``.
    """
    if field in BUILTIN_FIELDS:
        return getattr(chunk, field, "") or ""
    value = chunk.meta.get(field)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v not in (None, ""))
    return str(value or "")


def scope_of(chunk: Chunk, field: str, strategy: str, *,
            gap: int = 1, min_depth: int = 2) -> str:
    """The scope key ``chunk`` belongs to under (``field``, ``strategy``), or
    ``""`` when it does not scope — too shallow a path, or an empty value.

    ``gap``/``min_depth`` only mean anything for ``"path"``; ``"exact"``
    ignores them, there is nothing to cut a folder from.
    """
    value = field_value(chunk, field)
    if strategy == "path":
        return _path_scope_of(value, gap=gap, min_depth=min_depth)
    if strategy == "exact":
        return value.strip()
    raise ValueError(
        f"unknown mining.scope_strategy: {strategy!r} (expected one of {STRATEGIES})")


# --------------------------------------------------------------------------- #
# surfacing facets — one header, three consumers
# --------------------------------------------------------------------------- #
def facet_header(chunk: Chunk, fields: list[str] | tuple[str, ...],
                 labels: dict[str, str] | None = None,
                 max_value_chars: int = 120) -> str:
    """One line naming ``chunk``'s facets, or ``""`` when it has none of them.

    Deliberately the *only* place this string is built. It goes into the
    passage both index branches hold, into the prompts that extract facts and
    compose questions, and into the vector an injected distractor is embedded
    as — and the whole point is defeated the moment two of those disagree: a
    chunk retrieved as one string and judged as another is measured in an
    environment nobody runs.

    Not a source of facts. `verbatim_span` is checked against the chunk text
    alone (see facts.py), exactly as it is for the breadcrumb.
    """
    labels = labels or {}
    parts: list[str] = []
    for field in fields:
        value = " ".join(field_value(chunk, field).split())
        if not value:
            continue
        if max_value_chars > 0 and len(value) > max_value_chars:
            value = value[:max_value_chars].rstrip() + "…"
        parts.append(f"{labels.get(field, field)}: {value}")
    return " | ".join(parts)
