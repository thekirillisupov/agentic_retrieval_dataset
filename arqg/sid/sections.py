"""Section paths — the corpus ``title`` read as a breadcrumb (plan §3.1).

``title`` in this index is not a headline, it is the document's path in the
knowledge base::

    Общее пространство ЦКР (УСО, ТП, УУП)/Дефекты/Дефекты СберБизнес/<документ>

The last segment names the document; the prefix is the folder it lives in. That
prefix is the only *editorial* topical grouping the index carries. `document_id`
links a document to itself and to nothing else, and extracted entities are a
noisy proxy for topic — two chunks sharing a rare token are about one subject
only if the token happened to mean something. Two chunks in one folder are about
one subject because someone filed them there.

S1 therefore mines subgraphs *within* a scope built from this prefix, and
`index` — the chunk's position in its document — decides whether two chunks of
the *same* document are far enough apart to need a second query rather than
reading on.

The scope is deliberately **not** registered as an index tag on the entity
graph. As a tag it would bypass τ_idf through `EntityGraph.is_tag`, inflate the
df of every folder member and let `_extend` pull in an arbitrary sibling — the
same failure mode that already keeps `document_id` out of the graph. A folder is
*where to look*, not *what ties the chunks together*.
"""
from __future__ import annotations

SEP = "/"


def segments(title: str) -> list[str]:
    """Path segments of a breadcrumb title, empties dropped."""
    return [s for s in (p.strip() for p in (title or "").split(SEP)) if s]


def depth(title: str) -> int:
    return len(segments(title))


def leaf(title: str) -> str:
    """The document's own name — the last segment."""
    segs = segments(title)
    return segs[-1] if segs else ""


def scope_of(title: str, gap: int = 1, min_depth: int = 2) -> str:
    """The folder a document is mined within: its path minus the leaf, minus
    ``gap`` further levels.

    ``gap=0`` is the immediate parent folder (siblings only), ``gap=1`` also
    admits cousins one level up. Returns ``""`` when the result would be
    shallower than ``min_depth`` — a scope of one or two segments is a whole
    business domain, not a topic, and grouping on it is the same as not
    grouping at all. Callers treat ``""`` as "not scopeable".
    """
    segs = segments(title)
    cut = len(segs) - 1 - max(0, gap)
    if cut < max(1, min_depth):
        return ""
    return SEP.join(segs[:cut])


def shared_depth(titles: list[str]) -> int:
    """How many leading segments all the given titles have in common.

    1 means "they share only the corpus root", i.e. topically unrelated.
    """
    parts = [segments(t) for t in titles if segments(t)]
    if not parts:
        return 0
    n = 0
    for group in zip(*parts):
        if len(set(group)) != 1:
            break
        n += 1
    return n


def breadcrumb(title: str, max_segments: int = 0) -> str:
    """Human-readable path for a prompt: ``A > B > C``.

    Trimmed from the *left* when capped: the deepest segments carry the
    specifics, the shallow ones are the same for the whole corpus.
    """
    segs = segments(title)
    if max_segments and len(segs) > max_segments:
        segs = ["…"] + segs[-max_segments:]
    return " > ".join(segs)
