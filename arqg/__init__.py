"""arqg — Agentic Retrieval Question Generation.

A staged pipeline that turns a chunked, document-based knowledge base into a
high-quality synthetic dataset for *agentic / multi-hop retrieval*.

The defining property of the produced dataset: each question is answerable only
by combining information from several *neighbouring* chunks, and the gold chunk
set is verified to be *minimal and strictly necessary*.
"""

__version__ = "0.1.0"
