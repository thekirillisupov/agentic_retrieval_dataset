"""ЕИС / zakupki.gov.ru (44-ФЗ) as a corpus source.

Procurement documentation is the best near-duplicate material we have: every
notification is generated from the same regulated template, so thousands of
documents differ only in the customer, the object of purchase, the dates and the
amounts. That is exactly the distractor structure SID needs — a retriever that
survives this corpus is not matching on surface form.

Two ways in, both landing in the same document model:

``client``    SOAP client for the int44 data services. Needs a token and a
              Russian IP; see the README for how access works since 2025.
``tabular``   third-party dumps of the registry (CSV/XLSX), which need neither.

and then, source-independently:

``normalize`` shouted names, float amounts, SQL timestamps, ``||``-packed cells
``parse``     EIS XML -> ordered, Russian-labelled text sections
``facets``    the normalised record every source collapses to, and its prose
``merge``     several sources -> one corpus + per-chunk and per-document metadata
``corpus``    sections -> pipeline chunks + the near-duplicate report
"""
from .client import EisClient, EisConfig, EisError, SERVICES
from .corpus import ChunkOptions, build_corpus, duplicate_report
from .facets import Facets, SourceRef, make_facets
from .merge import SourceSpec, build_merged, merge_sources
from .parse import ProcurementDoc, Section, parse_xml, iter_documents, parse_path
from .tabular import PROFILES, TableProfile, detect_profile, iter_docs, iter_facets

__all__ = [
    "EisClient", "EisConfig", "EisError", "SERVICES",
    "ProcurementDoc", "Section", "parse_xml", "iter_documents", "parse_path",
    "PROFILES", "TableProfile", "detect_profile", "iter_docs", "iter_facets",
    "Facets", "SourceRef", "make_facets",
    "SourceSpec", "build_merged", "merge_sources",
    "ChunkOptions", "build_corpus", "duplicate_report",
]
