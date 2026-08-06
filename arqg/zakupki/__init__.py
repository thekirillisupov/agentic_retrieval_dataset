"""ЕИС / zakupki.gov.ru (44-ФЗ) as a corpus source.

Procurement documentation is the best near-duplicate material we have: every
notification is generated from the same regulated template, so thousands of
documents differ only in the customer, the object of purchase, the dates and the
amounts. That is exactly the distractor structure SID needs — a retriever that
survives this corpus is not matching on surface form.

Three layers, each usable on its own:

``client``  SOAP client for the int44 data services (needs a token — see README)
``parse``   EIS XML -> ordered, Russian-labelled text sections (no network)
``corpus``  sections -> ``{"file_name","index","raw_text",...}`` chunks + a
            near-duplicate report over the result
"""
from .client import EisClient, EisConfig, EisError, SERVICES
from .corpus import build_corpus, duplicate_report
from .parse import ProcurementDoc, parse_xml, iter_documents

__all__ = [
    "EisClient", "EisConfig", "EisError", "SERVICES",
    "ProcurementDoc", "parse_xml", "iter_documents",
    "build_corpus", "duplicate_report",
]
