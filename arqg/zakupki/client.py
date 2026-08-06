"""SOAP client for the ЕИС «сервисы отдачи данных» (44-ФЗ).

Access scheme as of 2025 (this is the only non-trivial part of the source):

* the FTP dump (``ftp://fz223free@ftp.zakupki.gov.ru``) that every older guide
  recommends **was shut down on 2025-01-01** — those recipes no longer work;
* everything now goes through https://int44.zakupki.gov.ru/eis-integration/services/,
  and every call must carry credentials:

  - **individuals** — a token obtained at https://zakupki.gov.ru/pmd/auth/welcome
    (login via Госуслуги → «Регистрация нового потребителя машиночитаемых
    данных» → «Физическое лицо»), passed as the ``individualPerson_token``
    SOAP header. Endpoint ``getDocsIP``;
  - **legal entities** — requests signed with a qualified ЭЦП, endpoint
    ``getDocsLE2``. The signing infrastructure (ГОСТ-криптография) is out of
    scope here; this client only carries the token flow.

The token is also required on the **archive download**, as an HTTP header — the
archive URL returned in the response is not public.

Two things bite everyone on first contact and are handled here:

1. **Tag order is significant.** The services validate against the XSD
   positionally; a correct-looking envelope with ``subsystemType`` before
   ``orgRegion`` is rejected. Envelopes are therefore assembled from ordered
   tuples, never from dicts.
2. **Quotas.** The services are rate- and volume-limited per consumer; requests
   are paced (``EisConfig.pause``) and retried with backoff on 429/5xx.

Nothing here is verified against a live endpoint from CI — zakupki.gov.ru
refuses connections from non-Russian address space (TLS reset at ClientHello),
so the network path must be exercised from a Russian IP. Run
``scripts/build_zakupki_corpus.py xsd`` first: it saves the live XSD next to
your data so element names below can be checked against the current schema.
"""
from __future__ import annotations

import datetime as dt
import os
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape

from ..utils import ensure_parent, log

BASE_URL = "https://int44.zakupki.gov.ru/eis-integration/services"

#: service name -> WSDL target namespace of its request elements. ``getDocsIP``
#: is the individual-token endpoint and is the one this client is written for;
#: the ``getDocsLE2`` namespace is only a best guess until you have read its XSD
#: (``scripts/build_zakupki_corpus.py xsd --service getDocsLE2``) — override with
#: ``EisConfig.ws_namespace`` if it differs.
SERVICES: dict[str, str] = {
    "getDocsIP": "http://zakupki.gov.ru/fz44/get-docs-ip/ws",
    "getDocsLE2": "http://zakupki.gov.ru/fz44/get-docs-le2/ws",
}

SOAP_ENV = "http://schemas.xmlsoap.org/soap/envelope/"

#: Подсистемы ЕИС. Documents live in exactly one of them, and the wrong
#: ``subsystemType`` returns an empty archive rather than an error.
SUBSYSTEMS = {
    "PRIZ": "извещения, протоколы, планы-графики (размещение заказа)",
    "RGK": "реестр контрактов",
    "RPGZ": "реестр планов-графиков",
    "RPEC": "реестр правил нормирования",
    "RDI": "реестр документов об исполнении",
    "RJ": "реестр жалоб, проверок",
    "NSI": "нормативно-справочная информация",
}

#: A conservative default set of 44-ФЗ document types that produce prose-heavy,
#: highly templated documents — the near-duplicate material this source is for.
#: The authoritative list is in the XSD; add types freely, unknown ones simply
#: come back empty.
DEFAULT_DOCUMENT_TYPES = (
    "epNotificationEF2020",      # извещение об электронном аукционе
    "epNotificationEOK2020",     # извещение об открытом конкурсе в эл. форме
    "epNotificationEZK2020",     # извещение о запросе котировок в эл. форме
)


class EisError(RuntimeError):
    """A SOAP fault, an HTTP error, or a response we could not make sense of."""


@dataclass
class EisConfig:
    """Everything the client needs; ``token`` is the only required field."""

    token: str = ""
    service: str = "getDocsIP"
    base_url: str = BASE_URL
    ws_namespace: str = ""           # override SERVICES[service] if the XSD differs
    mode: str = "PROD"               # PROD | TEST
    timeout: float = 180.0
    retries: int = 4
    pause: float = 2.0               # polite delay between calls (quota friendly)
    verify: bool = True
    user_agent: str = "arqg-zakupki/1.0"

    def __post_init__(self) -> None:
        self.token = self.token or os.environ.get("EIS_TOKEN", "")
        if self.service not in SERVICES and not self.ws_namespace:
            raise ValueError(
                f"unknown service {self.service!r}; pass ws_namespace explicitly "
                f"or pick one of {sorted(SERVICES)}")

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.service}"

    @property
    def namespace(self) -> str:
        return self.ws_namespace or SERVICES[self.service]


# --------------------------------------------------------------------------- #
# envelope construction
# --------------------------------------------------------------------------- #
Param = tuple[str, Any]   # (tag, text) or (tag, [nested Params])


def _render(params: Sequence[Param], indent: str = "      ") -> str:
    out: list[str] = []
    for tag, value in params:
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple)):
            inner = _render(value, indent + "  ")
            if not inner.strip():
                continue
            out.append(f"{indent}<{tag}>\n{inner}\n{indent}</{tag}>")
        else:
            out.append(f"{indent}<{tag}>{escape(str(value))}</{tag}>")
    return "\n".join(out)


def build_envelope(method: str, params: Sequence[Param], *, namespace: str,
                   token: str, mode: str = "PROD",
                   request_id: str = "", created: str = "") -> str:
    """Assemble a request envelope.

    ``params`` is an *ordered* sequence — the services validate positionally, so
    the caller's order is preserved verbatim and nothing is sorted on the way.
    """
    index: list[Param] = [
        ("id", request_id or str(uuid.uuid4())),
        ("createDateTime", created or dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        ("mode", mode),
    ]
    body = _render([("index", index), *params], indent="      ")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<soapenv:Envelope xmlns:soapenv="{SOAP_ENV}" xmlns:ws="{namespace}">\n'
        "  <soapenv:Header>\n"
        f"    <individualPerson_token>{escape(token)}</individualPerson_token>\n"
        "  </soapenv:Header>\n"
        "  <soapenv:Body>\n"
        f"    <ws:{method}>\n"
        f"{body}\n"
        f"    </ws:{method}>\n"
        "  </soapenv:Body>\n"
        "</soapenv:Envelope>\n"
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_response(xml_text: str) -> list[str]:
    """Pull ``archiveUrl`` values out of a response, raising on a SOAP fault.

    Element names are matched on the local part only: the services return
    several namespace prefixes (``ns2:``, ``ns5:`` …) and they have changed
    between schema versions.
    """
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError as e:
        raise EisError(f"response is not XML: {e}; first 500 chars: {xml_text[:500]!r}") from e

    for el in root.iter():
        if _local(el.tag) == "Fault":
            parts = {_local(c.tag): (c.text or "").strip() for c in el}
            raise EisError(f"SOAP fault: {parts}")

    urls = [(el.text or "").strip() for el in root.iter() if _local(el.tag) == "archiveUrl"]
    urls = [u for u in urls if u]
    if not urls:
        # An empty result is legitimate (no documents that day) — the caller
        # decides whether that is a problem, so report it as an empty list.
        log.info("zakupki: response carries no archiveUrl (empty selection?)")
    return urls


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
@dataclass
class EisClient:
    """Thin, synchronous SOAP caller. One instance per token."""

    cfg: EisConfig = field(default_factory=EisConfig)

    def __post_init__(self) -> None:
        if not self.cfg.token:
            raise EisError(
                "no ЕИС token: pass EisConfig(token=...) or export EIS_TOKEN. "
                "Individuals get one at https://zakupki.gov.ru/pmd/auth/welcome "
                "(Госуслуги → регистрация потребителя машиночитаемых данных).")
        try:
            import httpx  # noqa: F401
        except ImportError as e:                       # pragma: no cover
            raise EisError("httpx is required for the ЕИС client (pip install httpx)") from e

    # ---- transport ------------------------------------------------------- #
    def _client(self):
        import httpx
        return httpx.Client(timeout=self.cfg.timeout, verify=self.cfg.verify,
                            follow_redirects=True,
                            headers={"User-Agent": self.cfg.user_agent})

    def _post(self, envelope: str) -> str:
        import httpx
        headers = {"Content-Type": "text/xml; charset=utf-8", "SOAPAction": ""}
        last: Exception | None = None
        with self._client() as http:
            for attempt in range(self.cfg.retries):
                if attempt:
                    delay = self.cfg.pause * (2 ** attempt)
                    log.warning("zakupki: retry %d/%d in %.0fs (%s)",
                                attempt, self.cfg.retries - 1, delay, last)
                    time.sleep(delay)
                try:
                    r = http.post(self.cfg.endpoint, content=envelope.encode("utf-8"),
                                  headers=headers)
                except httpx.HTTPError as e:
                    last = e
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    last = EisError(f"HTTP {r.status_code}: {r.text[:300]}")
                    continue
                if r.status_code >= 400:
                    # 401/403 are credential problems — retrying cannot fix them.
                    raise EisError(f"HTTP {r.status_code} from {self.cfg.endpoint}: "
                                   f"{r.text[:500]}")
                return r.text
        raise EisError(f"{self.cfg.endpoint} unreachable after "
                       f"{self.cfg.retries} attempts: {last}")

    def call(self, method: str, params: Sequence[Param]) -> list[str]:
        """Send one request, return the archive URLs it produced."""
        envelope = build_envelope(method, params, namespace=self.cfg.namespace,
                                  token=self.cfg.token, mode=self.cfg.mode)
        log.debug("zakupki request %s:\n%s", method, envelope)
        urls = parse_response(self._post(envelope))
        log.info("zakupki: %s -> %d archive(s)", method, len(urls))
        time.sleep(self.cfg.pause)
        return urls

    # ---- documented methods ---------------------------------------------- #
    def by_reestr_number(self, reestr_number: str, *, subsystem: str = "PRIZ") -> list[str]:
        """Documents of one procurement, by its registry number."""
        return self.call("getDocsByReestrNumberRequest", [
            ("selectionParams", [
                ("subsystemType", subsystem),
                ("reestrNumber", reestr_number),
            ]),
        ])

    def by_org_region(self, org_region: str, document_type: str, exact_date: str, *,
                      subsystem: str = "PRIZ") -> list[str]:
        """All documents of one type, one region, one day.

        ``org_region`` is the customer's region code (ОКТМО-style two digits,
        ``"72"`` = Тюменская область), ``exact_date`` is ``YYYY-MM-DD``. The day
        granularity is the service's, not ours — a range is walked day by day by
        :meth:`iter_org_region_archives`.
        """
        return self.call("getDocsByOrgRegionRequest", [
            ("selectionParams", [
                ("orgRegion", org_region),
                ("subsystemType", subsystem),
                ("documentType44", document_type),
                ("periodInfo", [("exactDate", exact_date)]),
            ]),
        ])

    def nsi(self, nsi_code: str, *, subsystem: str = "NSI") -> list[str]:
        """Reference data (справочники: КТРУ, ОКПД2, ОКЕИ …)."""
        return self.call("getNsiRequest", [
            ("selectionParams", [
                ("subsystemType", subsystem),
                ("nsiCode44", nsi_code),
            ]),
        ])

    # ---- bulk walk -------------------------------------------------------- #
    def iter_org_region_archives(self, regions: Iterable[str],
                                 document_types: Iterable[str],
                                 dates: Iterable[str], *,
                                 subsystem: str = "PRIZ") -> Iterable[tuple[str, dict[str, str]]]:
        """Yield ``(archive_url, provenance)`` over the region × type × day grid.

        Failures on a single cell are logged and skipped: a month-long crawl
        should not die because one document type was retired mid-period.
        """
        for date in dates:
            for region in regions:
                for doctype in document_types:
                    try:
                        urls = self.by_org_region(region, doctype, date, subsystem=subsystem)
                    except EisError as e:
                        log.warning("zakupki: %s/%s/%s failed: %s", date, region, doctype, e)
                        continue
                    for url in urls:
                        yield url, {"date": date, "org_region": region,
                                    "document_type": doctype, "subsystem": subsystem}

    # ---- archive download -------------------------------------------------- #
    def download(self, url: str, dest: str) -> str:
        """Fetch one archive. The token goes in an HTTP header here, not the XML."""
        import httpx
        ensure_parent(dest)
        headers = {"individualPerson_token": self.cfg.token}
        last: Exception | None = None
        with self._client() as http:
            for attempt in range(self.cfg.retries):
                if attempt:
                    time.sleep(self.cfg.pause * (2 ** attempt))
                try:
                    with http.stream("GET", url, headers=headers) as r:
                        if r.status_code == 429 or r.status_code >= 500:
                            last = EisError(f"HTTP {r.status_code}")
                            continue
                        r.raise_for_status()
                        tmp = dest + ".part"
                        with open(tmp, "wb") as f:
                            for block in r.iter_bytes(1 << 16):
                                f.write(block)
                    os.replace(tmp, dest)
                    log.info("zakupki: saved %s (%d bytes)", dest, os.path.getsize(dest))
                    return dest
                except httpx.HTTPError as e:
                    last = e
        raise EisError(f"download failed for {url}: {last}")
