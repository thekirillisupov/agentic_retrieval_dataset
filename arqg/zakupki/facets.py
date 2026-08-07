"""The normalised record every ЕИС source collapses to, and the prose it renders.

One :class:`Facets` sits between "some dump's columns" and "a passage in the
corpus". Three things depend on that indirection:

* **merging** — dumps overlap, and two rows about the same procurement are
  merged field by field (:mod:`arqg.zakupki.merge`) before anything is rendered,
  so the corpus has one document per purchase, not one per source;
* **prose** — the same facts render as sentences rather than as form fields,
  because a passage reading ``Заказчик: КОМИТЕТ ПО РЕГУЛИРОВАНИЮ`` gives an
  embedding model far less to work with than a sentence does;
* **metadata** — the facets are exactly the facets worth filtering and searching
  on, so the sidecar is a projection of this object rather than a second,
  drifting extraction.

The corpus file itself stays in the pipeline's five-field format; everything
else goes to the sidecar, keyed by ``chunk_id``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .normalize import (clean_text, dedupe_indices, format_money, format_percent,
                        price_bucket, region_name, round_gloss, sentence, smart_case)
from .parse import ProcurementDoc, Section

#: Canonical EIS card URLs. A 19-digit registry number is enough to address the
#: document on the portal, which is the "path to the document" that survives
#: whichever dump it happened to arrive in.
NOTICE_URL_44 = "https://zakupki.gov.ru/epz/order/notice/view/common-info.html?regNumber={}"
NOTICE_URL_223 = ("https://zakupki.gov.ru/223/purchase/public/purchase/info/"
                  "common-info.html?regNumber={}")
CONTRACT_URL = "https://zakupki.gov.ru/epz/contract/contractCard/common-info.html?reestrNumber={}"


@dataclass
class SourceRef:
    """Where one contributing record physically came from."""

    dataset: str = ""        # profile name
    origin: str = ""         # publisher URL the dump was downloaded from
    licence: str = ""
    path: str = ""           # local file it was read from
    locator: str = ""        # row number, or archive!member for XML

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class Facets:
    """Everything we know about one procurement, normalised."""

    purchase_number: str = ""
    law: str = ""
    procedure: str = ""
    published: str = ""              # ISO date
    published_long: str = ""         # «11 августа 2020 года, 16:27»
    trade_date: str = ""
    trade_date_long: str = ""
    url: str = ""

    subject: str = ""
    lot_name: str = ""
    okpd2_code: str = ""
    okpd2_name: str = ""
    items: list[str] = field(default_factory=list)

    customer: str = ""
    customer_inn: str = ""
    sponsor: str = ""
    region_code: str = ""
    region: str = ""

    price_start: str = ""            # raw numeric text, kept for the sidecar
    price_final: str = ""
    security: str = ""
    advance: str = ""
    currency: str = ""
    small_business: str = ""

    phase: str = ""
    winner: str = ""
    winner_inn: str = ""
    contract_date: str = ""
    contract_date_long: str = ""
    contract_url: str = ""

    sources: list[SourceRef] = field(default_factory=list)

    # ---- identity --------------------------------------------------------- #
    @property
    def doc_id(self) -> str:
        return self.purchase_number

    @property
    def year(self) -> str:
        return self.published[:4] if self.published else ""

    @property
    def notice_url(self) -> str:
        """The portal address of the document, whichever dump carried it.

        Rebuilt from the registry number rather than taken from the dump, so the
        path survives a source that only linked its own search page. 44-ФЗ
        numbers are 19 digits and 223-ФЗ numbers 11, and the two laws live under
        different sections of the portal; anything else falls back to whatever
        link the dump provided.
        """
        number = self.purchase_number
        if number.isdigit():
            if len(number) >= 19:
                return NOTICE_URL_44.format(number)
            if len(number) == 11 and self.law.startswith("223"):
                return NOTICE_URL_223.format(number)
        return self.url

    def is_empty(self) -> bool:
        return not (self.subject or self.lot_name or self.items or self.customer)

    # ---- prose ------------------------------------------------------------ #
    def sections(self) -> list[Section]:
        """Render the facets as paragraphs, one section per topic."""
        out: list[Section] = []
        for title, paragraph in (
                ("Общие сведения о закупке", self._general()),
                ("Объект закупки", self._subject()),
                ("Заказчик", self._customer()),
                ("Цена контракта и обеспечение", self._price()),
                ("Результаты определения поставщика", self._outcome())):
            if paragraph:
                out.append(Section(title, [paragraph]))
        return out

    def _general(self) -> str:
        s: list[str] = []
        if self.purchase_number:
            head = f"закупка № {self.purchase_number}"
            if self.published_long:
                head += f" размещена в единой информационной системе {self.published_long}"
            s.append(sentence(head))
        elif self.published_long:
            s.append(sentence(f"закупка размещена {self.published_long}"))
        law_proc = []
        if self.law:
            law_proc.append(f"в соответствии с {self.law}")
        if self.procedure:
            law_proc.append(f"способом «{self.procedure.lower()}»")
        if law_proc:
            s.append(sentence("определение поставщика проводится " + " ".join(law_proc)))
        if self.trade_date_long:
            s.append(sentence(f"процедура назначена на {self.trade_date_long}"))
        if self.notice_url:
            s.append(sentence(f"документ опубликован по адресу {self.notice_url}"))
        return " ".join(s)

    def _subject(self) -> str:
        # The dumps repeat themselves across columns: purchase_name, okpd2_names
        # and item_descriptions are frequently the same sentence. Deduplicating
        # once, across all of them, is what keeps the passage from saying the
        # same thing three times — which teaches a retriever nothing and inflates
        # every similarity score computed over the corpus.
        subject = self.subject or self.lot_name
        fields = [subject, self.lot_name, self.okpd2_name, *self.items]
        keep = set(dedupe_indices(fields))

        s: list[str] = []
        if subject:
            s.append(sentence(f"объект закупки — {subject[0].lower() + subject[1:]}"))
        if 1 in keep and self.lot_name != subject:
            s.append(sentence(f"наименование лота — {self.lot_name}"))
        if self.okpd2_code or 2 in keep:
            code = f" {self.okpd2_code}" if self.okpd2_code else ""
            name = f" «{self.okpd2_name}»" if 2 in keep else ""
            s.append(sentence(f"предмет закупки отнесён к коду ОКПД2{code}{name}"))
        items = [fields[i] for i in sorted(keep) if i >= 3]
        if items:
            listed = "; ".join(items)
            s.append(sentence(
                f"в состав закупки входит {len(items)} позиц"
                f"{'ия' if len(items) == 1 else 'ии' if len(items) < 5 else 'ий'}: {listed}"))
        return " ".join(s)

    def _customer(self) -> str:
        s: list[str] = []
        if self.customer:
            inn = f" (ИНН {self.customer_inn})" if self.customer_inn else ""
            s.append(sentence(f"заказчиком выступает {self.customer}{inn}"))
        elif self.customer_inn:
            s.append(sentence(f"ИНН заказчика — {self.customer_inn}"))
        if self.region:
            code = f" (код региона {self.region_code})" if self.region_code else ""
            s.append(sentence(f"заказчик расположен в регионе: {self.region}{code}"))
        if self.sponsor and self.sponsor != self.customer:
            s.append(sentence(f"размещение закупки осуществляет {self.sponsor}"))
        return " ".join(s)

    def _price(self) -> str:
        s: list[str] = []
        if self.price_start:
            gloss = round_gloss(self.price_start)
            bucket = price_bucket(self.price_start)
            text = f"начальная (максимальная) цена контракта составляет {format_money(self.price_start)}"
            if gloss:
                text += f" ({gloss})"
            s.append(sentence(text))
            if bucket:
                s.append(sentence(f"по масштабу закупка относится к диапазону «{bucket}»"))
        if self.currency and self.currency.upper() not in ("RUB", "РУБ", "643"):
            s.append(sentence(f"валюта расчётов — {self.currency}"))
        if self.security:
            s.append(sentence(f"размер обеспечения заявки — {format_money(self.security)}"))
        if self.advance:
            s.append(sentence(f"предусмотрен аванс в размере {format_percent(self.advance)}"))
        if self.small_business:
            yes = self.small_business.lower() in ("да", "true", "1")
            s.append(sentence(
                "закупка проводится среди субъектов малого предпринимательства и "
                "социально ориентированных некоммерческих организаций" if yes else
                "закупка не ограничена субъектами малого предпринимательства"))
        return " ".join(s)

    def _outcome(self) -> str:
        s: list[str] = []
        if self.phase:
            s.append(sentence(f"этап определения поставщика — {self.phase.lower()}"))
        if self.winner:
            inn = f" (ИНН {self.winner_inn})" if self.winner_inn else ""
            s.append(sentence(f"победителем закупки признан {self.winner}{inn}"))
        if self.price_final:
            s.append(sentence(
                f"цена контракта по результатам закупки — {format_money(self.price_final)}"))
        if self.contract_date_long:
            s.append(sentence(f"контракт заключён {self.contract_date_long}"))
        if self.contract_url:
            s.append(sentence(f"карточка контракта: {self.contract_url}"))
        return " ".join(s)

    # ---- document --------------------------------------------------------- #
    def title(self) -> str:
        """Retrieval-heavy title.

        ``arqg.index`` prepends the title to a passage before embedding it, so
        this is the one place where the document's facets reach *every* chunk of
        it, including the ones that do not mention the customer or the year.
        """
        parts = [f"Закупка № {self.purchase_number}" if self.purchase_number else "Закупка"]
        subject = self.subject or self.lot_name or self.okpd2_name
        if subject:
            parts.append(subject[:160])
        if self.customer:
            parts.append(f"Заказчик — {self.customer[:90]}")
        tail = ", ".join(p for p in (self.region, self.year) if p)
        if tail:
            parts.append(tail)
        return ". ".join(parts)

    def to_doc(self, doc_type: str = "eisProcurement") -> ProcurementDoc | None:
        sections = self.sections()
        if len(sections) < 2:
            return None          # a single-section document can never form a window
        return ProcurementDoc(
            doc_type=doc_type,
            doc_id=self.purchase_number or "unknown",
            title=self.title(),
            sections=sections,
            file_ext=".txt",
            meta=self.summary())

    # ---- metadata --------------------------------------------------------- #
    def chunk_facets(self) -> dict[str, Any]:
        """The compact facet set repeated on every chunk.

        Deliberately narrow: these are the fields a query filters or reranks on,
        and they are written once per *chunk*. Provenance paths, the item count
        and the keyword list live in the document-level file instead — carrying
        them here made the sidecar twice the size of the corpus it describes.
        """
        return {
            "purchase_number": self.purchase_number,
            "source_url": self.notice_url,
            "law": self.law,
            "procedure": self.procedure,
            "published": self.published,
            "year": self.year,
            "region": self.region,
            "region_code": self.region_code,
            "customer": self.customer,
            "customer_inn": self.customer_inn,
            "okpd2_code": self.okpd2_code,
            "price_start": self.price_start,
            "price_bucket": price_bucket(self.price_start),
            "phase": self.phase,
            "winner": self.winner,
            "datasets": sorted({s.dataset for s in self.sources if s.dataset}),
        }

    def summary(self) -> dict[str, Any]:
        """Everything known about the document, including where it came from."""
        return {
            "purchase_number": self.purchase_number,
            "source_url": self.notice_url,
            "law": self.law,
            "procedure": self.procedure,
            "published": self.published,
            "year": self.year,
            "region": self.region,
            "region_code": self.region_code,
            "customer": self.customer,
            "customer_inn": self.customer_inn,
            "okpd2_code": self.okpd2_code,
            "okpd2_name": self.okpd2_name,
            "price_start": self.price_start,
            "price_bucket": price_bucket(self.price_start),
            "price_final": self.price_final,
            "phase": self.phase,
            "winner": self.winner,
            "winner_inn": self.winner_inn,
            "contract_url": self.contract_url,
            "n_items": len(self.items),
            "datasets": sorted({s.dataset for s in self.sources if s.dataset}),
            "licences": sorted({s.licence for s in self.sources if s.licence}),
            "sources": [s.to_dict() for s in self.sources],
        }

    def keywords(self) -> list[str]:
        """Facet terms worth having in a lexical index alongside the passage."""
        raw = [self.law, self.procedure, self.region, self.year,
               price_bucket(self.price_start), self.okpd2_code, self.okpd2_name,
               self.customer, self.winner, self.purchase_number]
        return [k for k in dict.fromkeys(clean_text(x) for x in raw) if k]

    # ---- merging ---------------------------------------------------------- #
    def merge(self, other: "Facets") -> "Facets":
        """Field-level union; ``self`` wins where both sides have a value.

        Dumps are slices of the same registry with different columns — one
        carries the winner, another the contract date — so a union is strictly
        more informative than picking a "best" source. Longer free-text values
        win because the short one is usually a truncation.
        """
        for name, mine in list(self.__dict__.items()):
            theirs = getattr(other, name)
            if name == "sources":
                self.sources = self.sources + [s for s in theirs if s not in self.sources]
            elif isinstance(mine, list):
                if not mine:
                    setattr(self, name, list(theirs))
            elif not mine:
                setattr(self, name, theirs)
            elif isinstance(mine, str) and isinstance(theirs, str) \
                    and len(theirs) > len(mine) and mine.lower() in theirs.lower():
                setattr(self, name, theirs)
        return self


def make_facets(values: dict[str, str], source: SourceRef) -> Facets:
    """Build facets from already-normalised canonical values."""
    f = Facets(sources=[source])
    for key, value in values.items():
        if hasattr(f, key) and value:
            setattr(f, key, value)
    if not f.region:
        f.region = region_name(f.region_code)
    f.customer = smart_case(f.customer)
    f.sponsor = smart_case(f.sponsor)
    f.winner = smart_case(f.winner)
    return f
