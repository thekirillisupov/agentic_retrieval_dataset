"""ЕИС XML -> ordered, Russian-labelled text sections. No network, no schema files.

The 44-ФЗ schemas are large, versioned (``…EF2020``, ``…EF2023`` …) and change a
few times a year, so a per-document-type field map would rot in a month. This
module goes the other way: it walks *any* EIS document generically and renders it
as the document it already is — a sequence of labelled blocks:

    Сведения о заказчике
    Полное наименование: Государственное бюджетное учреждение …
    ИНН: 7203001234

    Объект закупки. Позиция 1
    Наименование товара: Бумага офисная А4 …

Tags are translated through :data:`LABELS` where we know them and de-camel-cased
where we don't, so a schema revision that adds a field degrades to a slightly
uglier label instead of silently dropping data. Signature and certificate blobs
are the only things thrown away.

Everything is read from ``.xml`` files or (nested) ``.zip`` archives, which is
what the data services actually hand back.
"""
from __future__ import annotations

import io
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..utils import log

# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
#: Blobs and plumbing: base64 signatures, certificates, schema bookkeeping.
NOISE_TAGS = {
    "signature", "signatures", "cryptoSigns", "cryptoSign", "sertificate",
    "certificate", "signatureInfo", "binaryData", "digestValue", "signatureValue",
    "KeyInfo", "X509Data", "X509Certificate", "SignedInfo", "Signature",
    "schemeVersion", "versionNumber", "xmlns",
}

#: Tags whose numeric value reads as money.
MONEY_HINTS = ("price", "sum", "amount", "cost", "Price", "Sum", "Amount")

#: Known tags -> the label ЕИС itself uses in the human-readable card. Unknown
#: tags fall through to :func:`_readable`, so this list is a quality lever, not a
#: correctness requirement — extend it as you meet new document types.
LABELS: dict[str, str] = {
    # identity / publication
    "purchaseNumber": "Номер извещения",
    "docNumber": "Номер документа",
    "regNum": "Реестровый номер",
    "reestrNumber": "Реестровый номер",
    "number": "Номер",
    "id": "Идентификатор",
    "externalId": "Внешний идентификатор",
    "docPublishDate": "Дата размещения",
    "publishDTInEIS": "Дата размещения в ЕИС",
    "createDateTime": "Дата формирования",
    "directDate": "Дата направления",
    "signDate": "Дата подписания",
    "href": "Ссылка на документ",
    "IKZ": "Идентификационный код закупки",
    "printFormUrl": "Печатная форма",
    "versionSchemeVersion": "Версия схемы",
    # subject
    "purchaseObjectInfo": "Наименование объекта закупки",
    "purchaseObject": "Объект закупки",
    "purchaseObjects": "Объекты закупки",
    "purchaseObjectsInfo": "Сведения об объектах закупки",
    "notDrugPurchaseObjectsInfo": "Объекты закупки (не лекарственные средства)",
    "drugPurchaseObjectInfo": "Объекты закупки (лекарственные средства)",
    "name": "Наименование",
    "fullName": "Полное наименование",
    "shortName": "Сокращённое наименование",
    "OKPD2": "Код ОКПД2",
    "KTRU": "Позиция КТРУ",
    "OKEI": "Единица измерения",
    "code": "Код",
    "quantity": "Количество",
    "quantityValue": "Количество",
    "price": "Цена за единицу",
    "sum": "Сумма",
    "maxPrice": "Начальная (максимальная) цена контракта",
    "maxPriceInfo": "Сведения о цене контракта",
    "contractPrice": "Цена контракта",
    "currency": "Валюта",
    "priceRUB": "Цена, руб.",
    # customer / supplier
    "responsibleOrgInfo": "Сведения об организации, осуществляющей размещение",
    "purchaseResponsibleInfo": "Сведения о заказчике",
    "responsibleOrg": "Организация, осуществляющая размещение",
    "responsibleRole": "Роль организации",
    "customer": "Заказчик",
    "customerInfo": "Сведения о заказчике",
    "customerRequirementsInfo": "Требования заказчика",
    "customerRequirement": "Требование заказчика",
    "supplier": "Поставщик (подрядчик, исполнитель)",
    "supplierInfo": "Сведения о поставщике",
    "participant": "Участник закупки",
    "INN": "ИНН",
    "KPP": "КПП",
    "OGRN": "ОГРН",
    "consRegistryNum": "Код по сводному реестру",
    "postAddress": "Почтовый адрес",
    "factAddress": "Фактический адрес",
    "legalAddress": "Юридический адрес",
    "contactPerson": "Контактное лицо",
    "lastName": "Фамилия",
    "firstName": "Имя",
    "middleName": "Отчество",
    "contactEMail": "Адрес электронной почты",
    "email": "Адрес электронной почты",
    "contactPhone": "Телефон",
    "phone": "Телефон",
    "fax": "Факс",
    "additionalInfo": "Дополнительная информация",
    "responsibleInfo": "Ответственное должностное лицо",
    # procedure
    "placingWay": "Способ определения поставщика",
    "ETP": "Электронная площадка",
    "url": "Адрес в сети «Интернет»",
    "procedureInfo": "Порядок проведения процедуры",
    "collectingInfo": "Порядок подачи заявок",
    "collectingEndDate": "Дата и время окончания подачи заявок",
    "startDT": "Дата и время начала",
    "endDT": "Дата и время окончания",
    "biddingDate": "Дата проведения процедуры",
    "summarizingDate": "Дата подведения итогов",
    "scoringDate": "Дата рассмотрения и оценки заявок",
    "openingDate": "Дата вскрытия конвертов",
    "place": "Место",
    "order": "Порядок",
    "lot": "Лот",
    "lots": "Лоты",
    "notificationInfo": "Сведения об извещении",
    "commonInfo": "Общие сведения",
    "printForm": "Печатная форма",
    # money / guarantees / conditions
    "contractConditionsInfo": "Условия контракта",
    "contractGuarantee": "Обеспечение исполнения контракта",
    "applicationGuarantee": "Обеспечение заявки",
    "warrantyGuarantee": "Обеспечение гарантийных обязательств",
    "part": "Размер, %",
    "amount": "Размер",
    "procedureInfoDetails": "Детали процедуры",
    "financeSource": "Источник финансирования",
    "budgetInfo": "Бюджетные сведения",
    "KBK": "Код бюджетной классификации",
    "KOSGU": "КОСГУ",
    "financingInfo": "Сведения о финансировании",
    "paymentTermsInfo": "Порядок оплаты",
    "contractExecutionPlan": "Этапы исполнения контракта",
    "executionPeriod": "Срок исполнения",
    "startDate": "Дата начала",
    "endDate": "Дата окончания",
    "deliveryPlace": "Место поставки",
    "deliveryTerm": "Срок поставки",
    "deliveryPlacesInfo": "Места поставки",
    # requirements
    "requirementsInfo": "Требования к участникам закупки",
    "requirement": "Требование",
    "restrictions": "Ограничения и запреты",
    "restrictionsInfo": "Сведения об ограничениях",
    "preferences": "Преимущества",
    "preferensesInfo": "Сведения о преимуществах",
    "advantages": "Преимущества",
    "shortDescription": "Краткое описание",
    "content": "Содержание",
    "description": "Описание",
    "attachments": "Прилагаемые документы",
    "attachment": "Документ",
    "fileName": "Имя файла",
    "fileSize": "Размер файла",
    "docDescription": "Описание документа",
    "docKindInfo": "Вид документа",
}

#: Document types -> their official name. The fallback is the raw schema name
#: (``epNotificationEF2023``), which is what a procurement specialist recognises
#: anyway — de-camel-casing it would only produce a worse-looking guess.
DOC_TYPE_LABELS: dict[str, str] = {
    "epNotificationEF2020": "Извещение о проведении электронного аукциона",
    "epNotificationEOK2020": "Извещение о проведении открытого конкурса в электронной форме",
    "epNotificationEZK2020": "Извещение о проведении запроса котировок в электронной форме",
    "epNotificationEZP2020": "Извещение о проведении закупки у единственного поставщика",
    "epProtocolEF2020": "Протокол подведения итогов электронного аукциона",
    "epProtocolEOK2020": "Протокол подведения итогов открытого конкурса",
    "cancelNotice": "Извещение об отмене определения поставщика",
    "contract": "Контракт",
    "contractProcedure": "Сведения об исполнении контракта",
    "purchasePlan": "План закупок",
    "purchaseSchedule": "План-график закупок",
    "nsiOKPD2": "Справочник ОКПД2",
    "nsiKTRU": "Справочник КТРУ",
}


def doc_type_label(doc_type: str) -> str:
    return DOC_TYPE_LABELS.get(doc_type, doc_type)


#: Top-level branches we prefer to see first when a document has many.
SECTION_ORDER = (
    "commonInfo", "notificationInfo", "purchaseResponsibleInfo", "responsibleOrgInfo",
    "customer", "supplier", "lot", "lots", "purchaseObjectsInfo",
    "contractConditionsInfo", "procedureInfo", "requirementsInfo", "attachments",
)

_CAMEL_RE = re.compile(r"(?<=[a-zа-я0-9])(?=[A-ZА-Я])")
_DATETIME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::\d{2})?")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _readable(tag: str) -> str:
    """Fallback label for a tag we have no translation for.

    ``brandNewFieldName`` -> ``Brand new field name``, but ``purchaseKTRUCode``
    keeps its ``KTRU``: acronyms carry the meaning in these schemas, so only
    Titlecase words are folded down.
    """
    words = [w for w in _CAMEL_RE.sub(" ", tag).replace("_", " ").split() if w]
    if not words:
        return tag
    out = [words[0]]
    out += [w.lower() if w.istitle() else w for w in words[1:]]
    head = out[0]
    return (head[:1].upper() + head[1:]) + ("" if len(out) == 1 else " " + " ".join(out[1:]))


def label_of(tag: str) -> str:
    return LABELS.get(tag) or _readable(tag)


def _format_value(tag: str, raw: str) -> str:
    """Render dates, booleans and money the way the ЕИС card does."""
    value = " ".join(raw.split())
    if value.lower() in ("true", "false"):
        return "да" if value.lower() == "true" else "нет"
    m = _DATETIME_RE.match(value)
    if m:
        y, mo, d, hh, mm = m.groups()
        return f"{d}.{mo}.{y} {hh}:{mm}"
    m = _DATE_RE.match(value)
    if m:
        y, mo, d = m.groups()
        return f"{d}.{mo}.{y}"
    if any(h in tag for h in MONEY_HINTS):
        try:
            num = float(value)
        except ValueError:
            return value
        whole, frac = f"{num:,.2f}".split(".")
        return f"{whole.replace(',', ' ')},{frac} руб."
    return value


# --------------------------------------------------------------------------- #
# document model
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    """One labelled block of a document — the unit we chunk on."""

    title: str
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join([self.title] + self.lines)

    @property
    def n_chars(self) -> int:
        return len(self.text())


@dataclass
class ProcurementDoc:
    doc_type: str                       # epNotificationEF2020, contract, …
    doc_id: str                         # purchaseNumber / regNum / hash fallback
    title: str
    sections: list[Section] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def file_name(self) -> str:
        return f"{self.doc_type}_{self.doc_id}.xml"

    def to_dict(self) -> dict[str, Any]:
        return {"doc_type": self.doc_type, "doc_id": self.doc_id, "title": self.title,
                "file_name": self.file_name, "meta": self.meta,
                "sections": [{"title": s.title, "lines": s.lines} for s in self.sections]}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _children(el: ET.Element) -> list[ET.Element]:
    return [c for c in el if _local(c.tag) not in NOISE_TAGS]


def _walk(el: ET.Element, prefix: list[str], out: list[tuple[tuple[str, ...], str, str]]) -> None:
    """Collect ``(group_prefix, label, value)`` leaves in document order.

    Repeated siblings (lots, purchase positions, requirements) are numbered so a
    flattened line still says *which* position an amount belongs to — losing that
    would make every position of every notification look alike, which is the one
    thing this corpus must not do.
    """
    kids = _children(el)
    if not kids:
        raw = (el.text or "").strip()
        if raw:
            out.append((tuple(prefix), label_of(_local(el.tag)), _format_value(_local(el.tag), raw)))
        return

    counts = Counter(_local(c.tag) for c in kids)
    seen: Counter[str] = Counter()
    for c in kids:
        lt = _local(c.tag)
        name = label_of(lt)
        if counts[lt] > 1:
            seen[lt] += 1
            name = f"{name} {seen[lt]}"
        _walk(c, prefix + [name] if _children(c) else prefix, out)


def _lines_from_leaves(leaves: list[tuple[tuple[str, ...], str, str]]) -> list[str]:
    """Turn leaves into document-like lines, emitting a header when the group changes."""
    lines: list[str] = []
    current: tuple[str, ...] = ()
    for prefix, name, value in leaves:
        if prefix != current:
            current = prefix
            if prefix:
                lines.append(f"{'. '.join(prefix[-2:])}:")
        lines.append(f"{name}: {value}")
    return lines


def _doc_root(root: ET.Element) -> ET.Element:
    """Unwrap ``export`` / SOAP-ish envelopes down to the actual document element."""
    seen = 0
    node = root
    while seen < 4:
        if _local(node.tag) not in ("export", "Envelope", "Body", "body", "dataInfo",
                                    "documentBody", "content"):
            return node
        kids = _children(node)
        if not kids:
            return node
        node = kids[0]
        seen += 1
    return node


def _first_value(root: ET.Element, tags: tuple[str, ...]) -> str:
    for tag in tags:
        for el in root.iter():
            if _local(el.tag) == tag and (el.text or "").strip():
                return (el.text or "").strip()
    return ""


def parse_xml(data: bytes | str, source_name: str = "") -> ProcurementDoc | None:
    """Parse one EIS XML document. Returns ``None`` if it is not parseable."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        log.warning("zakupki: unparseable XML %s (%s)", source_name or "<buffer>", e)
        return None

    doc = _doc_root(root)
    doc_type = _local(doc.tag)

    doc_id = _first_value(doc, ("purchaseNumber", "reestrNumber", "regNum", "docNumber",
                                "externalId", "id"))
    if not doc_id:
        base = os.path.basename(source_name or "")
        doc_id = os.path.splitext(base)[0] or f"{abs(hash(data)) % 10**12:012d}"
    doc_id = re.sub(r"[^\w.-]+", "_", doc_id)[:80]

    # Leaves that sit directly under the document root are its identity block;
    # every branch child becomes a section of its own.
    sections: list[Section] = []
    head_leaves: list[tuple[tuple[str, ...], str, str]] = []
    branches: list[tuple[str, ET.Element]] = []
    kids = _children(doc)
    counts = Counter(_local(c.tag) for c in kids)
    seen: Counter[str] = Counter()
    for c in kids:
        lt = _local(c.tag)
        if not _children(c):
            _walk(c, [], head_leaves)
            continue
        name = label_of(lt)
        if counts[lt] > 1:
            seen[lt] += 1
            name = f"{name} {seen[lt]}"
        branches.append((name, c))

    if head_leaves:
        sections.append(Section("Общие сведения", _lines_from_leaves(head_leaves)))

    rank = {t: i for i, t in enumerate(SECTION_ORDER)}
    branches.sort(key=lambda b: rank.get(_local(b[1].tag), len(SECTION_ORDER)))
    for name, el in branches:
        leaves: list[tuple[tuple[str, ...], str, str]] = []
        _walk(el, [], leaves)
        if not leaves:
            continue
        lines = _lines_from_leaves(leaves)
        # ``commonInfo`` renders as «Общие сведения» too — fold it into the
        # identity block rather than repeating the header two lines later.
        if sections and sections[-1].title == name:
            sections[-1].lines.extend(lines)
        else:
            sections.append(Section(name, lines))

    if not sections:
        log.warning("zakupki: %s carries no readable fields", source_name or doc_type)
        return None

    subject = _first_value(doc, ("purchaseObjectInfo", "objectInfo", "name"))
    title = f"{doc_type_label(doc_type)} № {doc_id}"
    if subject:
        title = f"{title}. {subject[:180]}"

    meta = {
        "source_file": source_name,
        "purchase_number": _first_value(doc, ("purchaseNumber",)),
        "published": _first_value(doc, ("publishDTInEIS", "docPublishDate", "createDateTime")),
        "customer": _first_value(doc, ("fullName", "shortName")),
        "placing_way": _first_value(doc, ("placingWay",)),
        "subject": subject,
    }
    return ProcurementDoc(doc_type=doc_type, doc_id=doc_id, title=title,
                          sections=sections, meta=meta)


# --------------------------------------------------------------------------- #
# input walking: directories, xml files, (nested) zips
# --------------------------------------------------------------------------- #
def iter_documents(path: str, *, max_depth: int = 3) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, xml_bytes)`` for every XML under ``path``.

    ``path`` may be a directory, a single ``.xml``, or a ``.zip`` — the data
    services return zips, sometimes of zips, so archives are opened recursively.
    """
    if os.path.isdir(path):
        for entry in sorted(os.listdir(path)):
            yield from iter_documents(os.path.join(path, entry), max_depth=max_depth)
        return
    low = path.lower()
    if low.endswith(".xml"):
        with open(path, "rb") as f:
            yield path, f.read()
    elif low.endswith(".zip"):
        with open(path, "rb") as f:
            yield from _iter_zip(f.read(), path, max_depth)
    elif low.endswith((".part", ".json", ".jsonl", ".md", ".txt")):
        return
    else:
        log.debug("zakupki: skipping %s (not xml/zip)", path)


def _iter_zip(blob: bytes, name: str, depth: int) -> Iterator[tuple[str, bytes]]:
    if depth <= 0:
        log.warning("zakupki: nesting limit reached at %s", name)
        return
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as e:
        log.warning("zakupki: bad archive %s (%s)", name, e)
        return
    with zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            inner = f"{name}!{info.filename}"
            low = info.filename.lower()
            if low.endswith(".xml"):
                yield inner, zf.read(info)
            elif low.endswith(".zip"):
                yield from _iter_zip(zf.read(info), inner, depth - 1)


def parse_path(path: str) -> Iterator[ProcurementDoc]:
    """Parse every document under ``path``, skipping the unreadable ones."""
    n_seen = n_ok = 0
    for name, blob in iter_documents(path):
        n_seen += 1
        doc = parse_xml(blob, name)
        if doc is not None:
            n_ok += 1
            yield doc
    log.info("zakupki: parsed %d/%d XML documents from %s", n_ok, n_seen, path)
