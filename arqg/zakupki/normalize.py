"""Text normalisation for ЕИС dumps: from form fields to something readable.

Published extracts of zakupki.gov.ru are dumps of a database, not prose. Left
alone they produce passages a retriever cannot do much with:

    Заказчик: КОМИТЕТ РЕСПУБЛИКИ АДЫГЕЯ ПО РЕГУЛИРОВАНИЮ КОНТРАКТНОЙ СИСТЕМЫ
    Начальная (максимальная) цена контракта: 571883910.00
    Дата размещения: 2020-08-11 16:27:33
    Описание позиций: Мониторы || Нулевой клиент

Nearly every customer name in the HF dump is shouted in caps (2957 of 3000
sampled), amounts are bare floats, dates are timestamps, and multi-valued cells
are packed with ``||``. This module fixes each of those, and it is deliberately
conservative: an abbreviation it does not recognise keeps its capitals rather
than being lower-cased into nonsense, and an amount it cannot parse is passed
through untouched. Silently mangling a customer name is worse than leaving it
loud — the name is a retrieval key.
"""
from __future__ import annotations

import re
from typing import Iterable

# --------------------------------------------------------------------------- #
# whitespace, quotes, dashes
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"[\s   ]+")
_CTRL_RE = re.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]")
_QUOT_RE = re.compile(r'"([^"]{1,200})"')
_MULTI_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_EMPTY = {"", "-", "—", "nan", "none", "null", "n/a", "нет данных", "0"}


def clean_text(value: object) -> str:
    """Collapse whitespace, drop control characters, normalise quotes and dashes."""
    if value is None:
        return ""
    text = _CTRL_RE.sub(" ", str(value))
    text = _WS_RE.sub(" ", text).strip()
    if text.lower() in _EMPTY:
        return ""
    text = _QUOT_RE.sub(r"«\1»", text)
    text = text.replace(" - ", " — ").replace("''", "»").replace("``", "«")
    return _MULTI_PUNCT_RE.sub(r"\1", text)


def is_empty(value: object) -> bool:
    return not clean_text(value)


# --------------------------------------------------------------------------- #
# organisation names
# --------------------------------------------------------------------------- #
#: Abbreviations that must survive case folding. Anything not listed here keeps
#: its capitals when it is short (≤4 letters) — the risk of destroying an unknown
#: acronym outweighs a few extra shouted words.
KNOWN_ABBREV = {
    # legal forms
    "ООО", "ОАО", "ЗАО", "ПАО", "АО", "НАО", "ИП", "ГУП", "МУП", "ФГУП", "АНО",
    "НКО", "НП", "ТСЖ", "СНТ", "КФХ", "ФКУ", "ГКУ", "МКУ", "ГАУ", "МАУ", "ГБУ",
    "МБУ", "ФГБУ", "ФГАУ", "ФГКУ", "ГБУЗ", "МБУЗ", "ГАУЗ", "ГКУЗ", "ФГБОУ",
    "ФГАОУ", "МБОУ", "МАОУ", "МКОУ", "ГБОУ", "ГАОУ", "МБДОУ", "МАДОУ", "ГБПОУ",
    "ГАПОУ", "СОШ", "ДОУ", "ДШИ", "ДЮСШ", "ЦРБ", "ГКБ", "ОКБ", "РКБ", "ЦСОН",
    "МФЦ", "ЖКХ", "ТЭЦ", "ГЭС", "АЭС", "НИИ", "КБ", "ОКУ",
    # geography / administration
    "РФ", "РА", "РБ", "РТ", "РМ", "РК", "РС", "ХМАО", "ЯНАО", "АО", "МО", "СПб",
    "ЕАО", "ЧР", "КЧР", "КБР", "УР", "ЧАО", "НАО",
    # misc
    "ФКП", "ФСИН", "МВД", "МЧС", "ФНС", "ПФР", "ФСС", "ОМС", "ДМС", "ГИБДД",
    "РЖД", "ЕИС", "ЭТП", "НМЦК", "ОКПД", "ОКЕИ", "КТРУ", "ИКЗ", "ИНН", "КПП",
    "ОГРН", "КБК", "СМП", "СОНКО", "ЛПУ", "IT", "ИТ", "ГСМ", "ПО",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_ROMAN_RE = re.compile(r"^[IVXLCM]+$")
#: lower-cased connectives that stay lower inside a name
_STOPWORDS = {
    "и", "в", "во", "на", "по", "для", "от", "до", "с", "со", "к", "о", "об",
    "при", "за", "из", "у", "им", "имени", "а", "но", "или", "не",
}
#: Address and administrative abbreviations that belong in lower case.
_LOWER_ABBREV = {
    "г", "гор", "обл", "окр", "рн", "пос", "пгт", "дер", "ул", "пр", "просп",
    "пер", "наб", "ш", "д", "стр", "корп", "лит", "оф", "каб", "мкр", "тел",
}


#: Length at or below which an all-caps token is assumed to be an acronym.
#: Russian organisation names are full of unlisted abbreviations (АРЦСМП, ГКУЗ,
#: УМВД); above this length an all-caps token is overwhelmingly a real word
#: (ЦЕНТР, УЧРЕЖДЕНИЕ, БОЛЬНИЦА). The line is drawn where the cost flips: a
#: shouted word is merely ugly, a lower-cased acronym is no longer recognisable.
ACRONYM_MAX_LEN = 4


def _capitalise(word: str) -> str:
    """Upper-case the first letter, and the first letter inside an opening quote."""
    low = word.lower()
    out = re.sub(r"^(\W*)(\w)", lambda m: m.group(1) + m.group(2).upper(), low, count=1)
    return out


def _fold_word(word: str, *, first: bool, proper: bool) -> str:
    core = "".join(_WORD_RE.findall(word))
    if not core:
        return word
    low = core.lower()
    # Connectives and address abbreviations first: «ПО», «В», «ОБЛ.» all pass the
    # acronym length test below, and none of them is an acronym.
    if not first and not proper and (low in _STOPWORDS or low in _LOWER_ABBREV):
        return word.lower()
    if core.upper() in KNOWN_ABBREV or _ROMAN_RE.match(core):
        return word.upper()
    if len(core) <= ACRONYM_MAX_LEN and core.isupper() and not first:
        return word                                  # unlisted acronym — leave it
    if first or proper:
        return _capitalise(word)
    return word.lower()


def smart_case(name: object) -> str:
    """``КОМИТЕТ РЕСПУБЛИКИ АДЫГЕЯ ПО РЕГУЛИРОВАНИЮ`` ->
    ``Комитет Республики Адыгея по регулированию``.

    Only shouted names are touched; anything already mixed-case is left alone,
    because a publisher that bothered with case usually got it right.
    """
    text = clean_text(name)
    if not text:
        return ""
    letters = "".join(_WORD_RE.findall(text))
    if not letters or not letters.isupper():
        return text
    out: list[str] = []
    for i, word in enumerate(text.split(" ")):
        # Only the word carrying the opening quote is capitalised: a quoted name
        # is sentence-cased, not title-cased — «Центр хозяйственного обеспечения»,
        # not «Центр Хозяйственного Обеспечения».
        opens_quote = word.lstrip().startswith(("«", '"', "("))
        proper = opens_quote or any(is_proper_noun(t) for t in _WORD_RE.findall(word))
        out.append(_fold_word(word, first=(i == 0), proper=proper))
    return " ".join(out)


# --------------------------------------------------------------------------- #
# numbers, money, dates
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"^-?\d+(?:[  ]\d{3})*(?:[.,]\d+)?$")
MONTHS = ("января", "февраля", "марта", "апреля", "мая", "июня",
          "июля", "августа", "сентября", "октября", "ноября", "декабря")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}))?")
_DOTTED_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})(?:\s(\d{2}):(\d{2}))?$")


def parse_number(value: object) -> float | None:
    text = clean_text(value).replace(" ", " ")
    if not _NUM_RE.match(text):
        return None
    try:
        return float(text.replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def format_money(value: object, *, currency: str = "руб.") -> str:
    """``571883910.00`` -> ``571 883 910,00 руб.`` (unparseable values pass through)."""
    num = parse_number(value)
    if num is None:
        return clean_text(value)
    whole, frac = f"{abs(num):,.2f}".split(".")
    sign = "-" if num < 0 else ""
    # A plain space, not a typographic thin/non-breaking one: the grouped
    # number has to stay matchable by a lexical index and by a user pasting
    # the amount straight out of the passage into a query.
    return f"{sign}{whole.replace(chr(44), chr(32))},{frac} {currency}"


def round_gloss(value: object) -> str:
    """A human-scale restatement of an amount: ``около 571,9 млн``.

    Queries say «закупка примерно на полмиллиарда», never «571883910.00», so the
    rounded form is what makes a lexical hit possible at all.
    """
    num = parse_number(value)
    if num is None or abs(num) < 1_000_000:
        return ""
    if abs(num) >= 1_000_000_000:
        return f"около {num / 1_000_000_000:.2f}".replace(".", ",") + " млрд руб."
    return f"около {num / 1_000_000:.1f}".replace(".", ",") + " млн руб."


#: Coarse bands, for filtering and for queries that name a scale rather than a sum.
PRICE_BUCKETS = (
    (1_000_000, "до 1 млн руб."),
    (10_000_000, "от 1 до 10 млн руб."),
    (100_000_000, "от 10 до 100 млн руб."),
    (1_000_000_000, "от 100 млн до 1 млрд руб."),
    (float("inf"), "свыше 1 млрд руб."),
)


def price_bucket(value: object) -> str:
    num = parse_number(value)
    if num is None:
        return ""
    for edge, label in PRICE_BUCKETS:
        if abs(num) < edge:
            return label
    return ""


def parse_date(value: object) -> tuple[str, str]:
    """Return ``(iso_date, "11 августа 2020 года")``; empty strings if unparseable."""
    text = clean_text(value)
    m = _DATE_RE.match(text)
    if m:
        y, mo, d, hh, mm = m.groups()
    else:
        m = _DOTTED_RE.match(text)
        if not m:
            return "", ""
        d, mo, y, hh, mm = m.groups()
    try:
        month = MONTHS[int(mo) - 1]
    except (ValueError, IndexError):
        return "", ""
    long = f"{int(d)} {month} {y} года"
    if hh:
        long += f", {hh}:{mm}"
    return f"{y}-{mo}-{d}", long


def format_percent(value: object) -> str:
    """``30.00%`` / ``5.0`` -> ``30 %`` / ``5 %``."""
    text = clean_text(value).rstrip("%").strip()
    num = parse_number(text)
    if num is None:
        return clean_text(value)
    shown = f"{num:.2f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"{shown} %"


# --------------------------------------------------------------------------- #
# multi-valued cells
# --------------------------------------------------------------------------- #
_SPLIT_RE = re.compile(r"\s*\|\|\s*|\s*;\s{2,}")


def split_items(value: object, *, limit: int = 40) -> list[str]:
    """``"Мониторы || Нулевой клиент"`` -> ``["Мониторы", "Нулевой клиент"]``."""
    text = clean_text(value)
    if not text:
        return []
    parts = [clean_text(p) for p in _SPLIT_RE.split(text)]
    return [p for p in parts if p][:limit]


def dedupe_indices(phrases: Iterable[str]) -> list[int]:
    """Positions worth keeping: first occurrence wins, later echoes are dropped.

    The dumps repeat themselves — ``purchase_name``, ``okpd2_names`` and
    ``item_descriptions`` are often the same sentence three times over. Three
    copies in one passage teach a retriever nothing and inflate every similarity
    score computed over the corpus.

    Indices rather than strings, because the caller needs to know *which field*
    survived: an ОКПД2 name that merely echoes the subject should not be
    rendered, and comparing the text again downstream would not tell it apart
    from the subject itself.
    """
    kept: list[int] = []
    seen: list[str] = []
    for i, phrase in enumerate(phrases):
        text = clean_text(phrase)
        if not text:
            continue
        low = text.lower()
        if any(low == s or low in s for s in seen):
            continue
        kept.append(i)
        seen.append(low)
    return kept


def dedupe_phrases(phrases: Iterable[str]) -> list[str]:
    """:func:`dedupe_indices`, as the surviving strings."""
    items = [clean_text(p) for p in phrases]
    return [items[i] for i in dedupe_indices(items)]


def sentence(text: str) -> str:
    """Capitalise and terminate a rendered clause."""
    text = clean_text(text)
    if not text:
        return ""
    text = text[:1].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


# --------------------------------------------------------------------------- #
# reference data
# --------------------------------------------------------------------------- #
#: Region code -> name. Public reference table (the codes ЕИС puts in
#: ``orgRegion`` / dumps put in ``region_code``); a dump that ships only the code
#: is unsearchable by region name without it.
REGIONS: dict[str, str] = {
    "1": "Республика Адыгея",
    "2": "Республика Башкортостан",
    "3": "Республика Бурятия",
    "4": "Республика Алтай",
    "5": "Республика Дагестан",
    "6": "Республика Ингушетия",
    "7": "Кабардино-Балкарская Республика",
    "8": "Республика Калмыкия",
    "9": "Карачаево-Черкесская Республика",
    "10": "Республика Карелия",
    "11": "Республика Коми",
    "12": "Республика Марий Эл",
    "13": "Республика Мордовия",
    "14": "Республика Саха (Якутия)",
    "15": "Республика Северная Осетия — Алания",
    "16": "Республика Татарстан",
    "17": "Республика Тыва",
    "18": "Удмуртская Республика",
    "19": "Республика Хакасия",
    "20": "Чеченская Республика",
    "21": "Чувашская Республика",
    "22": "Алтайский край",
    "23": "Краснодарский край",
    "24": "Красноярский край",
    "25": "Приморский край",
    "26": "Ставропольский край",
    "27": "Хабаровский край",
    "28": "Амурская область",
    "29": "Архангельская область",
    "30": "Астраханская область",
    "31": "Белгородская область",
    "32": "Брянская область",
    "33": "Владимирская область",
    "34": "Волгоградская область",
    "35": "Вологодская область",
    "36": "Воронежская область",
    "37": "Ивановская область",
    "38": "Иркутская область",
    "39": "Калининградская область",
    "40": "Калужская область",
    "41": "Камчатский край",
    "42": "Кемеровская область — Кузбасс",
    "43": "Кировская область",
    "44": "Костромская область",
    "45": "Курганская область",
    "46": "Курская область",
    "47": "Ленинградская область",
    "48": "Липецкая область",
    "49": "Магаданская область",
    "50": "Московская область",
    "51": "Мурманская область",
    "52": "Нижегородская область",
    "53": "Новгородская область",
    "54": "Новосибирская область",
    "55": "Омская область",
    "56": "Оренбургская область",
    "57": "Орловская область",
    "58": "Пензенская область",
    "59": "Пермский край",
    "60": "Псковская область",
    "61": "Ростовская область",
    "62": "Рязанская область",
    "63": "Самарская область",
    "64": "Саратовская область",
    "65": "Сахалинская область",
    "66": "Свердловская область",
    "67": "Смоленская область",
    "68": "Тамбовская область",
    "69": "Тверская область",
    "70": "Томская область",
    "71": "Тульская область",
    "72": "Тюменская область",
    "73": "Ульяновская область",
    "74": "Челябинская область",
    "75": "Забайкальский край",
    "76": "Ярославская область",
    "77": "г. Москва",
    "78": "г. Санкт-Петербург",
    "79": "Еврейская автономная область",
    "80": "Забайкальский край",
    "82": "Республика Крым",
    "83": "Ненецкий АО",
    "86": "Ханты-Мансийский АО — Югра",
    "87": "Чукотский АО",
    "89": "Ямало-Ненецкий АО",
    "91": "Республика Крым",
    "92": "г. Севастополь",
    "99": "г. Байконур",
}


def region_name(code: object, fallback: str = "") -> str:
    key = clean_text(code).lstrip("0") or clean_text(code)
    return REGIONS.get(key, clean_text(fallback))


#: Words that are part of a region's *type*, not its name.
_REGION_GENERIC = {
    "республика", "республики", "область", "области", "край", "края", "округ",
    "округа", "автономная", "автономный", "автономного", "город", "города", "г",
    "ао", "кузбасс", "алания",
}

#: Proper nouns worth keeping capitalised inside a shouted organisation name.
#: Derived from the region table rather than hand-listed: «КОМИТЕТ РЕСПУБЛИКИ
#: АДЫГЕЯ» should not come back as «комитет республики адыгея», and the region is
#: by far the most common proper noun inside a customer's name.
PROPER_NOUNS: set[str] = {
    token.lower()
    for name in REGIONS.values()
    for token in _WORD_RE.findall(name)
    if token.lower() not in _REGION_GENERIC and len(token) > 2
}

#: Six-letter stems of the same nouns, so declined forms match too — a customer
#: is «... Тюменской области», and the table only knows «Тюменская».
_PROPER_STEMS: set[str] = {n[:6] for n in PROPER_NOUNS if len(n) >= 6}


def is_proper_noun(token: str) -> bool:
    low = token.lower()
    return low in PROPER_NOUNS or (len(low) >= 6 and low[:6] in _PROPER_STEMS)
