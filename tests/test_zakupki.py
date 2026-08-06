"""Offline tests for the ЕИС / zakupki.gov.ru export (no network, no token).

The network half is exercised at the envelope/response level only: zakupki.gov.ru
refuses connections from non-Russian address space, so a live call cannot be part
of a test suite. Everything downstream of the raw archive is fully covered.
"""
import csv
import io
import os
import sys
import zipfile
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.utils import read_jsonl
from arqg.zakupki.client import EisError, build_envelope, parse_response
from arqg.zakupki.corpus import (ChunkOptions, build_corpus, chunk_document,
                                 duplicate_report, jaccard)
from arqg.zakupki.parse import iter_documents, parse_path, parse_xml
from arqg.zakupki.facets import Facets, SourceRef
from arqg.zakupki.merge import (SourceSpec, build_merged, is_mergeable,
                                merge_sources)
from arqg.zakupki.normalize import (clean_text, dedupe_indices, format_money,
                                    format_percent, parse_date, price_bucket,
                                    region_name, round_gloss, smart_case,
                                    split_items)
from arqg.zakupki.tabular import (PROFILES, detect_profile, iter_docs,
                                  parse_column_overrides)

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_XML = os.path.join(HERE, "sample_zakupki_notification.xml")


def _sample_bytes() -> bytes:
    with open(SAMPLE_XML, "rb") as f:
        return f.read()


def _variant(n: int) -> bytes:
    """Same template, different customer / dates / amounts — a real near-duplicate."""
    text = _sample_bytes().decode("utf-8")
    return (text
            .replace("0173200001425000417", f"01732000014250004{17 + n:02d}")
            .replace("«Городская поликлиника № 14»", f"«Городская поликлиника № {14 + n}»")
            .replace("1487300.50", f"{1487300.50 + n * 1000:.2f}")
            .replace("2025-06-19", f"2025-06-{19 + n % 5:02d}")
            .encode("utf-8"))


# --------------------------------------------------------------------------- #
# envelope / response
# --------------------------------------------------------------------------- #
def test_envelope_preserves_tag_order():
    env = build_envelope(
        "getDocsByOrgRegionRequest",
        [("selectionParams", [
            ("orgRegion", "72"),
            ("subsystemType", "PRIZ"),
            ("documentType44", "epNotificationEF2020"),
            ("periodInfo", [("exactDate", "2025-06-11")]),
        ])],
        namespace="http://zakupki.gov.ru/fz44/get-docs-ip/ws",
        token="tok-1", request_id="req-1", created="2025-06-11T10:00:00")

    assert "<individualPerson_token>tok-1</individualPerson_token>" in env
    # index first, then the selection params in exactly the order given
    order = [env.index(t) for t in ("<index>", "<id>req-1</id>", "<mode>PROD</mode>",
                                    "<orgRegion>", "<subsystemType>",
                                    "<documentType44>", "<exactDate>")]
    assert order == sorted(order)


def test_envelope_escapes_and_drops_empties():
    env = build_envelope("getDocsByReestrNumberRequest",
                         [("selectionParams", [("subsystemType", "PRIZ"),
                                               ("reestrNumber", "a&b"),
                                               ("unused", "")])],
                         namespace="ns", token="t<k")
    assert "<reestrNumber>a&amp;b</reestrNumber>" in env
    assert "&lt;k" in env
    assert "<unused>" not in env


def test_parse_response_reads_archive_urls_regardless_of_prefix():
    xml = """<?xml version="1.0"?>
    <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body><ns7:getDocsByOrgRegionResponse xmlns:ns7="urn:x">
        <dataInfo><archiveUrl>https://int44.zakupki.gov.ru/a/1.zip</archiveUrl></dataInfo>
        <dataInfo><archiveUrl>https://int44.zakupki.gov.ru/a/2.zip</archiveUrl></dataInfo>
      </ns7:getDocsByOrgRegionResponse></soap:Body></soap:Envelope>"""
    assert parse_response(xml) == ["https://int44.zakupki.gov.ru/a/1.zip",
                                   "https://int44.zakupki.gov.ru/a/2.zip"]


def test_parse_response_raises_on_fault():
    xml = """<?xml version="1.0"?>
    <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body><soap:Fault>
        <faultcode>soap:Server</faultcode>
        <faultstring>Некорректный токен</faultstring>
      </soap:Fault></soap:Body></soap:Envelope>"""
    with pytest.raises(EisError, match="Некорректный токен"):
        parse_response(xml)


def test_parse_response_empty_selection_is_not_an_error():
    xml = """<?xml version="1.0"?>
    <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body><getDocsByOrgRegionResponse/></soap:Body></soap:Envelope>"""
    assert parse_response(xml) == []


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_notification_fields_and_sections():
    doc = parse_xml(_sample_bytes(), SAMPLE_XML)
    assert doc is not None
    assert doc.doc_type == "epNotificationEF2020"
    assert doc.doc_id == "0173200001425000417"
    assert doc.file_name == "epNotificationEF2020_0173200001425000417.xml"
    assert "Поставка бумаги офисной" in doc.title

    body = "\n".join(s.text() for s in doc.sections)
    assert "Номер извещения: 0173200001425000417" in body
    assert "ИНН: 7203001234" in body
    assert "Электронный аукцион" in body
    # dates and money are humanised
    assert "11.06.2025 09:14" in body
    assert "1 487 300,50 руб." in body
    # signature blobs never reach the text
    assert "MIIJ0AYJKoZIhvcNAQcC" not in body


def test_repeated_elements_stay_distinguishable():
    doc = parse_xml(_sample_bytes(), SAMPLE_XML)
    body = "\n".join(s.text() for s in doc.sections)
    assert "Объект закупки 1" in body and "Объект закупки 2" in body
    # each position keeps its own quantity, so the two are not interchangeable
    assert "Количество: 1200" in body and "Количество: 310" in body


def test_unknown_tags_degrade_to_readable_labels():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <export xmlns="urn:x"><someNewDoc2027>
      <purchaseNumber>123</purchaseNumber>
      <brandNewFieldName>значение</brandNewFieldName>
      <nested><anotherOne>42</anotherOne></nested>
    </someNewDoc2027></export>""".encode("utf-8")
    doc = parse_xml(xml, "new.xml")
    body = "\n".join(s.text() for s in doc.sections)
    assert "Brand new field name: значение" in body
    assert "Another one: 42" in body


def test_unparseable_xml_returns_none():
    assert parse_xml(b"<not xml", "broken.xml") is None


def test_iter_documents_reads_nested_zips(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("notice.xml", _sample_bytes())
    outer = tmp_path / "archive.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("day1/inner.zip", inner.getvalue())
        zf.writestr("day1/other.xml", _variant(1))
        zf.writestr("readme.txt", b"ignored")
    names = [n for n, _ in iter_documents(str(outer))]
    assert len(names) == 2
    assert all(n.endswith(".xml") for n in names)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def test_chunks_are_ordered_and_bounded():
    doc = parse_xml(_sample_bytes(), SAMPLE_XML)
    opts = ChunkOptions(max_chars=600, merge_below=200, min_chars=50)
    texts = chunk_document(doc, opts)
    assert len(texts) >= 2
    assert all(len(t) <= 600 + 200 for t in texts)   # split happens on line bounds
    assert texts[0].startswith("Общие сведения")


def test_long_section_split_repeats_its_header():
    doc = parse_xml(_sample_bytes(), SAMPLE_XML)
    texts = chunk_document(doc, ChunkOptions(max_chars=260, merge_below=0, min_chars=20))
    assert any("(продолжение" in t for t in texts)


def test_build_corpus_emits_pipeline_schema(tmp_path):
    archive = tmp_path / "raw" / "a.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(4):
            zf.writestr(f"n{i}.xml", _variant(i))

    corpus = tmp_path / "zakupki_index.jsonl"
    stats = build_corpus(parse_path(str(tmp_path / "raw")), str(corpus),
                         docs_path=str(tmp_path / "docs.jsonl"))
    assert stats["n_documents_parsed"] == 4
    assert stats["document_types"] == {"epNotificationEF2020": 4}

    records = list(read_jsonl(str(corpus)))
    assert records and stats["n_chunks"] == len(records)
    for rec in records:
        assert set(rec) == {"file_name", "index", "raw_text", "document_id", "title"}
        assert rec["raw_text"].strip()
    # index is 0-based and contiguous per file — arqg.windows relies on index ± 1
    per_file = {}
    for rec in records:
        per_file.setdefault(rec["file_name"], []).append(rec["index"])
    for name, idx in per_file.items():
        assert idx == list(range(len(idx))), name
    assert len(per_file) == 4


def test_duplicate_file_names_are_emitted_once(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("first.xml", "second.xml"):          # same purchase, fetched twice
        (raw / name).write_bytes(_sample_bytes())
    corpus = tmp_path / "c.jsonl"
    stats = build_corpus(parse_path(str(raw)), str(corpus))
    assert stats["n_documents_parsed"] == 2
    assert len({r["file_name"] for r in read_jsonl(str(corpus))}) == 1


# --------------------------------------------------------------------------- #
# near-duplicate report — the reason this source exists
# --------------------------------------------------------------------------- #
def test_report_separates_exact_from_structural_duplicates(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(6):
        (raw / f"n{i}.xml").write_bytes(_variant(i))
    corpus = tmp_path / "c.jsonl"
    build_corpus(parse_path(str(raw)), str(corpus))
    records = list(read_jsonl(str(corpus)))

    report = duplicate_report(records)
    assert report["n_files"] == 6
    # boilerplate (требования к участникам) repeats verbatim across notifications
    assert report["exact_duplicates"]["n_groups"] >= 1
    # ... and the amount/date-bearing passages are template-identical, not exact
    structural = report["structural_duplicates"]
    assert structural["n_groups"] >= report["exact_duplicates"]["n_groups"]
    assert structural["share_of_chunks"] > report["exact_duplicates"]["share_of_chunks"]
    assert structural["mean_jaccard_within_group"] > 0.5
    assert structural["examples"][0]["group_size"] >= 2


def test_jaccard_is_bounded():
    a = "поставка бумаги офисной а4 для нужд учреждения в срок до 30 сентября"
    assert jaccard(a, a) == 1.0
    assert jaccard(a, "совершенно другой текст про строительство моста") == 0.0


# --------------------------------------------------------------------------- #
# third-party dumps (the no-token path)
# --------------------------------------------------------------------------- #
def _dump(tmp_path, rows, header, sep=","):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "dump.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=sep)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


KAGGLE_HEADER = ["tender_id", "tender_name", "start_price", "tender_security",
                 "advance_money", "currency", "publication_date", "selection_phase",
                 "legislation", "url", "procedure", "for_small_business",
                 "customer_region_code", "customer_region", "customer_name",
                 "customer_inn", "winner_name", "winner_inn", "final_price"]


def _kaggle_row(n=0):
    return [f"017320000142500{400 + n:04d}",
            "Выполнение работ по капитальному ремонту кровли",
            f"{571883910 + n}.00", "28594195.50", "30.00%", "RUB",
            "2020-08-11 16:27:33", "Признана несостоявшейся", "44-ФЗ",
            "https://zakupki.gov.ru/epz/order/notice/ea44/view/common-info.html",
            "Электронный аукцион", "FALSE", "52", "Нижегородская область",
            "Администрация Вачского Муниципального Района", "5208002260", "", "", ""]


def test_profile_is_detected_from_the_header(tmp_path):
    path = _dump(tmp_path, [_kaggle_row()], KAGGLE_HEADER)
    profile = detect_profile(path)
    assert profile is not None and profile.name == "kaggle_biggest"
    assert "PDDL" in profile.licence


def test_row_renders_prose_not_form_fields(tmp_path):
    path = _dump(tmp_path, [_kaggle_row()], KAGGLE_HEADER)
    docs = list(iter_docs(path, PROFILES["kaggle_biggest"]))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "0173200001425000400"
    assert doc.file_name.endswith(".txt")     # a card, not an EIS XML document
    body = "\n".join(s.text() for s in doc.sections)

    assert "Заказчиком выступает Администрация" in body
    assert ("начальная (максимальная) цена контракта составляет "
            "571 883 910,00 руб. (около 571,9 млн руб.)".capitalize() in body
            or "571 883 910,00 руб. (около 571,9 млн руб.)" in body)
    assert "11 августа 2020 года" in body
    assert "предусмотрен аванс в размере 30 %" in body.lower()
    assert "Заказчик: " not in body and "Дата размещения: " not in body


def test_facets_reach_the_title_because_the_index_embeds_it(tmp_path):
    path = _dump(tmp_path, [_kaggle_row()], KAGGLE_HEADER)
    doc = next(iter(iter_docs(path, PROFILES["kaggle_biggest"])))
    assert doc.title.startswith("Закупка № 0173200001425000400")
    assert "Нижегородская область" in doc.title and "2020" in doc.title
    assert "капитальному ремонту кровли" in doc.title


def test_price_gloss_and_bucket_help_vague_queries(tmp_path):
    path = _dump(tmp_path, [_kaggle_row()], KAGGLE_HEADER)
    doc = next(iter(iter_docs(path, PROFILES["kaggle_biggest"])))
    body = "\n".join(s.text() for s in doc.sections)
    assert "около 571,9 млн руб." in body
    assert "от 100 млн до 1 млрд руб." in body


def test_empty_columns_never_produce_dangling_clauses(tmp_path):
    path = _dump(tmp_path, [_kaggle_row()], KAGGLE_HEADER)
    doc = next(iter(iter_docs(path, PROFILES["kaggle_biggest"])))
    body = "\n".join(s.text() for s in doc.sections)
    assert "победителем" not in body.lower()   # winner_name is empty in this row
    for line in body.splitlines():
        assert not line.rstrip().endswith(("—", "-", ":", "«"))


def test_doc_id_recovered_when_the_id_column_lost_digits(tmp_path):
    """XLSX dumps store 19-digit numbers as floats; the tail is gone for good."""
    path = _dump(tmp_path, [["1.762000055240009e+17",
                             "https://www.rts-tender.ru/x/number/0176200005524000887/etpName/fks",
                             "Поставка лекарственных препаратов"]],
                 ["Num_trade", "Trade", "Name"])
    profile = replace(PROFILES["hf_medicines"],
                      columns={"doc_id": "Num_trade", "url": "Trade", "subject": "Name"})
    doc = next(iter(iter_docs(path, profile)))
    assert doc.doc_id == "0176200005524000887"
    assert "Закупка № 0176200005524000887" in doc.sections[0].text()


def test_distinct_rows_do_not_collapse_onto_one_document(tmp_path):
    path = _dump(tmp_path, [_kaggle_row(i) for i in range(5)], KAGGLE_HEADER)
    corpus = tmp_path / "c.jsonl"
    stats = build_corpus(iter_docs(path, PROFILES["kaggle_biggest"]), str(corpus),
                         opts=ChunkOptions(merge_below=0, min_chars=40))
    assert stats["n_documents_parsed"] == stats["n_files"] == 5
    # every card must yield ≥2 chunks or it can never form a neighbour window
    per_file = {}
    for rec in read_jsonl(str(corpus)):
        per_file[rec["file_name"]] = per_file.get(rec["file_name"], 0) + 1
    assert min(per_file.values()) >= 2


def test_semicolon_dump_law_codes_and_item_lists(tmp_path):
    header = ["pn_lot_anon", "fz", "region_code", "min_publish_date", "purchase_name",
              "lot_name", "lot_price", "okpd2_code", "okpd2_names",
              "additional_code", "additional_code_names", "item_descriptions"]
    row = ["pn_lot_7031618", "44fz", "52", "2019-08-26",
           "Приобретение компьютерной техники", "", "123500.0", "26.2",
           "Компьютеры портативные массой не более 10 кг", "", "",
           "Мониторы || Нулевой клиент"]
    path = _dump(tmp_path, [row], header, sep=";")
    assert detect_profile(path).name == "zakupkihack"
    doc = next(iter(iter_docs(path, PROFILES["zakupkihack"])))
    body = "\n".join(s.text() for s in doc.sections)
    assert "в соответствии с 44-ФЗ" in body                  # 44fz -> 44-ФЗ
    assert "коду ОКПД2 26.2" in body
    assert "входит 2 позиции: Мониторы; Нулевой клиент" in body
    # the dump carries only a region code; the name makes it searchable
    assert "Нижегородская область" in body


def test_repeated_text_across_columns_is_collapsed(tmp_path):
    header = ["pn_lot_anon", "fz", "min_publish_date", "purchase_name", "lot_name",
              "lot_price", "okpd2_code", "okpd2_names", "item_descriptions"]
    row = ["pn_lot_1", "44fz", "2019-08-26", "Услуги по проведению финансового аудита",
           "", "123500.0", "69.2", "Услуги по проведению финансового аудита",
           "Услуги по проведению финансового аудита || Услуги по проведению финансового аудита"]
    path = _dump(tmp_path, [row], header, sep=";")
    doc = next(iter(iter_docs(path, PROFILES["zakupkihack"])))
    body = "\n".join(s.text() for s in doc.sections)
    assert body.lower().count("услуги по проведению финансового аудита") <= 2


def test_column_overrides_are_validated():
    assert parse_column_overrides(["subject=tender_name"]) == {"subject": "tender_name"}
    with pytest.raises(SystemExit, match="unknown field"):
        parse_column_overrides(["nonsense=col"])
    with pytest.raises(SystemExit, match="canonical=source"):
        parse_column_overrides(["subject"])


def test_unrecognised_dump_is_not_guessed_at(tmp_path):
    path = _dump(tmp_path, [["a", "b"]], ["foo", "bar"])
    assert detect_profile(path) is None


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_shouted_names_become_readable():
    assert smart_case("КОМИТЕТ РЕСПУБЛИКИ АДЫГЕЯ ПО РЕГУЛИРОВАНИЮ КОНТРАКТНОЙ СИСТЕМЫ") \
        == "Комитет республики Адыгея по регулированию контрактной системы"
    # the region survives even in a declined form the table does not list
    assert "Тюменской" in smart_case('ГКУ ТЮМЕНСКОЙ ОБЛАСТИ "УПРАВЛЕНИЕ ДОРОГ"')
    # a quoted name is sentence-cased, not title-cased
    assert smart_case('ФЕДЕРАЛЬНОЕ УЧРЕЖДЕНИЕ "ЦЕНТР ХОЗЯЙСТВЕННОГО ОБЕСПЕЧЕНИЯ"') \
        == "Федеральное учреждение «Центр хозяйственного обеспечения»"


def test_known_abbreviations_and_short_acronyms_keep_their_capitals():
    out = smart_case('ГБУЗ РА "АРЦСМП И МК"')
    assert out.startswith("ГБУЗ РА")
    assert "МК" in out                      # ≤4 letters: assumed to be an acronym
    assert " и " in out                     # a connective is not an acronym


def test_already_cased_names_are_left_alone():
    for name in ("Администрация г. Волгодонска", "ООО «Ромашка»"):
        assert smart_case(name) == name


def test_money_dates_and_percentages():
    assert format_money("571883910.00") == "571 883 910,00 руб."
    assert format_money("не указано") == "не указано"      # unparseable passes through
    assert round_gloss("571883910.00") == "около 571,9 млн руб."
    assert round_gloss("7808705950") == "около 7,81 млрд руб."
    assert round_gloss("123500") == ""                     # too small to be worth a gloss
    assert price_bucket("571883910.00") == "от 100 млн до 1 млрд руб."
    assert parse_date("2020-08-11 16:27:33") == ("2020-08-11", "11 августа 2020 года, 16:27")
    assert parse_date("что-то не то") == ("", "")
    assert format_percent("30.00%") == "30 %"


def test_control_characters_and_multivalued_cells():
    assert clean_text("  а\x00б   в ") == "а б в"
    assert clean_text("nan") == ""
    assert split_items("Мониторы || Нулевой клиент") == ["Мониторы", "Нулевой клиент"]


def test_dedupe_keeps_the_first_occurrence_and_reports_positions():
    phrases = ["Поставка бумаги", "поставка бумаги", "Поставка бумаги А4 и картриджей"]
    assert dedupe_indices(phrases) == [0, 2]


def test_region_code_resolves_to_a_searchable_name():
    assert region_name("52") == "Нижегородская область"
    assert region_name("02") == "Республика Башкортостан"
    assert region_name("999", fallback="Неизвестно") == "Неизвестно"


# --------------------------------------------------------------------------- #
# merging sources
# --------------------------------------------------------------------------- #
def _facets(number, **kw):
    f = Facets(purchase_number=number, sources=[SourceRef(dataset=kw.pop("dataset", "a"))])
    for key, value in kw.items():
        setattr(f, key, value)
    return f


def test_merge_is_a_field_level_union():
    a = _facets("0173200001425000400", subject="Поставка бумаги", customer="Заказчик А")
    b = _facets("0173200001425000400", winner="ООО «Ромашка»", contract_date="2020-09-01",
                dataset="b")
    merged = a.merge(b)
    assert merged.subject == "Поставка бумаги"          # kept from the first source
    assert merged.winner == "ООО «Ромашка»"             # filled in from the second
    assert {s.dataset for s in merged.sources} == {"a", "b"}


def test_merge_prefers_the_longer_of_two_truncated_values():
    a = _facets("1", subject="Поставка бумаги")
    b = _facets("1", subject="Поставка бумаги офисной А4 для нужд учреждения")
    assert a.merge(b).subject.endswith("для нужд учреждения")


def test_only_real_registry_numbers_are_held_open_for_merging():
    assert is_mergeable(_facets("0173200001425000400"))
    assert not is_mergeable(_facets("pn_lot_7031618"))   # anonymised: can never match
    assert not is_mergeable(_facets("zakupkihack_row00000001"))


def test_records_from_two_dumps_become_one_document(tmp_path):
    left = _dump(tmp_path / "a", [_kaggle_row()], KAGGLE_HEADER)
    right_header = ["Num_trade", "Trade", "Name", "Date_contract"]
    right = _dump(tmp_path / "b", [[
        "0173200001425000400",
        "https://zakupki.gov.ru/x/number/0173200001425000400/y",
        "Выполнение работ по капитальному ремонту кровли здания школы",
        "2020-09-01"]], right_header)

    specs = [SourceSpec(left, PROFILES["kaggle_biggest"]),
             SourceSpec(right, replace(PROFILES["hf_medicines"],
                                       columns={"doc_id": "Num_trade", "url": "Trade",
                                                "subject": "Name",
                                                "contract_date": "Date_contract"}))]
    merged = list(merge_sources(specs))
    assert len(merged) == 1
    facets = merged[0]
    assert facets.customer_inn == "5208002260"                    # only in the left dump
    assert facets.contract_date == "2020-09-01"                   # only in the right one
    assert facets.subject.endswith("здания школы")                # the fuller of the two
    assert {s.dataset for s in facets.sources} == {"kaggle_biggest", "hf_medicines"}


def test_build_merged_writes_corpus_and_both_sidecars(tmp_path):
    path = _dump(tmp_path, [_kaggle_row(i) for i in range(3)], KAGGLE_HEADER)
    corpus = tmp_path / "c.jsonl"
    meta = tmp_path / "c_meta.jsonl"
    docs = tmp_path / "c_docs.jsonl"
    stats = build_merged([SourceSpec(path, PROFILES["kaggle_biggest"])], str(corpus),
                         meta_path=str(meta), docs_path=str(docs))

    chunks = list(read_jsonl(str(corpus)))
    metas = list(read_jsonl(str(meta)))
    documents = list(read_jsonl(str(docs)))
    assert stats["n_files"] == len(documents) == 3
    assert len(metas) == len(chunks) == stats["n_chunks"]
    assert all(set(c) == {"file_name", "index", "raw_text", "document_id", "title"}
               for c in chunks)
    # the sidecar addresses chunks the same way the pipeline does
    assert [m["chunk_id"] for m in metas] == [f"{c['file_name']}::{c['index']}"
                                              for c in chunks]


def test_chunk_metadata_carries_the_facets_a_query_filters_on(tmp_path):
    path = _dump(tmp_path, [_kaggle_row()], KAGGLE_HEADER)
    meta = tmp_path / "m.jsonl"
    build_merged([SourceSpec(path, PROFILES["kaggle_biggest"])],
                 str(tmp_path / "c.jsonl"), meta_path=str(meta))
    record = next(iter(read_jsonl(str(meta))))
    assert record["region"] == "Нижегородская область"
    assert record["year"] == "2020"
    assert record["price_bucket"] == "от 100 млн до 1 млрд руб."
    assert record["procedure"] == "Электронный аукцион"
    assert record["source_url"].endswith("regNumber=0173200001425000400")
    assert record["section"]                     # which part of the document this is
    # provenance paths are NOT repeated per chunk — that doubled the sidecar
    assert "sources" not in record


def test_document_metadata_carries_the_paths(tmp_path):
    path = _dump(tmp_path, [_kaggle_row()], KAGGLE_HEADER)
    docs = tmp_path / "d.jsonl"
    build_merged([SourceSpec(path, PROFILES["kaggle_biggest"])],
                 str(tmp_path / "c.jsonl"), docs_path=str(docs))
    record = next(iter(read_jsonl(str(docs))))
    assert record["sources"][0]["path"] == path              # the dump file on disk
    assert record["sources"][0]["locator"] == "row 0"        # the row inside it
    assert "kaggle" in record["sources"][0]["origin"]        # where it was published
    assert "PDDL" in record["licences"][0]
    assert record["source_url"].startswith("https://zakupki.gov.ru/epz/order/notice/")
    assert record["chunk_ids"] and len(record["chunk_ids"]) == record["n_chunks"]
    assert "Нижегородская область" in record["keywords"]


def test_portal_url_is_rebuilt_from_the_registry_number():
    f = _facets("0173200001425000400", law="44-ФЗ", url="https://example.org/search?q=1")
    assert f.notice_url.endswith("/epz/order/notice/view/common-info.html"
                                 "?regNumber=0173200001425000400")
    f = _facets("31705770243", law="223-ФЗ", url="https://example.org/search?q=1")
    assert "/223/purchase/public/purchase/info/" in f.notice_url
    # nothing recognisable: keep whatever link the dump gave us
    f = _facets("pn_lot_1", url="https://example.org/search?q=1")
    assert f.notice_url == "https://example.org/search?q=1"
