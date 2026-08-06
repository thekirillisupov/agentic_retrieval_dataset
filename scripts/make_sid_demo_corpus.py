#!/usr/bin/env python3
"""Generate a small deterministic Russian corpus for SID dry runs.

The sample corpus in ``tests/sample_chunks.jsonl`` is 8 chunks — too small for
subgraph mining, density percentiles or a distractor band to mean anything. This
builds ~60 chunks over 15 documents with the properties the pipeline needs:

* rare entities repeated across *different* documents (bridges for S1),
* dates, amounts and codes (discriminating attributes for L2 perturbation),
* a couple of near-duplicate documents (non-singleton fact groups for G_REP),
* enough topical spread that τ_sim / τ_low percentiles are not degenerate.

    python scripts/make_sid_demo_corpus.py tests/sample_corpus_sid.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys

ORGS = [
    ("Север", "Архангельск", "морской электроники"),
    ("Мерида", "Калининград", "портовой логистики"),
    ("Аквилон", "Мурманск", "энергетического оборудования"),
    ("Гранит-Инжиниринг", "Петрозаводск", "промышленной автоматизации"),
    ("Ветра", "Астрахань", "речного судостроения"),
]
PRODUCTS = [
    ("Тритон-3", "гидроакустический буй", "солёности и температуры воды"),
    ("Барс-7", "навигационный модуль", "координат и курса судна"),
    ("Кайра-2", "подводный дрон", "состояния подводных конструкций"),
    ("Лоцман-5", "терминал диспетчеризации", "загрузки причалов"),
    ("Сивуч-1", "автономный регистратор", "вибрации силовых установок"),
]
PEOPLE = ["Алексей Громов", "Ирина Ковалёва", "Пётр Заславский",
          "Марина Верещагина", "Дмитрий Аникеев", "Ольга Тарасенко"]
PARKS = [
    ("Зелёные холмы", "южного хребта", "горного оленя"),
    ("Синие озёра", "северного плато", "серого журавля"),
    ("Каменная гряда", "восточного предгорья", "снежного барса"),
]


def org_doc(i: int, rng: random.Random) -> tuple[str, list[str]]:
    name, city, field = ORGS[i]
    prod, kind, measures = PRODUCTS[i]
    person = PEOPLE[i]
    founded = 1992 + i * 3
    launch = founded + 9
    plant = launch + 8
    staff = 120 + i * 45
    turnover = 2 + i
    code = f"ТУ-{4000 + i * 137}"
    return f"org_{i:02d}.txt", [
        f"Компания «{name}» была основана в {founded} году в городе {city} группой "
        f"инженеров, ранее работавших на профильном заводе. Первоначально "
        f"предприятие занималось ремонтом оборудования и поставками запасных "
        f"частей, а через несколько лет перешло к собственным разработкам в "
        f"области {field}.",

        f"Главным продуктом компании «{name}» стал {kind} «{prod}», который вышел "
        f"на рынок в {launch} году. Устройство предназначено для измерения "
        f"{measures} и передаёт данные по спутниковому каналу связи. Изделие "
        f"выпускается по техническим условиям {code}.",

        f"В {plant} году «{name}» открыла второе производственное подразделение, "
        f"где было налажено серийное изготовление комплектующих для линейки "
        f"«{prod}». Новый участок снизил себестоимость продукции почти на треть и "
        f"обеспечил работой около {staff} специалистов. Руководителем "
        f"подразделения был назначен инженер {person}.",

        f"К 2021 году годовой оборот предприятия «{name}» превысил {turnover} млрд "
        f"рублей, а доля экспортных поставок достигла {12 + i * 4} процентов. "
        f"Основными заказчиками оставались научные институты и профильные "
        f"предприятия отрасли.",
    ]


def project_doc(i: int, rng: random.Random) -> tuple[str, list[str]]:
    """Cross-references organisations and products — the cross-document bridges."""
    a, b = ORGS[i % len(ORGS)], ORGS[(i + 2) % len(ORGS)]
    pa, pb = PRODUCTS[i % len(PRODUCTS)], PRODUCTS[(i + 2) % len(PRODUCTS)]
    year = 2016 + i
    budget = 340 + i * 55
    return f"project_{i:02d}.txt", [
        f"Совместный проект «Открытое море {year}» был запущен в {year} году при "
        f"участии компаний «{a[0]}» и «{b[0]}». Общий бюджет программы составил "
        f"{budget} млн рублей, из которых чуть более половины пришлось на "
        f"опытно-конструкторские работы.",

        f"В рамках программы «Открытое море {year}» на испытательном полигоне "
        f"проверялась совместная работа изделий «{pa[0]}» и «{pb[0]}». Испытания "
        f"проходили в акватории вблизи города {a[1]} и продолжались одиннадцать "
        f"месяцев.",

        f"По итогам испытаний {year + 1} года было решено доработать интерфейс "
        f"передачи данных изделия «{pb[0]}». Ответственным за доработку назначили "
        f"специалистов компании «{b[0]}», работы завершились в {year + 2} году.",

        f"Отчёт о программе «Открытое море {year}» отмечает, что стоимость "
        f"эксплуатации оборудования снизилась на {8 + i * 3} процентов по "
        f"сравнению с предыдущим поколением техники.",
    ]


def park_doc(i: int, rng: random.Random) -> tuple[str, list[str]]:
    name, ridge, animal = PARKS[i]
    founded = 1990 + i * 4
    area = 12 + i * 5
    routes = 4 + i
    center = founded + 18
    visitors = 150 + i * 40
    return f"park_{i:02d}.txt", [
        f"Национальный парк «{name}» был учреждён в {founded} году на территории "
        f"площадью около {area} тысяч гектаров в предгорьях {ridge}. Парк "
        f"создавался для сохранения редких видов и ограничения хозяйственной "
        f"деятельности в водоохранной зоне.",

        f"Через территорию парка «{name}» проходят {routes} оборудованных "
        f"туристических маршрута общей протяжённостью свыше восьмидесяти "
        f"километров. Самый популярный из них выводит к смотровой площадке над "
        f"долиной.",

        f"В {center} году на территории парка «{name}» был открыт "
        f"научно-исследовательский центр, занимающийся мониторингом популяции "
        f"{animal} и изучением местной флоры. Финансирование центра ведётся из "
        f"средств целевой программы.",

        f"Ежегодно национальный парк «{name}» посещают около {visitors} тысяч "
        f"туристов, причём основной поток приходится на летние месяцы. Доходы от "
        f"продажи разрешений направляются на содержание маршрутов.",
    ]


def doc_title(file_name: str) -> str:
    """A breadcrumb title, as the real corpora carry: the document's path in the
    knowledge base, not a headline. S1 mines within a folder of this path, so the
    demo corpus has to have folders — with a flat name it would only ever
    exercise the unscoped fallback. Companies and the joint programmes that
    reference them share a branch, which is where the cross-document bridges
    are; the parks are a separate branch that must not bridge into them.
    """
    stem = file_name.removesuffix(".txt")
    is_summary = stem.endswith("_svod")
    kind, num = stem.removesuffix("_svod").split("_")
    i = int(num)
    if kind == "park":
        branch, folder, leaf = "Заповедники", "Национальные парки", f"Парк «{PARKS[i][0]}»"
    elif kind == "project":
        branch, folder = "Морская электроника", "Совместные программы"
        leaf = f"Программа «Открытое море {2016 + i}»"
    else:
        branch, folder = "Морская электроника", "Профили компаний"
        leaf = f"Компания «{ORGS[i][0]}»"
    return f"База знаний/{branch}/{'Сводные обзоры' if is_summary else folder}/{leaf}"


def build() -> list[dict]:
    rng = random.Random(7)
    docs: list[tuple[str, list[str]]] = []
    for i in range(len(ORGS)):
        docs.append(org_doc(i, rng))
    for i in range(4):
        docs.append(project_doc(i, rng))
    for i in range(len(PARKS)):
        docs.append(park_doc(i, rng))

    # near-duplicate sources: the same facts restated in another document, so
    # G_REP has real multi-member fact groups to find
    for src in ("org_00.txt", "project_01.txt"):
        body = dict(docs)[src]
        docs.append((src.replace(".txt", "_svod.txt"),
                     [f"Из сводного обзора отрасли. {p}" for p in body[:3]]))

    rows: list[dict] = []
    for file_name, paras in docs:
        for idx, text in enumerate(paras):
            rows.append({"file_name": file_name, "index": idx, "raw_text": text,
                         "document_id": file_name.replace(".txt", ""),
                         "title": doc_title(file_name)})
    return rows


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_corpus_sid.jsonl"
    rows = build()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} chunks over "
          f"{len({r['file_name'] for r in rows})} documents -> {out}")


if __name__ == "__main__":
    main()
