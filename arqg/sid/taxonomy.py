"""S2 — coverage taxonomy and cell sampling (plan §4).

Axis A1 (search mechanics) is the only coverage axis: each value changes the
distribution of actions an optimal policy takes. ``has_negation`` is an
orthogonal flag. Distractor type is a *generation parameter*, not an axis.

Submechanics are local diversity: they go into the composer prompt and force the
N candidates of one 1-of-N batch apart, but they do not expand the cell space
(§4.4). The plan has an LLM propose them per corpus; v1 ships a fixed grounded
list per mechanic — same role, no extra stage, and a corpus that cannot support
a cell simply fails its gates and shows up in ``gate_stats``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

MECHANICS: dict[str, dict[str, str]] = {
    "entity_chain": {
        "plan": "Последовательно зависимые запросы: второй запрос невозможно "
                "сформулировать, не узнав результат первого.",
        "compose": "Вопрос должен вести от стартовой сущности к целевой через "
                   "промежуточную, которая НЕ названа в вопросе.",
    },
    "constraint_intersection": {
        "plan": "Независимые ограничения, каждое сужает множество кандидатов; "
                "объединение (join) — в конце.",
        "compose": "Вопрос задаёт 2–3 независимых условия; ни одно по отдельности "
                   "не определяет ответ, вместе — определяют однозначно.",
    },
    "set_aggregation": {
        "plan": "Собрать всех членов множества по предикату; мощность множества "
                "заранее неизвестна.",
        "compose": "Вопрос требует перечислить/посчитать все объекты, "
                   "удовлетворяющие предикату. Не указывай, сколько их.",
    },
    "comparison": {
        "plan": "Симметричный ретривал двух и более сущностей, затем сравнение "
                "их свойств.",
        "compose": "Вопрос сравнивает 2+ сущности по общему свойству; данные о "
                   "каждой лежат в разных фрагментах.",
    },
    "temporal_resolution": {
        "plan": "Определить релевантную версию/дату; отбросить устаревшее.",
        "compose": "Вопрос требует выбрать актуальное (или относящееся к "
                   "конкретному моменту) значение среди нескольких во времени.",
    },
    "disambiguation_first": {
        "plan": "Стартовая сущность неоднозначна; сначала нужно прижать референт.",
        "compose": "Вопрос стартует с неоднозначного описания (без имени "
                   "собственного), которое сначала нужно разрешить.",
    },
}

SUBMECHANICS: dict[str, list[str]] = {
    "entity_chain": [
        "организация → её подразделение → свойство подразделения",
        "продукт → его создатель → свойство создателя",
        "событие → его участник → более поздний факт об участнике",
        "документ → упомянутый объект → характеристика объекта",
        "место → размещённый там объект → его показатель",
    ],
    "constraint_intersection": [
        "время + местоположение",
        "тип объекта + числовой порог",
        "принадлежность организации + период",
        "назначение объекта + материал/технология",
        "роль человека + место работы",
    ],
    "set_aggregation": [
        "все объекты одного класса у одной организации",
        "все события в заданном интервале",
        "все места, где присутствует организация",
        "все свойства/модели одного продуктового ряда",
        "все участники одного проекта",
    ],
    "comparison": [
        "две организации по одному показателю",
        "два периода одной организации",
        "два продукта по техническому параметру",
        "два места по масштабу",
        "план против факта",
    ],
    "temporal_resolution": [
        "актуальное значение против более раннего",
        "значение на конкретный год",
        "порядок событий во времени",
        "что изменилось между двумя датами",
        "первое/последнее событие в ряду",
    ],
    "disambiguation_first": [
        "описание роли вместо имени",
        "омонимичное название объекта",
        "объект, названный только по функции",
        "сущность, заданная косвенным признаком",
        "аббревиатура/код вместо полного имени",
    ],
}


@dataclass(frozen=True)
class Cell:
    mechanic: str
    submechanic: str
    has_negation: bool

    @property
    def key(self) -> str:
        return f"{self.mechanic}|{'neg' if self.has_negation else 'pos'}"


class CellSampler:
    """Round-robin over mechanics so coverage stays uniform by construction
    (plan §4.1 asks for ±30% of the mean), with submechanics rotated inside a
    1-of-N batch so the N candidates are not six copies of one template."""

    def __init__(self, cfg) -> None:
        self.mechanics = [m for m in (cfg.taxonomy.mechanics or list(MECHANICS))
                          if m in MECHANICS]
        self.negation_rate = cfg.taxonomy.negation_rate
        self.rng = random.Random(cfg.taxonomy.seed)
        self._i = 0

    def next_mechanic(self) -> str:
        m = self.mechanics[self._i % len(self.mechanics)]
        self._i += 1
        return m

    def batch(self, mechanic: str, n: int) -> list[Cell]:
        subs = SUBMECHANICS[mechanic][:]
        self.rng.shuffle(subs)
        return [Cell(mechanic, subs[i % len(subs)],
                     self.rng.random() < self.negation_rate)
                for i in range(n)]
