"""Russian-language prompts for generation and verification.

Design goals encoded here:
* Questions must REQUIRE combining >= 2 neighbouring chunks (true multi-hop).
* Questions must read like REAL USER questions: several style variants are
  sampled (simple user, novice, expert, search query) so the dataset matches
  the query distribution a retriever sees in production, not exam phrasing.
* Questions must be self-contained (no "согласно тексту", no dangling pronouns).
* Questions must be paraphrased in the asker's own words — copying rare
  phrases verbatim from the source makes retrieval trivially easy.
* Answers must be fully grounded in the cited chunks.
* The judge independently shrinks the gold set to strictly-necessary chunks.
"""
from __future__ import annotations

from .schema import Chunk


def format_chunks(chunks: list[Chunk]) -> str:
    """Render chunks with explicit, citable ids for the model to reference."""
    parts = []
    for c in chunks:
        parts.append(f"[CHUNK {c.id}]\n{c.raw_text.strip()}\n[/CHUNK {c.id}]")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# GENERATION — question styles
# --------------------------------------------------------------------------- #
# Each style mimics a kind of real user. Weights are configured in
# `generate.styles`; one style is sampled per question and recorded in the
# dataset so you can analyse/balance the mix afterwards.
STYLES: dict[str, str] = {
    "simple_user": (
        "Обычный пользователь. Пиши коротко и просто, разговорным языком — как "
        "реальный человек пишет в чат поддержки или в поисковую строку. Примерно "
        "5–15 слов. Без канцелярита и сложных оборотов. Естественные начала: "
        "«как…», «сколько…», «когда…», «можно ли…», «что будет, если…», «где…», "
        "«кто…». Одно простое предложение, без перечислений внутри вопроса."
    ),
    "novice": (
        "Новичок, не знакомый с терминологией предметной области. Он описывает, "
        "что ему нужно, СВОИМИ бытовыми словами и НЕ использует специальные "
        "термины из текста — заменяй термины простыми описаниями (например, "
        "вместо точного названия устройства — «прибор, который …»). Вопрос "
        "короткий и простой, как в живой переписке."
    ),
    "expert": (
        "Специалист в теме. Использует корректную терминологию из предметной "
        "области, формулирует точно и по делу, одним предложением. Без воды, но "
        "и без искусственной наукообразности."
    ),
    "search_query": (
        "Короткий поисковый запрос, как в строке поиска: 3–8 слов, может быть "
        "БЕЗ вопросительного знака и без глагола (например: «сроки гарантии "
        "после замены модуля»). Никаких вежливых оборотов. Запрос всё равно "
        "должен однозначно подразумевать конкретный ответ."
    ),
}

DEFAULT_STYLE = "simple_user"

GEN_SYSTEM = (
    "Ты — эксперт по созданию обучающих датасетов для систем информационного "
    "поиска (retrieval) и агентных RAG-систем. Ты отлично имитируешь то, как "
    "РЕАЛЬНЫЕ пользователи формулируют вопросы: просто, естественно, без "
    "канцелярита. Ты пишешь вопросы и ответы строго на русском языке, не "
    "выдумываешь факты и опираешься только на предоставленный текст."
)

GEN_USER_TEMPLATE = """Ниже даны несколько ИДУЩИХ ПОДРЯД фрагментов (chunks) из одного документа. У каждого фрагмента есть идентификатор в скобках [CHUNK <id>].

{chunks}

ЗАДАЧА. Придумай ОДИН вопрос, который мог бы задать реальный пользователь, ища эту информацию. Главное требование: чтобы дать полный и точный ответ, необходимо ОБЪЕДИНИТЬ информацию минимум из ДВУХ разных фрагментов. По одному любому фрагменту ответить полностью должно быть НЕВОЗМОЖНО.

СТИЛЬ ВОПРОСА (обязательно соблюдай):
{style}

Рабочие стратегии связывания фрагментов:
- «мост»: сущность/термин вводится в одном фрагменте, а нужная деталь о ней — в соседнем;
- агрегация: ответ собирается из значений, разнесённых по фрагментам;
- сравнение: сопоставить два объекта/числа/события из разных фрагментов;
- условие + следствие: правило в одном фрагменте, применение/исключение — в другом.

ТРЕБОВАНИЯ К ВОПРОСУ:
1. Вопрос самодостаточен: понятен без доступа к фрагментам. ЗАПРЕЩЕНЫ отсылки «согласно тексту», «в документе», «в данном фрагменте», «выше». Подставляй конкретные имена, названия, термины — ровно столько, сколько нужно, чтобы вопрос был однозначным.
2. Формулируй СВОИМИ словами: не копируй дословно редкие формулировки и целые словосочетания из фрагментов — перефразируй, используй синонимы. Иначе поиск по вопросу становится тривиальным.
3. Вопрос звучит естественно. Реальные пользователи НЕ задают вопросов вида «Какие сведения о компании, основанной в 1998 году, и её продукте приведены…» — это перегруженный экзаменационный стиль, его избегай.
4. Не раскрывай ответ внутри вопроса.
5. У вопроса есть конкретный ответ, который опирается на фрагменты, а не на общеизвестные факты.
6. Ответ должен полностью следовать из указанных фрагментов: краткий, но достаточный.

ПРИМЕР (на постороннюю тему, только для калибровки стиля):
- Плохо: «Согласно тексту, какие требования предъявляются к руководителю подразделения, открытого в 2015 году?» — отсылка к тексту, канцелярит.
- Плохо: «Какова численность сотрудников завода и кто его руководитель?» — двойной вопрос-перечисление, так люди не спрашивают.
- Хорошо: «Кто руководит заводом, где делают аккумуляторы для буёв „Тритон“?» — коротко, естественно, а ответ требует связать фрагмент о продукте и фрагмент о заводе.

Укажи в `required_chunk_ids` МИНИМАЛЬНЫЙ набор идентификаторов фрагментов (из приведённых выше), которые действительно НЕОБХОДИМЫ для ответа. В наборе должно быть не менее двух идентификаторов.

Верни СТРОГО валидный JSON-объект без какого-либо текста вокруг:
{{
  "reasoning": "<кратко: какие сведения из каких фрагментов соединяются>",
  "question": "<вопрос на русском, в заданном стиле>",
  "answer": "<точный ответ на русском>",
  "required_chunk_ids": ["<id>", "<id>", ...],
  "question_type": "multi_hop | aggregation | comparison | condition",
  "self_contained": true
}}"""


def gen_user(chunks: list[Chunk], style: str = DEFAULT_STYLE) -> str:
    style_text = STYLES.get(style, STYLES[DEFAULT_STYLE])
    return GEN_USER_TEMPLATE.format(chunks=format_chunks(chunks), style=style_text)


# --------------------------------------------------------------------------- #
# GENERATION (v2) — document-level, simple OR hard, no anchor limit
# --------------------------------------------------------------------------- #
# This second prompt version operates over a WHOLE document (many fragments) and
# produces either a simple single-passage question or a hard question that may
# rest on as many passages as needed. Output schema is identical (gold via
# required_chunk_ids) so it flows through the same verify/finalize stages.
DOC_GEN_SYSTEM = (
    "Ты — эксперт по созданию обучающих датасетов для систем информационного "
    "поиска (retrieval). Тебе дают целый документ, разбитый на пронумерованные "
    "фрагменты. Ты умеешь придумывать как простые фактологические вопросы по "
    "одному фрагменту, так и сложные вопросы, требующие собрать сведения из "
    "многих фрагментов по всему документу. Ты пишешь строго на русском языке, "
    "не выдумываешь факты и опираешься только на предоставленный текст."
)

# Per-difficulty instruction blocks injected into the shared template.
DOC_DIFFICULTY: dict[str, str] = {
    "simple": (
        "СЛОЖНОСТЬ: ПРОСТОЙ вопрос.\n"
        "Выбери ОДИН фрагмент, в котором целиком содержится ответ, и задай по нему "
        "короткий конкретный вопрос. Полный и точный ответ должен находиться "
        "ПОЛНОСТЬЮ внутри этого одного фрагмента — другие фрагменты не нужны. "
        "В `required_chunk_ids` укажи РОВНО ОДИН идентификатор — тот самый фрагмент."
    ),
    "hard": (
        "СЛОЖНОСТЬ: СЛОЖНЫЙ вопрос.\n"
        "Задай вопрос, для полного ответа на который необходимо СОБРАТЬ и "
        "СОПОСТАВИТЬ сведения из НЕСКОЛЬКИХ фрагментов, по возможности разнесённых "
        "по всему документу (двух, трёх или больше — ограничения сверху нет). "
        "По любому одному фрагменту ответить полностью должно быть НЕВОЗМОЖНО. "
        "Подходящие приёмы: объединение нескольких фактов в один ответ, агрегация "
        "или перечисление разнесённых по тексту пунктов, сравнение объектов из "
        "разных частей документа, связка «правило + исключение». В "
        "`required_chunk_ids` перечисли ВСЕ фрагменты, действительно необходимые "
        "для ответа (их должно быть не менее двух), и НЕ добавляй лишних."
    ),
}

DOC_GEN_USER_TEMPLATE = """Ниже дан документ, разбитый на пронумерованные фрагменты. У каждого фрагмента есть идентификатор в скобках [CHUNK <id>].

{chunks}

ЗАДАЧА. Придумай ОДИН вопрос на русском языке, который мог бы задать реальный пользователь, ища эту информацию.

{difficulty}

СТИЛЬ ВОПРОСА (обязательно соблюдай):
{style}

ОБЩИЕ ТРЕБОВАНИЯ:
1. Вопрос самодостаточен и понятен без доступа к документу. ЗАПРЕЩЕНЫ отсылки «согласно тексту», «в документе», «в данном фрагменте», «выше». Подставляй конкретные имена, названия, термины.
2. Формулируй СВОИМИ словами, не копируй дословно редкие словосочетания из фрагментов — перефразируй.
3. Не раскрывай ответ внутри вопроса.
4. У вопроса есть конкретный ответ, опирающийся на указанные фрагменты, а не на общеизвестные факты.
5. Ответ краткий, но достаточный, и полностью следует из указанных фрагментов.
6. Все идентификаторы в `required_chunk_ids` бери ТОЛЬКО из приведённых выше фрагментов.

Верни СТРОГО валидный JSON-объект без какого-либо текста вокруг:
{{
  "reasoning": "<кратко: какие сведения из каких фрагментов используются>",
  "question": "<вопрос на русском, в заданном стиле>",
  "answer": "<точный ответ на русском>",
  "required_chunk_ids": ["<id>", ...],
  "question_type": "factoid | multi_hop | aggregation | comparison | condition",
  "self_contained": true
}}"""


def doc_gen_user(chunks: list[Chunk], difficulty: str, style: str = DEFAULT_STYLE) -> str:
    return DOC_GEN_USER_TEMPLATE.format(
        chunks=format_chunks(chunks),
        difficulty=DOC_DIFFICULTY.get(difficulty, DOC_DIFFICULTY["hard"]),
        style=STYLES.get(style, STYLES[DEFAULT_STYLE]),
    )


# --------------------------------------------------------------------------- #
# JUDGE 1 — groundedness / standalone / specificity
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "Ты — строгий рецензент качества датасетов для информационного поиска. Ты "
    "беспристрастно проверяешь вопросы и отвечаешь только в формате JSON."
)

GROUNDEDNESS_TEMPLATE = """Тебе дан ВОПРОС, предложенный ОТВЕТ и набор фрагментов-источников.

ВОПРОС:
{question}

ПРЕДЛОЖЕННЫЙ ОТВЕТ:
{answer}

ФРАГМЕНТЫ-ИСТОЧНИКИ:
{chunks}

Важно про стиль: вопрос намеренно написан так, как пишут реальные пользователи — коротко, разговорно, иногда в виде поискового запроса без вопросительного знака. Простота и разговорность — НЕ недостаток и НЕ причина для отрицательных оценок. Оценивай только указанные ниже критерии.

Оцени строго и верни JSON:
{{
  "supported": <true, если ответ полностью подтверждается фрагментами и ничего не выдумано; иначе false>,
  "answer_correct": <true, если предложенный ответ фактически верен относительно фрагментов>,
  "standalone": <true, если вопрос понятен сам по себе и НЕ ссылается на «текст/документ/фрагмент/выше»>,
  "specific": <true, если у вопроса есть однозначный ответ; краткая или разговорная формулировка — это нормально, false только если вопрос реально допускает много разных ответов>,
  "answerable_from_world_knowledge": <true, если на вопрос можно уверенно ответить БЕЗ этих фрагментов, по общеизвестным фактам>,
  "notes": "<краткое обоснование на русском>"
}}"""


def groundedness_user(question: str, answer: str, chunks: list[Chunk]) -> str:
    return GROUNDEDNESS_TEMPLATE.format(
        question=question, answer=answer, chunks=format_chunks(chunks)
    )


# --------------------------------------------------------------------------- #
# JUDGE 2 — minimality / necessity (the multi-hop guarantee)
# --------------------------------------------------------------------------- #
MINIMALITY_TEMPLATE = """АНАЛИЗ НЕОБХОДИМОСТИ ФРАГМЕНТОВ.

ВОПРОС:
{question}

КАНДИДАТЫ-ФРАГМЕНТЫ (каждый со своим id):
{chunks}

Определи МИНИМАЛЬНЫЙ набор фрагментов, который строго НЕОБХОДИМ и ДОСТАТОЧЕН, чтобы полностью ответить на вопрос. Фрагмент включается, только если без него ответить полностью нельзя.

Затем проверь: можно ли полностью ответить на вопрос, используя лишь ОДИН какой-либо фрагмент (любой из перечисленных)?

Верни JSON:
{{
  "necessary_chunk_ids": ["<id>", ...],
  "answerable": <true, если по минимальному набору на вопрос можно ответить полностью>,
  "single_chunk_sufficient": <true, если какого-то ОДНОГО фрагмента уже достаточно для полного ответа>,
  "notes": "<кратко на русском>"
}}"""


def minimality_user(question: str, chunks: list[Chunk]) -> str:
    return MINIMALITY_TEMPLATE.format(question=question, chunks=format_chunks(chunks))


# --------------------------------------------------------------------------- #
# CLUE DECOMPOSITION — for collect-all-positives
# --------------------------------------------------------------------------- #
CLUE_SYSTEM = (
    "Ты — эксперт по декомпозиции вопросов для систем информационного поиска. "
    "Ты разбиваешь пару «вопрос+ответ» на атомарные факты-подсказки (clues), по "
    "которым потом ищут источники в корпусе. Отвечаешь только в формате JSON на "
    "русском языке."
)

CLUE_TEMPLATE = """Дан ВОПРОС, его ЭТАЛОННЫЙ ОТВЕТ и фрагменты-источники, на которых ответ основан.

ВОПРОС:
{question}

ЭТАЛОННЫЙ ОТВЕТ:
{answer}

ФРАГМЕНТЫ-ИСТОЧНИКИ:
{chunks}

Разбей вопрос на АТОМАРНЫЕ подсказки (clues) — минимальные самодостаточные факты, которые в совокупности необходимы и достаточны, чтобы получить ответ. Правила:
1. Каждая подсказка — это УТВЕРЖДЕНИЕ (факт), а не вопрос.
2. Подсказка самодостаточна: содержит конкретные сущности (имена, названия, даты), без местоимений и без отсылок «этот/тот/выше». По ней можно искать в корпусе и проверять отдельный фрагмент.
3. Один атомарный факт = одна подсказка. Как правило, на каждый фрагмент-источник приходится своя подсказка.
4. Не добавляй фактов, которых нет во фрагментах. Не дроби один факт на части, которые по отдельности теряют смысл.
5. Для каждой подсказки укажи `source_gold_ids` — идентификаторы тех фрагментов-источников (из приведённых выше), откуда взят этот факт.

Верни СТРОГО валидный JSON-объект:
{{
  "clues": [
    {{"clue": "<факт-утверждение на русском>", "source_gold_ids": ["<id>", ...]}},
    ...
  ]
}}"""


def clue_user(question: str, answer: str, chunks: list[Chunk]) -> str:
    return CLUE_TEMPLATE.format(
        question=question, answer=answer, chunks=format_chunks(chunks))


# --------------------------------------------------------------------------- #
# CLUE ENTAILMENT — does a retrieved passage state the clue's fact?
# --------------------------------------------------------------------------- #
ENTAIL_SYSTEM = (
    "Ты — строгий экстрактор фактов. Тебе дают ФАКТ и ФРАГМЕНТ текста; ты "
    "решаешь, утверждается ли этот факт во фрагменте. Отвечаешь только JSON."
)

ENTAIL_TEMPLATE = """ФАКТ (подсказка):
{clue}

ФРАГМЕНТ:
{passage}

Вопрос: подтверждает ли ФРАГМЕНТ этот ФАКТ? Ответь «да» только если фрагмент действительно содержит/утверждает именно этот факт (те же сущности и значения), пусть даже другими словами или это другой документ-дубликат. Если фрагмент лишь упоминает тему, относится к другому объекту, противоречит факту или не содержит его — ответь «нет».

Верни JSON:
{{
  "supports": <true, если фрагмент подтверждает факт; иначе false>,
  "notes": "<очень кратко на русском>"
}}"""


def entailment_user(clue: str, passage: str) -> str:
    return ENTAIL_TEMPLATE.format(clue=clue, passage=passage.strip())


# --------------------------------------------------------------------------- #
# MuSiQue -> dialogue: anaphora rewriting of a bridge hop question
# --------------------------------------------------------------------------- #
# A MuSiQue hop question references an earlier hop's answer with a "#k" token.
# The earlier answer is already visible in the transcript (the bot said it), so
# the follow-up must refer back to it with a pronoun or a definite description
# instead of naming the entity — that is the conversational anaphora we want.
# MuSiQue is an English dataset, so these prompts are in English to match.
ANAPHORA_SYSTEM = (
    "You turn multi-hop sub-questions into natural conversational follow-up "
    "messages. A user is chatting with an assistant. An earlier answer is "
    "already visible in the transcript, so the follow-up must refer back to it "
    "with a pronoun or a short definite description (e.g. 'he', 'that city', "
    "'this company') instead of repeating the entity by name. Reply only in JSON."
)

ANAPHORA_TEMPLATE = """Here is the conversation so far:

{transcript}

The user's next question, in raw decomposed form, is:
    {raw_question}

In this raw question, each #N placeholder stands for an answer the assistant ALREADY gave above:
{ref_map}

Rewrite the raw question as the user's next chat message. Requirements:
1. Replace every #N placeholder with a natural anaphoric reference — a pronoun or a short definite description — to the corresponding earlier answer. Do NOT write the entity's name.
2. Keep the rest of the question's meaning exactly; do not add or drop any constraint.
3. Sound like a real person's short follow-up message, not an exam question.
4. One sentence, in English.

Return STRICT JSON and nothing else:
{{"message": "<the rewritten user message>"}}"""


def anaphora_user(transcript: str, raw_question: str, ref_map: str) -> str:
    return ANAPHORA_TEMPLATE.format(
        transcript=transcript, raw_question=raw_question, ref_map=ref_map)
