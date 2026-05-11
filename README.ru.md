# Mnemosyne — плагин долговременной памяти для Hermes Agent

> 🇬🇧 Read in English: [README.md](README.md)

**Mnemosyne** (Μνημοσύνη — древнегреческая титанида памяти, мать девяти муз) — это плагин долговременной памяти для [Hermes Agent](https://github.com/NousResearch/hermes-agent). Он объединяет двух существующих провайдеров памяти — **[Honcho](https://github.com/plastic-labs/honcho)** (модель пользователя: стиль, предпочтения, личность в диалоге) и **[Hindsight](https://hindsight.app)** (хранилище фактов: граф сущностей, мульти-стратегический семантический поиск) — за единым интерфейсом `MemoryProvider`, и добавляет поверх дат-теги, счётчик повторений, разрешение конфликтов, recovery после сбоев и явное забывание.

Естественно сочетается со своей мифологической дочерью [Hermes-Mneme](https://github.com/johnnykor82/hermes-mneme) — отдельным плагином для **краткосрочной памяти в рамках сессии**.

## Зачем композитный провайдер памяти

Hermes Agent разрешает только одного внешнего провайдера памяти одновременно. Это проблема, потому что два лучших провайдера в экосистеме — взаимодополняющие, а не взаимозаменяемые:

- **Honcho** силён в моделировании *кто такой пользователь* — диалектическое рассуждение, peer card, стиль общения, устойчивые предпочтения. Но он слаб в точном фактологическом поиске: нет угасания, нет дедупликации, LLM-генерируемые представления.
- **Hindsight** — наоборот: сильный фактологический стор, граф сущностей, гибридный семантический + keyword recall, но не моделирует пользователя как личность.

Mnemosyne оборачивает оба провайдера в единый `MemoryProvider` и маршрутизирует каждый вызов туда, где он будет обработан лучше:

- **Записи (writes)** → в оба провайдера (Honcho кормит модель пользователя; Hindsight строит граф фактов).
- **Авто-инъекция (prefetch)** → факты из Hindsight + peer card из Honcho; шумный session summary / user representation от Honcho подавляется (рекомендуется выставить в Honcho `recallMode: tools`).
- **Tools** → шесть кураторских инструментов `memory_*` с чёткими role-based описаниями, которые под капотом маршрутизируются в нужный провайдер.

Композитный слой также добавляет:

1. **Дат-теги** на каждый запомненный факт (`fact_store.db`) — чтобы запрос "что мы обсуждали на прошлой неделе?" реально работал.
2. **Счётчик повторений** — отслеживает, как часто факт упоминается; важные факты поднимаются выше в recall.
3. **Явное забывание** — инструмент `memory_forget` + CLI, soft-delete по сигнатуре с семантической дедупликацией, чтобы тот же факт не вернулся обратно.

## Статус

**Scaffold (Stage A).** Сейчас — чистая fan-out делегация в двух внутренних провайдеров. Композитный слой (дат-теги, счётчик повторений, conflict resolver, забывание, anchor card, prefetch fusion) — в roadmap ниже.

## Требования

- **Python 3.11+**
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** установлен и работает (плагин предполагает Hermes venv в `~/.hermes/hermes-agent/venv`, переопределяется через `HERMES_VENV`).
- **[Honcho](https://honcho.dev)** — Python-клиент (`honcho-ai`). Уже стоит, если вы пользовались Honcho как memory provider.
- **[Hindsight](https://hindsight.app)** — ставится через `hermes memory setup` с выбором `hindsight`.
- **macOS или Linux.** Плагин на чистом Python, инсталлятор работает на обеих ОС.

## Установка

```bash
git clone https://github.com/johnnykor82/mnemosyne.git \
  ~/.hermes/plugins/mnemosyne
cd ~/.hermes/plugins/mnemosyne
./install.sh
```

Установщик найдёт Hermes venv (по умолчанию `~/.hermes/hermes-agent/venv`, переопределить можно через `HERMES_VENV=...`), поставит две Python-зависимости (`honcho-ai`, `hindsight-client>=0.4.22`) и проверит, что они импортируются.

Плагин живёт в `$HERMES_HOME/plugins/mnemosyne/` (по умолчанию `~/.hermes/plugins/mnemosyne/`) и переживает `git pull` самого `hermes-agent` — это user-installed аддон, а не часть ядра.

## Активация

```bash
hermes config set memory.provider mnemosyne
hermes gateway restart
```

Проверить, что плагин загрузился:

```bash
tail -f ~/.hermes/logs/agent.log | grep -i mnemosyne
```

Откатиться на предыдущего провайдера:

```bash
hermes config set memory.provider honcho   # или тот, который у вас был
hermes gateway restart
```

## Конфигурация

Mnemosyne читает конфигурацию из `mnemosyne/config.json` (создаётся в директории плагина при первом запуске). Большинство значений по умолчанию работают из коробки — плагин рассчитан на то, чтобы свежая установка "Just Works" против локального LiteLLM-прокси на `http://localhost:8000`.

Что можно переопределить (через секцию `hindsight_env` в `config.json` или через shell env vars — shell важнее):

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | `openai` | Имя провайдера эмбеддингов для Hindsight |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL` | `http://localhost:8000/v1` | Endpoint эмбеддингов (по умолчанию локальный LiteLLM) |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` | `sk-local-litellm` | Плейсхолдер для локального прокси. Замените, если идёте в реальный API. |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` | `jina-embeddings-v5-text-small-retrieval-mlx` | Модель эмбеддингов по умолчанию (Jina MLX локально) |
| `HINDSIGHT_API_RERANKER_PROVIDER` | `cohere` | Имя провайдера реранкера |
| `HINDSIGHT_API_RERANKER_COHERE_BASE_URL` | `http://localhost:4000/v1/rerank` | Endpoint реранкера |
| `HINDSIGHT_API_RERANKER_COHERE_API_KEY` | `sk-local-litellm` | Плейсхолдер для локального прокси |
| `HINDSIGHT_API_RERANKER_COHERE_MODEL` | `rerank` | Модель реранкера по умолчанию |

Внутреннее хранилище плагина:

- `fact_store.db` — SQLite-стор для дат-тегов, счётчиков повторений, сигнатур забытого. Создаётся при первом запуске.
- `recovery_cursor.json` — offset в `~/.hermes/sessions/` для импорта после сбоев.

Оба в `.gitignore` — это локальное runtime-состояние, не часть плагина.

## Инструменты для LLM

| Инструмент | Назначение |
|---|---|
| `memory_profile` | Чтение и обновление профильной карточки пользователя (имя, роль, стиль общения, устойчивые предпочтения). Маршрутизируется в Honcho. |
| `memory_reasoning` | Вопросы *о пользователе как личности* — стиль, привычки, поведенческие паттерны, что работает с ним лучше. Маршрутизируется в Honcho. |
| `memory_conclude` | Зафиксировать устойчивый вывод о пользователе (предпочтение, привычка, стиль). Маршрутизируется в Honcho. |
| `memory_recall` | **Главный инструмент долговременной памяти.** "Помнишь, как мы...", "мы это обсуждали", мульти-стратегический семантический + entity-graph поиск по всем прошлым сессиям. Маршрутизируется в Hindsight. |
| `memory_reflect` | LLM-синтез по фактам из прошлых разговоров — резюме, охватывающее несколько источников ("к чему мы пришли по X?"). Маршрутизируется в Hindsight. |
| `memory_forget` | Явное забывание через soft-delete сигнатуры. Реализован в композитном слое. |

## Hooks

Mnemosyne подключается к Hermes в шести точках жизненного цикла (объявлены в `plugin.yaml`):

- `on_memory_write` — перехват операций записи в память.
- `on_session_end` — финализация состояния в конце сессии.
- `on_session_switch` — обработка переключения между сессиями.
- `on_pre_compress` — дедупликация перед компрессией.
- `on_delegation` — события делегирования агенту.
- `on_turn_start` — инициализация на каждом ходу.

## Обновление

Когда на `main` появляются новые коммиты:

```bash
cd ~/.hermes/plugins/mnemosyne
git pull
./install.sh              # переустановит зависимости, если требования изменились
hermes gateway restart
```

Ваши runtime-данные (`fact_store.db`, `recovery_cursor.json`) лежат в `.gitignore` и переживут обновление.

## Вклад в разработку

Issues и pull requests приветствуются. Стандартный GitHub-flow:

1. **Issues** — открывайте issue с описанием проблемы или идеи фичи.
2. **Pull requests** — fork, branch, commit, push, открываете PR против `main`.

Перед PR:

- Проверьте, что изменение работает на **macOS** и **Linux**, если оно затрагивает `install.sh` или пути в файловой системе.
- Запустите `ruff check` для базовой проверки стиля.
- Держите коммиты сфокусированными — одна тема на коммит.

## Лицензия

[Apache-2.0](LICENSE)

## Roadmap

| Этап | Статус | Что |
|---|---|---|
| Scaffold (Stage A) | ✅ текущий | Fan-out делегация, объединение схем инструментов |
| 1, 2 | в планах | Дат-теги + счётчик повторений (SQLite `fact_store`) |
| 3 | в планах | Pre-write дедуп — короткий recall перед каждым `retain` |
| 5 | в планах | Anchor card — вручную закреплённые факты всегда в prefetch |
| 6 | в планах | Кураторские инструменты — шесть `memory_*` с tight role descriptions |
| 6′ | в планах | Conflict resolver — два голоса, когда Honcho и Hindsight расходятся |
| 7 | в планах | Prefetch fusion — anchor → Honcho peer card → Hindsight recall |
| 7′ | в планах | Recovery — повторная отправка транскриптов из `~/.hermes/sessions/` после сбоев |
| 8 | в планах | Bulk import — последние 90 дней session transcripts по требованию |
| 10 | в планах | Встроенный memory bridge — `on_memory_write` → Hindsight с `mention_count=10` |
| 11 | в планах | Забывание — `memory_forget` tool + CLI, soft-delete через `forgotten:` тег |
