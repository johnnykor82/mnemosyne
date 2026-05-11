# Mnemosyne — Long-Term Memory Plugin for Hermes Agent

> 🇷🇺 Читать на русском: [README.ru.md](README.ru.md)

**Mnemosyne** (Μνημοσύνη — the Greek Titaness of memory, mother of the nine Muses) is a composite long-term memory plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It combines two existing memory providers — **[Honcho](https://github.com/plastic-labs/honcho)** (user model: style, preferences, conversational personality) and **[Hindsight](https://hindsight.app)** (fact store: entity graph, multi-strategy semantic recall) — behind a single `MemoryProvider` interface, and adds date tagging, repetition counting, conflict detection, crash recovery, and explicit forgetting on top.

Pairs naturally with its mythological daughter [Hermes-Mneme](https://github.com/johnnykor82/hermes-mneme) — a separate plugin that handles *short-term, in-session* context engineering.

## Why a composite memory provider

Hermes Agent permits only one external memory provider at a time. That's a problem, because the two best providers in the ecosystem are complementary, not interchangeable:

- **Honcho** is excellent at modelling *who the user is* — dialectic reasoning, peer cards, communication style, stable preferences. It's weaker at precise factual recall: no decay, no deduplication, LLM-generated representations.
- **Hindsight** is the inverse — strong factual store, entity graph, hybrid semantic + keyword recall, but no model of the user as a person.

Mnemosyne wraps both behind a single `MemoryProvider` and routes every call to whichever inner provider is right for the job:

- **Writes** → both providers (Honcho keeps its user-model fed; Hindsight builds the fact graph).
- **Auto-inject (prefetch)** → Hindsight facts + Honcho peer card; Honcho's noisy session summary / user representation is suppressed (we recommend configuring Honcho to `recallMode: tools`).
- **Tools** → six curated `memory_*` tools with tight role-based descriptions that route to the right inner provider under the hood.

The composite layer also adds:

1. **Date tags** on every retained fact (`fact_store.db`) — so "what did we discuss last week?" actually works.
2. **Repetition counter** tracking how often a fact comes up — important facts surface higher in recall.
3. **Explicit forgetting** — `memory_forget` tool + CLI, soft-delete via signature with semantic deduplication so the same fact doesn't sneak back in.

## Status

**Scaffold (Stage A).** Currently pure fan-out delegation to the two inner providers. The composite layer (date tags, repetition counter, conflict resolver, forgetting, anchor card, prefetch fusion) is on the roadmap below.

## Requirements

- **Python 3.11+**
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** installed and working (the plugin assumes a Hermes venv at `~/.hermes/hermes-agent/venv`, configurable via `HERMES_VENV`).
- **[Honcho](https://honcho.dev)** — Python client (`honcho-ai`). Already in place if you've been using Honcho as your memory provider.
- **[Hindsight](https://hindsight.app)** — install via `hermes memory setup` and pick `hindsight`.
- **macOS or Linux.** The plugin is platform-agnostic Python; the installer covers both.

## Installation

```bash
git clone https://github.com/johnnykor82/mnemosyne.git \
  ~/.hermes/plugins/mnemosyne
cd ~/.hermes/plugins/mnemosyne
./install.sh
```

The installer detects your Hermes venv (default `~/.hermes/hermes-agent/venv`, override with `HERMES_VENV=...`), installs the two Python dependencies (`honcho-ai`, `hindsight-client>=0.4.22`), and verifies they import cleanly.

The plugin lives in `$HERMES_HOME/plugins/mnemosyne/` (default `~/.hermes/plugins/mnemosyne/`) and survives `git pull` of `hermes-agent` itself — it's a user-installed addon, not a core component.

## Activation

```bash
hermes config set memory.provider mnemosyne
hermes gateway restart
```

Verify it loaded:

```bash
tail -f ~/.hermes/logs/agent.log | grep -i mnemosyne
```

To roll back to a previous provider:

```bash
hermes config set memory.provider honcho   # or whatever you were using
hermes gateway restart
```

## Configuration

Mnemosyne reads its configuration from `mnemosyne/config.json` (created at plugin directory on first run). Most defaults work out of the box — the plugin is designed so that a fresh install "Just Works" against a local LiteLLM proxy at `http://localhost:8000`.

Key things you can override (via `config.json` `hindsight_env` block, or via shell env vars — shell wins over config):

| Variable | Default | Purpose |
|---|---|---|
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | `openai` | Embedding provider name passed to Hindsight |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL` | `http://localhost:8000/v1` | Where the embeddings endpoint lives (local LiteLLM by default) |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY` | `sk-local-litellm` | Placeholder for the local proxy. Replace if pointing to a real API. |
| `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` | `jina-embeddings-v5-text-small-retrieval-mlx` | Default embedding model (Jina MLX local) |
| `HINDSIGHT_API_RERANKER_PROVIDER` | `cohere` | Reranker provider name |
| `HINDSIGHT_API_RERANKER_COHERE_BASE_URL` | `http://localhost:4000/v1/rerank` | Reranker endpoint |
| `HINDSIGHT_API_RERANKER_COHERE_API_KEY` | `sk-local-litellm` | Placeholder for the local proxy |
| `HINDSIGHT_API_RERANKER_COHERE_MODEL` | `rerank` | Default reranker model |

Plugin-internal storage:

- `fact_store.db` — SQLite store for date tags, repetition counters, forgotten signatures. Created on first run.
- `recovery_cursor.json` — offset into `~/.hermes/sessions/` for crash-recovery imports.

Both are gitignored — they are local runtime state, not part of the plugin.

## Tools exposed to the LLM

| Tool | Purpose |
|---|---|
| `memory_profile` | Read or update the user's profile card (name, role, communication style, stable preferences). Routes to Honcho. |
| `memory_reasoning` | Questions *about the user as a person* — style, habits, behavioural patterns, what works best with them. Routes to Honcho. |
| `memory_conclude` | Record a stable user-related conclusion (preference, habit, style). Routes to Honcho. |
| `memory_recall` | **The main long-term memory tool.** "Do you remember when…", "we discussed this", multi-strategy semantic + entity-graph search over all past conversations. Routes to Hindsight. |
| `memory_reflect` | LLM synthesis across past-conversation facts — summaries spanning multiple sources ("what did we conclude about X?"). Routes to Hindsight. |
| `memory_forget` | Explicit forgetting via signature soft-delete. Implemented inside the composite layer. |

## Hooks

Mnemosyne hooks into Hermes at six lifecycle points (declared in `plugin.yaml`):

- `on_memory_write` — intercept memory store operations.
- `on_session_end` — finalize state at session end.
- `on_session_switch` — handle multi-session context.
- `on_pre_compress` — pre-compression deduplication.
- `on_delegation` — agent delegation events.
- `on_turn_start` — per-turn initialization.

## Updating

When new commits land on `main`:

```bash
cd ~/.hermes/plugins/mnemosyne
git pull
./install.sh              # reinstalls deps if requirements changed
hermes gateway restart
```

Your runtime data (`fact_store.db`, `recovery_cursor.json`) is gitignored and survives updates.

## Contributing

Contributions and bug reports are very welcome. Standard GitHub flow:

1. **Issues** — open an issue describing the problem or feature idea.
2. **Pull requests** — fork, branch, commit, push, open a PR against `main`.

Before submitting a PR:

- Verify your change works on both **macOS** and **Linux** if it touches `install.sh` or filesystem paths.
- Run `ruff check` to catch obvious style issues.
- Keep commits focused — one concern per commit.

## License

[Apache-2.0](LICENSE)

## Roadmap

| Stage | Status | What |
|---|---|---|
| Scaffold (Stage A) | ✅ current | Fan-out delegation, union of inner tool schemas |
| 1, 2 | planned | Date tags + repetition counter (SQLite `fact_store`) |
| 3 | planned | Pre-write dedup — short recall before each `retain` |
| 5 | planned | Anchor card — manually-curated pinned facts always in prefetch |
| 6 | planned | Curated tools — six renamed `memory_*` with tight role descriptions |
| 6′ | planned | Conflict resolver — two-voice display when Honcho and Hindsight disagree |
| 7 | planned | Prefetch fusion — anchor → Honcho peer card → Hindsight recall |
| 7′ | planned | Recovery — re-send transcripts from `~/.hermes/sessions/` after crashes |
| 8 | planned | Bulk import — last 90 days of session transcripts on demand |
| 10 | planned | Built-in memory bridge — `on_memory_write` → Hindsight with `mention_count=10` |
| 11 | planned | Forgetting — `memory_forget` tool + CLI, soft-delete via `forgotten:` tag |
