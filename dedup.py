"""Recall-time fuzzy dedup.

Hindsight returns top-N semantically-close-to-query candidates, not
top-N pairwise-distinct candidates. With 40+ paraphrases of one fact in
the bank ("Hermes tested memory_forget on Barsik" / "Assistant deleted
Barsik info" / …) all of them survive the existing exact-canonical-key
filter and bloat the agent's context window.

This module groups recall results by **content similarity between
candidates** so we can keep one representative per cluster.

Two-stage hybrid:

1.  **Jaccard over content tokens** — fast, catches paraphrases that
    share entities.
2.  **Embedding cosine** — confirms the merge for pairs that pass
    Jaccard. Critically, this *prevents* false merges: "Barsik got sick"
    and "Barsik died" have high Jaccard but low cosine, so they stay
    separate. This is the layer that protects fact integrity.

Both thresholds (and the entire feature) are configurable via
``prefetch.*`` config keys / ``MNEMOSYNE_DEDUP_*`` env vars.

Fail-open everywhere: any error in embedding fetch silently degrades to
Jaccard-only, then to no-op (return original list).
"""

from __future__ import annotations

import logging
import math
import os
import re
import urllib.error
import urllib.request
import json as _json
from typing import Iterable, List, Optional, Sequence

from . import config

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_STOP = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "from", "by", "for", "with", "and",
    "or", "but", "this", "that", "these", "those", "it", "its", "as",
    "has", "have", "had", "do", "does", "did", "not", "no",
    # Russian
    "и", "в", "на", "не", "что", "это", "как", "по", "из", "к", "у", "о",
    "от", "за", "со", "до", "для", "при", "без", "то", "же", "о", "об",
    "а", "но", "или", "если", "так", "уже", "был", "была", "было", "были",
    "есть", "его", "её", "их", "там", "тут", "ещё", "ли", "бы",
    # Common noise in our recall lines
    "user", "assistant", "пользователь", "ассистент",
}


def _content_tokens(text: str) -> set:
    return {
        t.lower() for t in _WORD_RE.findall(text or "")
        if len(t) > 1 and t.lower() not in _STOP
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _fetch_embeddings(texts: List[str], timeout: float) -> Optional[List[List[float]]]:
    """Call the OpenAI-compatible embeddings endpoint that Hindsight is
    already wired to (litellm @ localhost:8000 with Jina v5 by default,
    see __init__.py:_inject_hindsight_routing_env).

    Returns a list of vectors aligned with `texts`, or None on any error.
    """
    base = os.environ.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL",
                          "http://localhost:8000/v1")
    key = os.environ.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
                         "sk-local-litellm")
    model = os.environ.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL",
                           "jina-embeddings-v5-text-small-retrieval-mlx")
    url = base.rstrip("/") + "/embeddings"

    body = _json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.debug("mnemosyne.dedup: embeddings fetch failed: %s", exc)
        return None
    except Exception as exc:  # malformed json, etc.
        logger.debug("mnemosyne.dedup: embeddings parse failed: %s", exc)
        return None

    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != len(texts):
        return None
    out: List[List[float]] = []
    for item in items:
        emb = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(emb, list):
            return None
        out.append([float(x) for x in emb])
    return out


def cluster_lines(lines: Iterable[str]) -> List[str]:
    """Greedy-cluster `lines` by hybrid similarity, return one
    representative (the longest line) per cluster, preserving original
    order of first occurrence.

    No-op if dedup is disabled in config or input has ≤1 unique line.
    """
    items = [l for l in (line.strip() for line in lines) if l]
    if not items:
        return []

    if not bool(config.get("prefetch", "dedup_enabled", default=True)):
        # Even with feature off, drop exact duplicates by lowercase eq
        # — same behaviour as before.
        seen: set = set()
        out: List[str] = []
        for it in items:
            k = it.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    jac_min = float(config.get("prefetch", "dedup_jaccard_min", default=0.5))
    cos_min = float(config.get("prefetch", "dedup_cosine_min", default=0.88))
    use_emb = bool(config.get("prefetch", "dedup_use_embeddings", default=True))
    emb_timeout = float(config.get("prefetch", "dedup_embedding_timeout", default=4.0))

    # Pre-compute token sets and length once.
    tokens = [_content_tokens(t) for t in items]
    lengths = [len(t) for t in items]

    # Gate embeddings: only fetch if any pair could plausibly need them
    # (i.e., any Jaccard ≥ jac_min). Saves a network call on small inputs
    # or when nothing looks similar.
    embeddings: Optional[List[List[float]]] = None
    if use_emb and len(items) >= 2:
        any_pair = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _jaccard(tokens[i], tokens[j]) >= jac_min:
                    any_pair = True
                    break
            if any_pair:
                break
        if any_pair:
            embeddings = _fetch_embeddings(items, timeout=emb_timeout)

    # Greedy clustering: each item joins the first existing cluster
    # whose representative passes BOTH thresholds.
    cluster_reps: List[int] = []  # indices into `items`
    assigned: List[int] = [-1] * len(items)

    for i, _ in enumerate(items):
        placed = False
        for ci, rep_idx in enumerate(cluster_reps):
            jac = _jaccard(tokens[i], tokens[rep_idx])
            if jac < jac_min:
                continue
            if embeddings is not None:
                cos = _cosine(embeddings[i], embeddings[rep_idx])
                if cos < cos_min:
                    continue  # token-similar but semantically different — keep apart
            assigned[i] = ci
            placed = True
            # Promote longer text as cluster representative — usually the
            # original/canonical phrasing rather than a short paraphrase.
            if lengths[i] > lengths[rep_idx]:
                cluster_reps[ci] = i
            break
        if not placed:
            assigned[i] = len(cluster_reps)
            cluster_reps.append(i)

    # Emit one line per cluster, in order of first occurrence.
    return [items[idx] for idx in cluster_reps]
