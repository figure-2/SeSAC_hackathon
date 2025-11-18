"""Integrated RAG pipeline that combines Gemini & HJ retrievers."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from sentence_transformers import CrossEncoder

from ..retrievers import get_gemini_contexts, get_hj_contexts
from ..utils.config import ensure_keys, load_config

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_config() -> Dict[str, str]:
    config = load_config()
    ensure_keys(
        config,
        [
            "RETRIEVE_K",
            "RERANK_K",
            "INTEGRATED_RERANK_TOPK",
            "FINETUNED_RERANKER_MODEL",
            "BASELINE_RERANKER_MODEL",
        ],
    )
    return config


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load (or cache) the cross-encoder reranker."""
    config = _get_config()
    rag_core_root = Path(__file__).resolve().parents[1]
    candidate_model = rag_core_root / config["FINETUNED_RERANKER_MODEL"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if candidate_model.exists():
        model_name = str(candidate_model)
        LOGGER.info("Using finetuned reranker from %s", model_name)
    else:
        model_name = config["BASELINE_RERANKER_MODEL"]
        LOGGER.warning(
            "Finetuned reranker not found at %s. Falling back to %s",
            candidate_model,
            model_name,
        )

    return CrossEncoder(model_name, max_length=512, device=device)


def _call_retrievers(
    query: str,
    gemini_k: int,
    hj_k: int,
) -> List[Dict]:
    """Fetch candidate contexts from both retrievers in parallel."""
    contexts: List[Dict] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(get_gemini_contexts, query, gemini_k): "gemini",
            executor.submit(get_hj_contexts, query, hj_k): "hj",
        }

        for future, label in futures.items():
            try:
                results = future.result()
                LOGGER.info("%s retriever returned %d candidates", label, len(results))
                contexts.extend(results)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("%s retriever failed: %s", label, exc)

    return contexts


def rerank_contexts(
    query: str,
    contexts: Iterable[Dict],
    top_k: int,
) -> List[Dict]:
    """Score and sort contexts using the shared reranker."""
    context_list = list(contexts)
    if not context_list:
        return []

    reranker = _get_reranker()
    pairs: List[Tuple[str, str]] = [(query, ctx["content"]) for ctx in context_list]
    scores = reranker.predict(pairs)

    enriched: List[Dict] = []
    for ctx, score in zip(context_list, scores):
        metadata = dict(ctx.get("metadata") or {})
        metadata["rerank_score"] = float(score)
        enriched.append({"content": ctx["content"], "metadata": metadata})

    enriched.sort(key=lambda item: item["metadata"]["rerank_score"], reverse=True)
    return enriched[:top_k]


def run_integrated_pipeline(
    query: str,
    *,
    gemini_k: int | None = None,
    hj_k: int | None = None,
    final_k: int | None = None,
) -> List[Dict]:
    """High-level helper that runs retrieval + shared reranking."""
    config = _get_config()
    gemini_top_k = int(
        gemini_k
        or config.get("GEMINI_RETRIEVE_K")
        or config.get("RETRIEVE_K")
        or 20
    )
    hj_top_k = int(
        hj_k
        or config.get("HJ_RETRIEVE_K")
        or config.get("RETRIEVE_K")
        or 20
    )
    final_top_k = int(final_k or config.get("INTEGRATED_RERANK_TOPK") or 5)

    raw_contexts = _call_retrievers(query, gemini_top_k, hj_top_k)
    LOGGER.info("Total retrieved contexts before rerank: %d", len(raw_contexts))

    reranked = rerank_contexts(query, raw_contexts, final_top_k)
    LOGGER.info("Returning top %d contexts after rerank", len(reranked))
    return reranked


def build_context_block(contexts: Iterable[Dict]) -> str:
    """Render contexts into a prompt-friendly string."""
    lines: List[str] = []
    for idx, ctx in enumerate(contexts, 1):
        metadata = ctx.get("metadata") or {}
        source_db = metadata.get("source_db", "unknown")
        chunk_id = metadata.get("chunk_id", f"chunk_{idx}")
        score = metadata.get("rerank_score")

        header = f"[{idx}] ({source_db}) {chunk_id}"
        if score is not None:
            header += f" | score={score:.4f}"
        lines.append(header)
        lines.append(ctx["content"])
        lines.append("")  # blank line between entries

    return "\n".join(lines).strip()
