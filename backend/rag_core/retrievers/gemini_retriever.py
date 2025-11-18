"""Retrieval helpers for the original (Gemini) history vector store."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import chromadb
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..utils.config import ensure_keys, load_config

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_config() -> Dict[str, str]:
    config = load_config()
    ensure_keys(
        config,
        [
            "VECTOR_DB_DIR",
            "FINETUNED_COLLECTION_NAME",
            "FINETUNED_EMBEDDING_MODEL",
            "BASELINE_EMBEDDING_MODEL",
        ],
    )
    return config


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model used for the Gemini vector store."""
    config = _get_config()
    rag_core_root = Path(__file__).resolve().parents[1]
    candidate_model = rag_core_root / config["FINETUNED_EMBEDDING_MODEL"]

    if candidate_model.exists():
        model_name = str(candidate_model)
        LOGGER.info("Using finetuned embedding model at %s", model_name)
    else:
        model_name = config["BASELINE_EMBEDDING_MODEL"]
        LOGGER.warning(
            "Finetuned embedding model not found at %s, falling back to %s",
            candidate_model,
            model_name,
        )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.api.models.Collection.Collection:
    """Return a ChromaDB collection handle for the Gemini corpus."""
    config = _get_config()
    rag_core_root = Path(__file__).resolve().parents[1]
    db_root = rag_core_root / config["VECTOR_DB_DIR"]
    collection_name = config["FINETUNED_COLLECTION_NAME"] or config[
        "BASELINE_COLLECTION_NAME"
    ]

    preferred_path = db_root / collection_name
    client = None
    for candidate in (preferred_path, db_root):
        sqlite_file = candidate / "chroma.sqlite3"
        if sqlite_file.exists():
            client = chromadb.PersistentClient(path=str(candidate))
            LOGGER.info("Connecting to Gemini Chroma collection at %s", candidate)
            break

    if client is None:
        LOGGER.warning(
            "Could not find chroma.sqlite3 under %s – attempting direct connection.",
            db_root,
        )
        client = chromadb.PersistentClient(path=str(db_root))

    try:
        return client.get_collection(name=collection_name)
    except Exception:
        LOGGER.warning(
            "Collection '%s' not found, creating a new handle (metadata only).",
            collection_name,
        )
        return client.get_or_create_collection(name=collection_name)


def get_gemini_contexts(query: str, top_k: int | None = None) -> List[Dict]:
    """Return candidate contexts from the Gemini vector DB.

    Args:
        query: User query string.
        top_k: Optional override for number of chunks to fetch.

    Returns:
        A list of dictionaries with ``content`` and ``metadata`` keys.
    """
    config = _get_config()
    retrieve_k = int(
        top_k
        or config.get("GEMINI_RETRIEVE_K")
        or config.get("RETRIEVE_K")
        or 20
    )

    embeddings = _get_embeddings()
    collection = _get_collection()

    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieve_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    contexts: List[Dict] = []
    for idx, content in enumerate(documents):
        metadata = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
        metadata.setdefault("source_db", "gemini")
        if idx < len(distances):
            metadata.setdefault("distance", float(distances[idx]))

        contexts.append(
            {
                "content": content,
                "metadata": metadata,
            }
        )

    return contexts
