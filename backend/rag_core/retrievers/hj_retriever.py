"""Retrieval helper for the HJ YouTube script vector store."""

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
            "HJ_VECTOR_DB_DIR",
            "HJ_COLLECTION_NAME",
            "HJ_EMBEDDING_MODEL",
        ],
    )
    return config


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model for the HJ vector store."""
    config = _get_config()
    model_name = config["HJ_EMBEDDING_MODEL"]

    LOGGER.info("Loading HJ embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.api.models.Collection.Collection:
    config = _get_config()
    vector_db_dir = Path(config["HJ_VECTOR_DB_DIR"]).expanduser()
    collection_name = config["HJ_COLLECTION_NAME"]

    if not vector_db_dir.exists():
        LOGGER.error("HJ vector DB directory not found: %s", vector_db_dir)

    client = chromadb.PersistentClient(path=str(vector_db_dir))
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        LOGGER.warning(
            "Collection '%s' not found in HJ vector DB. Creating handle.",
            collection_name,
        )
        return client.get_or_create_collection(name=collection_name)


def get_hj_contexts(query: str, top_k: int | None = None) -> List[Dict]:
    """Return candidate contexts from the HJ YouTube vector DB."""
    config = _get_config()
    retrieve_k = int(
        top_k
        or config.get("HJ_RETRIEVE_K")
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
        metadata.setdefault("source_db", "hj")
        if idx < len(distances):
            metadata.setdefault("distance", float(distances[idx]))

        contexts.append(
            {
                "content": content,
                "metadata": metadata,
            }
        )

    return contexts
