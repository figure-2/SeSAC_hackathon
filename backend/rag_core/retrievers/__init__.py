"""
Retriever modules for the integrated History Docent RAG system.

This package exposes helper functions that return context documents from
individual vector stores (e.g., the original Gemini corpus and the HJ
YouTube corpus). Each retriever returns a list of dictionaries with two
keys:

    - ``content``: the chunk text
    - ``metadata``: metadata dictionary that must include ``source_db``
"""

from .gemini_retriever import get_gemini_contexts  # noqa: F401
from .hj_retriever import get_hj_contexts  # noqa: F401
