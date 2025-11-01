"""Data loading utilities for legal datasets."""

from .legalbench_loader import (
    LegalBenchRAGLoader,
    LegalBenchQuery,
    LegalBenchSnippet,
    load_legalbench_rag
)

__all__ = [
    'LegalBenchRAGLoader',
    'LegalBenchQuery',
    'LegalBenchSnippet',
    'load_legalbench_rag',
]
