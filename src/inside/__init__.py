"""
INSIDE: INternal States for hallucInation DEtection

This module implements hallucination detection using LLM internal states and
intent-aware retrieval for Self-RAG systems.

Key Components:
    - eigenscore: EigenScore computation for semantic consistency measurement
    - internal_states: Extract and process LLM internal embeddings
    - feature_clipping: Test-time activation truncation for overconfidence detection
    - intent_detector: Query intent classification for adaptive retrieval
    - hallucination_detector: Unified interface for hallucination detection

References:
    - INSIDE Paper: Internal States for Hallucination Detection
"""

from .eigenscore import EigenScore, compute_eigenscore
from .internal_states import InternalStatesExtractor
from .feature_clipping import FeatureClipper
from .intent_detector import IntentDetector, QueryIntent
from .hallucination_detector import HallucinationDetector

__all__ = [
    "EigenScore",
    "compute_eigenscore",
    "InternalStatesExtractor",
    "FeatureClipper",
    "IntentDetector",
    "QueryIntent",
    "HallucinationDetector",
]

__version__ = "0.1.0"
