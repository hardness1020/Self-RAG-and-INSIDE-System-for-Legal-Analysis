"""
Intent-Aware Retriever (INSIDE Extension)

Extends the base retriever with intent-aware strategies that adapt retrieval
based on query type (factual, exploratory, comparative, procedural).

Different intents require different approaches:
- Factual: High precision, specific documents
- Exploratory: Diverse results, broader coverage
- Comparative: Contrasting documents
- Procedural: Sequential, step-by-step documents
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import base retrieval components
from .retriever import LegalRetriever
from ..inside.intent_detector import IntentDetector, QueryIntent, get_retrieval_strategy


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with intent and strategy information."""
    documents: List[str]
    scores: List[float]
    metadata: List[Dict]
    query_intent: str
    strategy_used: str
    num_retrieved: int


class INSIDERetriever:
    """
    Intent-aware retriever that adapts strategies based on query intent.

    Extends LegalRetriever with:
    1. Automatic intent detection
    2. Intent-specific retrieval strategies
    3. Diversity-based reranking
    4. Adaptive top-k selection

    Args:
        base_retriever: Base LegalRetriever instance
        intent_detector: IntentDetector instance (optional, creates default if None)
        enable_diversity: Whether to use diversity-based reranking
        config: Additional configuration parameters
    """

    def __init__(
        self,
        base_retriever: LegalRetriever,
        intent_detector: Optional[IntentDetector] = None,
        enable_diversity: bool = True,
        config: Optional[Dict] = None
    ):
        self.base_retriever = base_retriever
        self.intent_detector = intent_detector or IntentDetector(method='rules')
        self.enable_diversity = enable_diversity
        self.config = config or {}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        intent: Optional[QueryIntent] = None,
        return_metadata: bool = True
    ) -> RetrievalResult:
        """
        Retrieve documents with intent-aware strategy.

        Args:
            query: Query string
            top_k: Number of documents to retrieve (overrides strategy default)
            intent: Optional pre-detected intent (auto-detects if None)
            return_metadata: Whether to return document metadata

        Returns:
            RetrievalResult with documents and strategy information
        """
        # Detect intent if not provided
        if intent is None:
            intent = self.intent_detector.detect_intent(query)

        # Get retrieval strategy for this intent
        strategy = get_retrieval_strategy(intent)

        # Use provided top_k or strategy default
        k = top_k if top_k is not None else strategy['top_k']

        # Base retrieval (get more than needed for reranking)
        candidate_k = int(k * 2) if self.enable_diversity else k
        base_results = self.base_retriever.retrieve(
            query=query,
            top_k=candidate_k
        )

        # Apply intent-specific processing
        if strategy['rerank_method'] == 'diversity' and self.enable_diversity:
            documents, scores, metadata = self._rerank_for_diversity(
                documents=base_results['documents'],
                scores=base_results['scores'],
                metadata=base_results.get('metadata', []),
                target_k=k,
                diversity_weight=strategy['diversity']
            )
        elif strategy['rerank_method'] == 'contrast':
            documents, scores, metadata = self._rerank_for_contrast(
                documents=base_results['documents'],
                scores=base_results['scores'],
                metadata=base_results.get('metadata', []),
                target_k=k
            )
        elif strategy['rerank_method'] == 'sequential':
            documents, scores, metadata = self._rerank_for_sequence(
                documents=base_results['documents'],
                scores=base_results['scores'],
                metadata=base_results.get('metadata', []),
                target_k=k
            )
        else:
            # Default: just take top-k by relevance
            documents = base_results['documents'][:k]
            scores = base_results['scores'][:k]
            metadata = base_results.get('metadata', [])[:k]

        return RetrievalResult(
            documents=documents,
            scores=scores,
            metadata=metadata if return_metadata else [],
            query_intent=intent.value,
            strategy_used=strategy['description'],
            num_retrieved=len(documents)
        )

    def _rerank_for_diversity(
        self,
        documents: List[str],
        scores: List[float],
        metadata: List[Dict],
        target_k: int,
        diversity_weight: float = 0.7
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Rerank using Maximal Marginal Relevance (MMR) for diversity.

        MMR = λ * relevance - (1-λ) * max_similarity_to_selected

        Args:
            documents: Candidate documents
            scores: Relevance scores
            metadata: Document metadata
            target_k: Target number of documents
            diversity_weight: Weight for diversity (λ)

        Returns:
            Tuple of (reranked_documents, reranked_scores, reranked_metadata)
        """
        if len(documents) <= target_k:
            return documents, scores, metadata

        # Get embeddings for documents
        doc_embeddings = self.base_retriever.embedding_model.encode(
            documents,
            convert_to_tensor=False
        )

        selected_indices = []
        remaining_indices = list(range(len(documents)))

        # Select first document (highest relevance)
        selected_indices.append(0)
        remaining_indices.remove(0)

        # Iteratively select diverse documents
        while len(selected_indices) < target_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance component
                relevance = scores[idx]

                # Diversity component (max similarity to already selected)
                max_similarity = max(
                    self._cosine_similarity(doc_embeddings[idx], doc_embeddings[sel_idx])
                    for sel_idx in selected_indices
                )

                # MMR score
                mmr = diversity_weight * relevance - (1 - diversity_weight) * max_similarity
                mmr_scores.append((idx, mmr))

            # Select document with highest MMR score
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return reranked results
        reranked_docs = [documents[i] for i in selected_indices]
        reranked_scores = [scores[i] for i in selected_indices]
        reranked_metadata = [metadata[i] for i in selected_indices] if metadata else []

        return reranked_docs, reranked_scores, reranked_metadata

    def _rerank_for_contrast(
        self,
        documents: List[str],
        scores: List[float],
        metadata: List[Dict],
        target_k: int
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Rerank to find contrasting documents for comparison queries.

        Selects documents that are relevant but present different perspectives.

        Args:
            documents: Candidate documents
            scores: Relevance scores
            metadata: Document metadata
            target_k: Target number of documents

        Returns:
            Tuple of (reranked_documents, reranked_scores, reranked_metadata)
        """
        # Use diversity with high weight to get contrasting views
        return self._rerank_for_diversity(
            documents=documents,
            scores=scores,
            metadata=metadata,
            target_k=target_k,
            diversity_weight=0.8  # Higher diversity for contrasts
        )

    def _rerank_for_sequence(
        self,
        documents: List[str],
        scores: List[float],
        metadata: List[Dict],
        target_k: int
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Rerank for procedural queries to create logical sequence.

        Attempts to order documents in a logical progression.

        Args:
            documents: Candidate documents
            scores: Relevance scores
            metadata: Document metadata
            target_k: Target number of documents

        Returns:
            Tuple of (reranked_documents, reranked_scores, reranked_metadata)
        """
        # For now, use relevance-based ordering
        # Could be enhanced with sequence detection (e.g., "first", "then", "finally")
        top_indices = np.argsort(scores)[::-1][:target_k]

        reranked_docs = [documents[i] for i in top_indices]
        reranked_scores = [scores[i] for i in top_indices]
        reranked_metadata = [metadata[i] for i in top_indices] if metadata else []

        return reranked_docs, reranked_scores, reranked_metadata

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of documents per query

        Returns:
            List of RetrievalResult objects
        """
        results = []
        for query in queries:
            result = self.retrieve(query=query, top_k=top_k)
            results.append(result)
        return results

    def get_statistics(self, results: List[RetrievalResult]) -> Dict[str, any]:
        """
        Compute statistics over retrieval results.

        Args:
            results: List of RetrievalResult objects

        Returns:
            Dictionary with statistics
        """
        intents = [r.query_intent for r in results]
        strategies = [r.strategy_used for r in results]
        num_retrieved = [r.num_retrieved for r in results]

        # Count intent distribution
        intent_counts = {intent: intents.count(intent) for intent in set(intents)}

        # Count strategy distribution
        strategy_counts = {strategy: strategies.count(strategy) for strategy in set(strategies)}

        return {
            'total_queries': len(results),
            'intent_distribution': intent_counts,
            'strategy_distribution': strategy_counts,
            'avg_documents_retrieved': np.mean(num_retrieved),
            'min_documents_retrieved': min(num_retrieved),
            'max_documents_retrieved': max(num_retrieved)
        }


def create_inside_retriever(
    index_path: str,
    embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    enable_diversity: bool = True,
    config: Optional[Dict] = None
) -> INSIDERetriever:
    """
    Factory function to create an INSIDERetriever.

    Args:
        index_path: Path to FAISS index
        embedding_model_name: Name of embedding model
        enable_diversity: Whether to enable diversity reranking
        config: Additional configuration

    Returns:
        Configured INSIDERetriever
    """
    # Create base retriever
    base_retriever = LegalRetriever(
        index_path=index_path,
        embedding_model_name=embedding_model_name
    )

    # Create INSIDE retriever
    return INSIDERetriever(
        base_retriever=base_retriever,
        enable_diversity=enable_diversity,
        config=config
    )


class HybridRetriever:
    """
    Hybrid retriever that combines base and INSIDE retrievers.

    Uses INSIDE for intent detection and strategy selection,
    then blends results from both retrievers.
    """

    def __init__(
        self,
        base_retriever: LegalRetriever,
        inside_retriever: INSIDERetriever,
        blend_weight: float = 0.5
    ):
        self.base_retriever = base_retriever
        self.inside_retriever = inside_retriever
        self.blend_weight = blend_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> RetrievalResult:
        """
        Retrieve using hybrid approach.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            RetrievalResult with blended results
        """
        # Get results from both retrievers
        base_result = self.base_retriever.retrieve(query=query, top_k=top_k * 2)
        inside_result = self.inside_retriever.retrieve(query=query, top_k=top_k * 2)

        # Blend scores
        # TODO: Implement sophisticated score blending and deduplication

        # For now, return INSIDE result (which uses base retriever internally)
        return inside_result
