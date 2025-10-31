"""
EigenScore: Semantic Consistency Measurement

Implements the EigenScore metric from the INSIDE paper, which uses eigenvalues
of sentence embeddings' covariance matrix to measure semantic consistency.

The key insight: hallucinated content tends to show lower semantic diversity,
resulting in a different eigenvalue distribution (differential entropy).
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats
import math


class EigenScore:
    """
    EigenScore metric for hallucination detection.

    Computes differential entropy of eigenvalues from the covariance matrix
    of sentence embeddings. Lower EigenScore suggests potential hallucination.

    Args:
        normalize: Whether to normalize embeddings before computing covariance
        top_k_eigenvalues: Number of top eigenvalues to use (None = all)
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        normalize: bool = True,
        top_k_eigenvalues: Optional[int] = None,
        epsilon: float = 1e-10
    ):
        self.normalize = normalize
        self.top_k_eigenvalues = top_k_eigenvalues
        self.epsilon = epsilon

    def compute(
        self,
        embeddings: torch.Tensor,
        return_details: bool = False
    ) -> float:
        """
        Compute EigenScore from sentence embeddings.

        Args:
            embeddings: Sentence embeddings tensor (num_sentences, hidden_dim)
            return_details: Whether to return detailed information

        Returns:
            EigenScore (differential entropy) or dict with details if return_details=True
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        if embeddings.shape[0] < 2:
            # Need at least 2 sentences for covariance
            if return_details:
                return {"score": 0.0, "num_sentences": embeddings.shape[0], "eigenvalues": []}
            return 0.0

        # Normalize embeddings if requested
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute covariance matrix
        embeddings_np = embeddings.cpu().numpy()
        cov_matrix = np.cov(embeddings_np, rowvar=False)  # Shape: (hidden_dim, hidden_dim)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix)

        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Filter out negative eigenvalues (numerical errors)
        eigenvalues = eigenvalues[eigenvalues > self.epsilon]

        if len(eigenvalues) == 0:
            if return_details:
                return {"score": 0.0, "num_sentences": embeddings.shape[0], "eigenvalues": []}
            return 0.0

        # Use top-k eigenvalues if specified
        if self.top_k_eigenvalues is not None:
            eigenvalues = eigenvalues[:self.top_k_eigenvalues]

        # Compute differential entropy
        # H = 0.5 * log((2πe)^n * |Σ|) where |Σ| is product of eigenvalues
        n = len(eigenvalues)
        log_det = np.sum(np.log(eigenvalues + self.epsilon))
        differential_entropy = 0.5 * (n * math.log(2 * math.pi * math.e) + log_det)

        if return_details:
            return {
                "score": float(differential_entropy),
                "num_sentences": embeddings.shape[0],
                "eigenvalues": eigenvalues.tolist(),
                "top_eigenvalue": float(eigenvalues[0]),
                "log_det": float(log_det),
                "condition_number": float(eigenvalues[0] / eigenvalues[-1])
            }

        return float(differential_entropy)

    def compute_batch(
        self,
        embeddings_list: List[torch.Tensor]
    ) -> List[float]:
        """
        Compute EigenScore for a batch of embedding sets.

        Args:
            embeddings_list: List of embedding tensors

        Returns:
            List of EigenScores
        """
        scores = []
        for embeddings in embeddings_list:
            score = self.compute(embeddings)
            scores.append(score)
        return scores


def compute_eigenscore(
    embeddings: torch.Tensor,
    normalize: bool = True,
    top_k: Optional[int] = None
) -> float:
    """
    Convenience function to compute EigenScore.

    Args:
        embeddings: Sentence embeddings (num_sentences, hidden_dim)
        normalize: Whether to normalize embeddings
        top_k: Number of top eigenvalues to use

    Returns:
        EigenScore value
    """
    scorer = EigenScore(normalize=normalize, top_k_eigenvalues=top_k)
    return scorer.compute(embeddings)


class EigenScoreAggregator:
    """
    Aggregate EigenScores from multiple generations for robust detection.

    The INSIDE paper shows that using multiple generations improves detection.
    """

    def __init__(
        self,
        aggregation_method: str = 'mean',
        threshold: Optional[float] = None
    ):
        """
        Args:
            aggregation_method: Method to aggregate scores ('mean', 'min', 'median')
            threshold: Threshold for hallucination detection (lower = hallucination)
        """
        self.aggregation_method = aggregation_method
        self.threshold = threshold

    def aggregate(self, scores: List[float]) -> float:
        """Aggregate multiple EigenScores."""
        if not scores:
            return 0.0

        if self.aggregation_method == 'mean':
            return np.mean(scores)
        elif self.aggregation_method == 'min':
            return np.min(scores)
        elif self.aggregation_method == 'median':
            return np.median(scores)
        elif self.aggregation_method == 'max':
            return np.max(scores)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def detect_hallucination(self, scores: List[float]) -> Tuple[bool, float]:
        """
        Detect hallucination based on aggregated score.

        Args:
            scores: List of EigenScores

        Returns:
            Tuple of (is_hallucination, aggregated_score)
        """
        agg_score = self.aggregate(scores)

        if self.threshold is not None:
            is_hallucination = agg_score < self.threshold
        else:
            is_hallucination = False

        return is_hallucination, agg_score


class AdaptiveEigenScore:
    """
    Adaptive EigenScore that adjusts threshold based on query characteristics.

    Different query types may have different baseline EigenScore distributions.
    """

    def __init__(
        self,
        base_threshold: float = 5.0,
        query_length_adjustment: bool = True,
        confidence_adjustment: bool = True
    ):
        self.base_threshold = base_threshold
        self.query_length_adjustment = query_length_adjustment
        self.confidence_adjustment = confidence_adjustment

    def compute_adaptive_threshold(
        self,
        query_length: int,
        model_confidence: Optional[float] = None
    ) -> float:
        """
        Compute adaptive threshold based on query characteristics.

        Args:
            query_length: Length of query in tokens
            model_confidence: Model confidence score (e.g., mean token probability)

        Returns:
            Adjusted threshold
        """
        threshold = self.base_threshold

        # Adjust for query length (longer queries may have higher baseline entropy)
        if self.query_length_adjustment:
            length_factor = min(1.5, 1.0 + (query_length - 20) / 100)
            threshold *= length_factor

        # Adjust for model confidence (higher confidence → stricter threshold)
        if self.confidence_adjustment and model_confidence is not None:
            confidence_factor = 0.8 + 0.4 * model_confidence  # Range: [0.8, 1.2]
            threshold *= confidence_factor

        return threshold

    def detect_hallucination_adaptive(
        self,
        eigenscore: float,
        query_length: int,
        model_confidence: Optional[float] = None
    ) -> Tuple[bool, float, float]:
        """
        Detect hallucination with adaptive threshold.

        Returns:
            Tuple of (is_hallucination, eigenscore, adaptive_threshold)
        """
        adaptive_threshold = self.compute_adaptive_threshold(query_length, model_confidence)
        is_hallucination = eigenscore < adaptive_threshold

        return is_hallucination, eigenscore, adaptive_threshold


def compute_pairwise_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between sentence embeddings.

    High average similarity might indicate lack of diversity (potential hallucination).

    Args:
        embeddings: Sentence embeddings (num_sentences, hidden_dim)

    Returns:
        Pairwise similarity matrix (num_sentences, num_sentences)
    """
    # Normalize embeddings
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity
    similarity = torch.mm(embeddings_norm, embeddings_norm.t())

    return similarity


def analyze_embedding_distribution(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Analyze the distribution properties of embeddings.

    Args:
        embeddings: Sentence embeddings (num_sentences, hidden_dim)

    Returns:
        Dictionary with distribution statistics
    """
    embeddings_np = embeddings.cpu().numpy()

    # Compute statistics
    mean_norm = np.linalg.norm(embeddings_np, axis=1).mean()
    std_norm = np.linalg.norm(embeddings_np, axis=1).std()

    # Pairwise similarity
    similarity = compute_pairwise_similarity(embeddings)
    # Exclude diagonal
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool)
    pairwise_sim = similarity[mask]

    return {
        "mean_norm": float(mean_norm),
        "std_norm": float(std_norm),
        "mean_pairwise_similarity": float(pairwise_sim.mean()),
        "std_pairwise_similarity": float(pairwise_sim.std()),
        "min_pairwise_similarity": float(pairwise_sim.min()),
        "max_pairwise_similarity": float(pairwise_sim.max())
    }


def calibrate_threshold(
    hallucinated_embeddings: List[torch.Tensor],
    factual_embeddings: List[torch.Tensor],
    percentile: float = 10.0
) -> float:
    """
    Calibrate EigenScore threshold using labeled examples.

    Args:
        hallucinated_embeddings: List of embeddings from hallucinated responses
        factual_embeddings: List of embeddings from factual responses
        percentile: Percentile of factual scores to use as threshold

    Returns:
        Calibrated threshold
    """
    scorer = EigenScore()

    # Compute scores for hallucinated examples
    hallucinated_scores = [
        scorer.compute(emb) for emb in hallucinated_embeddings if emb.shape[0] >= 2
    ]

    # Compute scores for factual examples
    factual_scores = [
        scorer.compute(emb) for emb in factual_embeddings if emb.shape[0] >= 2
    ]

    # Use percentile of factual scores as threshold
    threshold = np.percentile(factual_scores, percentile)

    # Compute detection metrics
    if hallucinated_scores and factual_scores:
        tp = sum(1 for s in hallucinated_scores if s < threshold)
        fp = sum(1 for s in factual_scores if s < threshold)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(hallucinated_scores) if hallucinated_scores else 0

        print(f"Threshold: {threshold:.2f}")
        print(f"Precision: {precision:.2%}, Recall: {recall:.2%}")

    return threshold
