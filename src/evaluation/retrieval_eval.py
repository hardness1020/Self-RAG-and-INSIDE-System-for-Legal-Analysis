"""
Retrieval Evaluation Script

Evaluates the retrieval pipeline using standard IR metrics:
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)

Usage:
    uv run python -m src.evaluation.retrieval_eval --config configs/retrieval_config.yaml --test-data data/test_queries.json
"""

import os
import sys
from pathlib import Path
import argparse
import json
import yaml
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retriever import LegalRetriever, load_retriever_from_config


def load_test_data(test_file: str) -> List[Dict[str, Any]]:
    """
    Load test queries with ground truth relevant documents.

    Expected format:
    [
        {
            "query": "What are the elements of negligence?",
            "relevant_doc_ids": [1, 5, 12],  # IDs of relevant documents
            # OR
            "relevant_texts": ["text snippet 1", "text snippet 2"]
        },
        ...
    ]

    Args:
        test_file: Path to test data JSON file

    Returns:
        List of test queries
    """
    with open(test_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} test queries")
    return data


def calculate_precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    Calculate Precision@k.

    Precision@k = (# relevant docs in top k) / k

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: List of ground truth relevant document IDs
        k: Cutoff rank

    Returns:
        Precision@k score
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    num_relevant = len(retrieved_at_k & relevant_set)
    return num_relevant / k


def calculate_recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    Calculate Recall@k.

    Recall@k = (# relevant docs in top k) / (total # relevant docs)

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: List of ground truth relevant document IDs
        k: Cutoff rank

    Returns:
        Recall@k score
    """
    if len(relevant_ids) == 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    num_relevant = len(retrieved_at_k & relevant_set)
    return num_relevant / len(relevant_ids)


def calculate_reciprocal_rank(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    """
    Calculate Reciprocal Rank (RR).

    RR = 1 / rank of first relevant document

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: List of ground truth relevant document IDs

    Returns:
        Reciprocal rank score
    """
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def calculate_average_precision(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    """
    Calculate Average Precision (AP).

    AP = sum(Precision@k * is_relevant(k)) / num_relevant

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_ids: List of ground truth relevant document IDs

    Returns:
        Average precision score
    """
    if len(relevant_ids) == 0:
        return 0.0

    relevant_set = set(relevant_ids)
    num_relevant = 0
    sum_precision = 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            num_relevant += 1
            precision_at_k = num_relevant / rank
            sum_precision += precision_at_k

    if num_relevant == 0:
        return 0.0

    return sum_precision / len(relevant_ids)


def evaluate_retrieval(
    retriever: LegalRetriever,
    test_queries: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, Any]:
    """
    Evaluate retrieval system on test queries.

    Args:
        retriever: LegalRetriever instance
        test_queries: List of test queries with ground truth
        k_values: List of k values for Precision@k and Recall@k

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating retrieval on {len(test_queries)} queries...")

    # Initialize metrics
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    reciprocal_ranks = []
    average_precisions = []

    # Evaluate each query
    for query_data in tqdm(test_queries, desc="Evaluating queries"):
        query = query_data['query']
        relevant_ids = query_data.get('relevant_doc_ids', [])

        # Skip queries without ground truth
        if not relevant_ids:
            continue

        # Retrieve documents
        max_k = max(k_values)
        results = retriever.retrieve(query, top_k=max_k * 2)  # Retrieve more to ensure coverage

        # Extract retrieved IDs
        retrieved_ids = [doc.get('doc_id', doc.get('id', idx)) for idx, doc in enumerate(results)]

        # Calculate metrics
        for k in k_values:
            precision = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
            recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)

        # Reciprocal rank
        rr = calculate_reciprocal_rank(retrieved_ids, relevant_ids)
        reciprocal_ranks.append(rr)

        # Average precision
        ap = calculate_average_precision(retrieved_ids, relevant_ids)
        average_precisions.append(ap)

    # Aggregate metrics
    results = {
        'num_queries': len(test_queries),
        'precision@k': {k: np.mean(precision_at_k[k]) for k in k_values},
        'recall@k': {k: np.mean(recall_at_k[k]) for k in k_values},
        'mrr': np.mean(reciprocal_ranks),
        'map': np.mean(average_precisions),
    }

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nNumber of queries evaluated: {results['num_queries']}\n")

    print("Precision@k:")
    for k, value in results['precision@k'].items():
        print(f"  P@{k:2d}: {value:.4f}")

    print("\nRecall@k:")
    for k, value in results['recall@k'].items():
        print(f"  R@{k:2d}: {value:.4f}")

    print(f"\nMean Reciprocal Rank (MRR): {results['mrr']:.4f}")
    print(f"Mean Average Precision (MAP): {results['map']:.4f}")
    print("=" * 80)


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Retrieval System")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval_config.yaml",
        help="Path to retrieval configuration"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/embeddings",
        help="Directory with FAISS index"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test queries JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/retrieval_eval_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help="K values for Precision@k and Recall@k"
    )

    args = parser.parse_args()

    # Load retriever
    print("Loading retrieval system...")
    retriever = load_retriever_from_config(args.config)

    # Load index
    print(f"Loading index from {args.index_dir}...")
    retriever.load_index(args.index_dir)
    print(f"Index loaded: {retriever.get_num_documents()} documents")

    # Load test data
    test_queries = load_test_data(args.test_data)

    # Evaluate
    results = evaluate_retrieval(retriever, test_queries, args.k_values)

    # Print results
    print_results(results)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    # Example usage for testing
    print("Retrieval Evaluation Script")
    print("=" * 80)
    print("\nThis script evaluates the retrieval pipeline using standard IR metrics.")
    print("\nUsage:")
    print("  uv run python -m src.evaluation.retrieval_eval \\")
    print("      --config configs/retrieval_config.yaml \\")
    print("      --index-dir data/embeddings \\")
    print("      --test-data data/test_queries.json")
    print("\nTest data format:")
    print("  [{")
    print('    "query": "What are the elements of negligence?",')
    print('    "relevant_doc_ids": [1, 5, 12]')
    print("  }, ...]")
    print("\nMetrics computed:")
    print("  - Precision@k: Fraction of retrieved docs that are relevant")
    print("  - Recall@k: Fraction of relevant docs that are retrieved")
    print("  - MRR: Mean Reciprocal Rank of first relevant doc")
    print("  - MAP: Mean Average Precision across all queries")
