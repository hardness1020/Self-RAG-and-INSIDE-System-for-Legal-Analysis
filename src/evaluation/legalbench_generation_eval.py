"""
LegalBench Generation Evaluation Module

Evaluates generation quality on LegalBench-RAG benchmark.
Compares: No-RAG, Basic RAG, Self-RAG, Self-RAG+INSIDE.

Metrics:
- F1 Score: Token-level overlap with ground truth
- ROUGE-L: Longest common subsequence
- Hallucination Rate: From reflection tokens
- Utility Score: From ISUSE tokens
"""

from typing import Dict, List, Any, Optional, Tuple
import json
from collections import Counter
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenization.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    return normalize_text(text).split()


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        F1 score (0-1)
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # Count token frequencies
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    # Calculate overlap
    overlap = sum((pred_counter & gt_counter).values())

    if overlap == 0:
        return 0.0

    # Precision and recall
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """
    Compute ROUGE-L (Longest Common Subsequence) score.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        ROUGE-L F1 score (0-1)
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # Compute LCS length using dynamic programming
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    # ROUGE-L precision and recall
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    # ROUGE-L F1
    rouge_l = 2 * (precision * recall) / (precision + recall)

    return rouge_l


def extract_hallucination_metrics(reflection_tokens: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract hallucination metrics from reflection tokens.

    Args:
        reflection_tokens: Dictionary with ISSUP, ISREL, etc.

    Returns:
        Dictionary with hallucination metrics
    """
    metrics = {
        'has_reflection_tokens': len(reflection_tokens) > 0,
        'is_supported': False,
        'is_relevant': False,
        'utility_score': 0.0,
        'hallucination_detected': False,
    }

    # ISSUP: Support level
    issup = reflection_tokens.get('issup', '')
    if '[Fully Supported]' in issup:
        metrics['is_supported'] = True
        metrics['hallucination_detected'] = False
    elif '[Partially Supported]' in issup:
        metrics['is_supported'] = False
        metrics['hallucination_detected'] = True
    elif '[No Support]' in issup:
        metrics['is_supported'] = False
        metrics['hallucination_detected'] = True

    # ISREL: Relevance
    isrel = reflection_tokens.get('isrel', '')
    if '[Relevant]' in isrel:
        metrics['is_relevant'] = True

    # ISUSE: Utility (normalize to 0-1)
    isuse = reflection_tokens.get('isuse', '')
    for i in range(5, 0, -1):
        if f'[Utility:{i}]' in isuse:
            metrics['utility_score'] = i / 5.0
            break

    return metrics


def evaluate_generation(
    prediction: str,
    ground_truth: str,
    reflection_tokens: Optional[Dict[str, str]] = None,
    eigenscore: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single generated response.

    Args:
        prediction: Generated answer
        ground_truth: Ground truth snippet/answer
        reflection_tokens: Optional reflection tokens from Self-RAG
        eigenscore: Optional INSIDE EigenScore

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        # Lexical metrics
        'f1_score': compute_f1_score(prediction, ground_truth),
        'rouge_l': compute_rouge_l(prediction, ground_truth),

        # Length metrics
        'prediction_length': len(prediction.split()),
        'ground_truth_length': len(ground_truth.split()),

        # Hallucination metrics (if available)
        'has_reflection_tokens': False,
        'is_supported': None,
        'is_relevant': None,
        'utility_score': None,
        'hallucination_detected': None,

        # INSIDE metrics (if available)
        'eigenscore': eigenscore,
        'inside_hallucination_detected': eigenscore > 5.0 if eigenscore else None,
    }

    # Extract reflection token metrics
    if reflection_tokens:
        hallucination_metrics = extract_hallucination_metrics(reflection_tokens)
        metrics.update(hallucination_metrics)

    return metrics


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple examples.

    Args:
        results: List of per-example metric dictionaries

    Returns:
        Aggregated metrics
    """
    if not results:
        return {}

    # Count valid examples for each metric
    n = len(results)

    # Aggregate lexical metrics
    aggregated = {
        'num_examples': n,
        'avg_f1_score': sum(r['f1_score'] for r in results) / n,
        'avg_rouge_l': sum(r['rouge_l'] for r in results) / n,
        'avg_prediction_length': sum(r['prediction_length'] for r in results) / n,
        'avg_ground_truth_length': sum(r['ground_truth_length'] for r in results) / n,
    }

    # Aggregate hallucination metrics (only for examples with reflection tokens)
    results_with_tokens = [r for r in results if r['has_reflection_tokens']]

    if results_with_tokens:
        n_with_tokens = len(results_with_tokens)
        aggregated.update({
            'num_with_reflection_tokens': n_with_tokens,
            'hallucination_rate': sum(
                1 for r in results_with_tokens if r['hallucination_detected']
            ) / n_with_tokens,
            'support_rate': sum(
                1 for r in results_with_tokens if r['is_supported']
            ) / n_with_tokens,
            'relevance_rate': sum(
                1 for r in results_with_tokens if r['is_relevant']
            ) / n_with_tokens,
            'avg_utility_score': sum(
                r['utility_score'] for r in results_with_tokens
            ) / n_with_tokens,
        })

    # Aggregate INSIDE metrics
    results_with_eigenscore = [r for r in results if r['eigenscore'] is not None]

    if results_with_eigenscore:
        n_with_eigenscore = len(results_with_eigenscore)
        aggregated.update({
            'num_with_eigenscore': n_with_eigenscore,
            'avg_eigenscore': sum(
                r['eigenscore'] for r in results_with_eigenscore
            ) / n_with_eigenscore,
            'inside_hallucination_rate': sum(
                1 for r in results_with_eigenscore if r['inside_hallucination_detected']
            ) / n_with_eigenscore,
        })

    return aggregated


def compare_methods(
    results_by_method: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple methods on the same dataset.

    Args:
        results_by_method: Dictionary mapping method name to list of results

    Returns:
        Dictionary with aggregated metrics per method
    """
    comparison = {}

    for method_name, results in results_by_method.items():
        comparison[method_name] = aggregate_metrics(results)

    return comparison


def evaluate_dataset(
    predictions_file: str,
    queries_file: str,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate predictions on LegalBench dataset.

    Args:
        predictions_file: JSON file with predictions
        queries_file: LegalBench queries file
        output_file: Optional output file for detailed results

    Returns:
        Evaluation metrics
    """
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # Load queries
    with open(queries_file, 'r') as f:
        queries_data = json.load(f)

    queries = queries_data['tests']

    # Evaluate each prediction
    results = []

    for pred_data in predictions:
        query_id = pred_data['query_id']
        prediction = pred_data['answer']

        # Get ground truth
        query_data = queries[query_id]
        if query_data['snippets']:
            ground_truth = query_data['snippets'][0]['answer']
        else:
            ground_truth = ""

        # Evaluate
        metrics = evaluate_generation(
            prediction=prediction,
            ground_truth=ground_truth,
            reflection_tokens=pred_data.get('reflection_tokens'),
            eigenscore=pred_data.get('eigenscore'),
        )

        results.append({
            'query_id': query_id,
            'query': query_data['query'],
            'prediction': prediction,
            'ground_truth': ground_truth,
            **metrics
        })

    # Aggregate
    aggregated = aggregate_metrics(results)

    # Save detailed results if requested
    if output_file:
        output = {
            'aggregated_metrics': aggregated,
            'per_example_results': results,
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

    return aggregated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generation on LegalBench")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="JSON file with predictions",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="data/legalbench-rag/queries.json",
        help="LegalBench queries file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results",
    )

    args = parser.parse_args()

    metrics = evaluate_dataset(
        predictions_file=args.predictions,
        queries_file=args.queries,
        output_file=args.output,
    )

    print("\n" + "=" * 80)
    print("LegalBench Generation Evaluation Results")
    print("=" * 80)
    print(json.dumps(metrics, indent=2))
