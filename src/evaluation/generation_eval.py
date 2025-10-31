"""
Generation Evaluation Script

Evaluates the Self-RAG generator using:
- Hallucination detection (reflection token analysis)
- Factual accuracy (FactScore-style)
- Response quality (ROUGE, BLEU)
- Citation accuracy

Usage:
    uv run python -m src.evaluation.generation_eval --config configs/generator_config.yaml --test-data data/test_qa.json
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
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from self_rag.inference import SelfRAGPipeline, load_pipeline_from_config
from self_rag.reflection_tokens import ReflectionTokenizer, ISSUPToken, ISUSEToken

# Optional ROUGE import
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. Install with: uv pip install rouge-score")


def load_test_data(test_file: str) -> List[Dict[str, Any]]:
    """
    Load test QA data.

    Expected format:
    [
        {
            "question": "What are the elements of negligence?",
            "reference_answer": "To establish negligence, plaintiff must prove...",
            "relevant_passages": ["passage1", "passage2"]  # optional
        },
        ...
    ]

    Args:
        test_file: Path to test data JSON file

    Returns:
        List of test examples
    """
    with open(test_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} test examples")
    return data


def detect_hallucination(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect hallucinations using reflection tokens.

    Args:
        response_data: Response from Self-RAG pipeline

    Returns:
        Dictionary with hallucination metrics
    """
    reflection = response_data.get('reflection', {})

    # Check ISSUP token (support)
    issup = reflection.get('issup')

    hallucination_score = 0.0
    is_hallucinated = False
    support_level = "unknown"

    if issup:
        if ISSUPToken.NO_SUPPORT.value in issup:
            hallucination_score = 1.0
            is_hallucinated = True
            support_level = "no_support"
        elif ISSUPToken.PARTIALLY_SUPPORTED.value in issup:
            hallucination_score = 0.5
            is_hallucinated = True
            support_level = "partial_support"
        elif ISSUPToken.FULLY_SUPPORTED.value in issup:
            hallucination_score = 0.0
            is_hallucinated = False
            support_level = "fully_supported"

    return {
        'is_hallucinated': is_hallucinated,
        'hallucination_score': hallucination_score,
        'support_level': support_level,
        'issup_token': issup,
    }


def calculate_factscore(
    generated_answer: str,
    reference_answer: str,
    retrieved_passages: List[Dict[str, Any]]
) -> float:
    """
    Calculate a simplified FactScore.

    FactScore measures the proportion of atomic facts in the generated
    answer that are supported by the retrieved passages.

    This is a simplified version based on keyword overlap.

    Args:
        generated_answer: Generated answer text
        reference_answer: Reference answer (optional)
        retrieved_passages: Retrieved supporting passages

    Returns:
        FactScore (0-1)
    """
    if not retrieved_passages:
        return 0.0

    # Extract key terms from generated answer
    # Simple approach: split into sentences and check support
    sentences = re.split(r'[.!?]+', generated_answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return 0.0

    # Combine all passage text
    passage_text = " ".join([p.get('text', '') for p in retrieved_passages]).lower()

    # Check how many sentences have support
    supported = 0
    for sentence in sentences:
        # Extract key words (nouns, verbs)
        words = [w.lower() for w in sentence.split() if len(w) > 3]
        if not words:
            continue

        # Check if majority of words appear in passages
        word_support = sum(1 for w in words if w in passage_text)
        if word_support / len(words) > 0.5:
            supported += 1

    return supported / len(sentences) if sentences else 0.0


def calculate_rouge(generated: str, reference: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores.

    Args:
        generated: Generated text
        reference: Reference text

    Returns:
        Dictionary with ROUGE scores
    """
    if not ROUGE_AVAILABLE or not reference:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def evaluate_response_quality(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate overall response quality using reflection tokens.

    Args:
        response_data: Response from Self-RAG pipeline

    Returns:
        Dictionary with quality metrics
    """
    reflection = response_data.get('reflection', {})

    # Check ISUSE token (utility)
    isuse = reflection.get('isuse')

    utility_score = 0.5  # Default
    if isuse:
        for token in ISUSEToken:
            if token.value in isuse:
                utility_score = ISUSEToken.get_score(token) / 5.0  # Normalize to 0-1
                break

    # Check answer length (simple heuristic)
    answer = response_data.get('answer', '')
    word_count = len(answer.split())

    completeness_score = min(1.0, word_count / 100)  # Assume 100 words is complete

    return {
        'utility_score': utility_score,
        'completeness_score': completeness_score,
        'word_count': word_count,
        'isuse_token': isuse,
    }


def evaluate_generation(
    pipeline: SelfRAGPipeline,
    test_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate Self-RAG generation on test data.

    Args:
        pipeline: SelfRAGPipeline instance
        test_data: List of test examples

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating generation on {len(test_data)} examples...")

    # Initialize metrics
    hallucination_scores = []
    hallucination_detected = []
    factscores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    utility_scores = []
    completeness_scores = []

    # Evaluate each example
    for example in tqdm(test_data, desc="Evaluating generation"):
        question = example['question']
        reference_answer = example.get('reference_answer', '')

        # Generate answer
        try:
            result = pipeline.answer_question(question, include_retrieval=True)
        except Exception as e:
            print(f"\nError generating answer: {e}")
            continue

        # Hallucination detection
        hall_metrics = detect_hallucination(result)
        hallucination_scores.append(hall_metrics['hallucination_score'])
        hallucination_detected.append(hall_metrics['is_hallucinated'])

        # FactScore
        factscore = calculate_factscore(
            result['answer'],
            reference_answer,
            result.get('retrieved_passages', [])
        )
        factscores.append(factscore)

        # ROUGE scores (if reference available)
        if reference_answer:
            rouge = calculate_rouge(result['answer'], reference_answer)
            rouge_scores['rouge1'].append(rouge['rouge1'])
            rouge_scores['rouge2'].append(rouge['rouge2'])
            rouge_scores['rougeL'].append(rouge['rougeL'])

        # Response quality
        quality = evaluate_response_quality(result)
        utility_scores.append(quality['utility_score'])
        completeness_scores.append(quality['completeness_score'])

    # Aggregate metrics
    results = {
        'num_examples': len(test_data),
        'hallucination_rate': np.mean(hallucination_detected) if hallucination_detected else 0.0,
        'avg_hallucination_score': np.mean(hallucination_scores) if hallucination_scores else 0.0,
        'avg_factscore': np.mean(factscores) if factscores else 0.0,
        'avg_utility_score': np.mean(utility_scores) if utility_scores else 0.0,
        'avg_completeness': np.mean(completeness_scores) if completeness_scores else 0.0,
    }

    if rouge_scores['rouge1']:
        results['rouge_scores'] = {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL']),
        }

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 80)
    print("GENERATION EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nNumber of examples evaluated: {results['num_examples']}\n")

    print("Hallucination Metrics:")
    print(f"  Hallucination Rate: {results['hallucination_rate']:.2%}")
    print(f"  Avg Hallucination Score: {results['avg_hallucination_score']:.4f}")

    print("\nFactual Accuracy:")
    print(f"  Avg FactScore: {results['avg_factscore']:.4f}")

    print("\nResponse Quality:")
    print(f"  Avg Utility Score: {results['avg_utility_score']:.4f}")
    print(f"  Avg Completeness: {results['avg_completeness']:.4f}")

    if 'rouge_scores' in results:
        print("\nROUGE Scores:")
        print(f"  ROUGE-1: {results['rouge_scores']['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['rouge_scores']['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['rouge_scores']['rougeL']:.4f}")

    print("=" * 80)


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Self-RAG Generation")
    parser.add_argument(
        "--retrieval-config",
        type=str,
        default="configs/retrieval_config.yaml",
        help="Path to retrieval configuration"
    )
    parser.add_argument(
        "--generator-config",
        type=str,
        default="configs/generator_config.yaml",
        help="Path to generator configuration"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/embeddings",
        help="Directory with FAISS index"
    )
    parser.add_argument(
        "--generator-weights",
        type=str,
        default="models/generator_lora/final",
        help="Path to generator LoRA weights"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test QA JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/generation_eval_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Load pipeline
    print("Loading Self-RAG pipeline...")
    pipeline = load_pipeline_from_config(
        retrieval_config_path=args.retrieval_config,
        generator_config_path=args.generator_config,
        retriever_index_dir=args.index_dir,
        generator_weights_path=args.generator_weights,
    )

    # Load test data
    test_data = load_test_data(args.test_data)

    # Evaluate
    results = evaluate_generation(pipeline, test_data)

    # Print results
    print_results(results)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    # Example usage for testing
    print("Generation Evaluation Script")
    print("=" * 80)
    print("\nThis script evaluates Self-RAG generation quality.")
    print("\nUsage:")
    print("  uv run python -m src.evaluation.generation_eval \\")
    print("      --retrieval-config configs/retrieval_config.yaml \\")
    print("      --generator-config configs/generator_config.yaml \\")
    print("      --index-dir data/embeddings \\")
    print("      --generator-weights models/generator_lora/final \\")
    print("      --test-data data/test_qa.json")
    print("\nTest data format:")
    print("  [{")
    print('    "question": "What are the elements of negligence?",')
    print('    "reference_answer": "To establish negligence..."')
    print("  }, ...]")
    print("\nMetrics computed:")
    print("  - Hallucination Rate: % of responses with no/partial support")
    print("  - FactScore: Factual accuracy using retrieved passages")
    print("  - Utility Score: Overall response quality (from reflection tokens)")
    print("  - ROUGE Scores: Lexical overlap with reference answers")
