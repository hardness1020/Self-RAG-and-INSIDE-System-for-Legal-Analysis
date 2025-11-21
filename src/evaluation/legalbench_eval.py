"""
LegalBench-RAG Evaluation Script

Evaluates retrieval systems using the LegalBench-RAG benchmark.
Implements both document-level and snippet-level (character span) metrics.

Based on: "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation
          in the Legal Domain" (Pipitone & Houir Alami, 2024)

Metrics:
- Document-level: Precision@k, Recall@k (standard IR metrics)
- Snippet-level: Snippet Precision@k, Snippet Recall@k (with IoU overlap)
- Aggregated by dataset: ContractNLI, CUAD, MAUD, PrivacyQA
"""

import os
import sys
from pathlib import Path
import argparse
import json
import yaml
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retriever import LegalRetriever, load_retriever_from_config
from data.legalbench_loader import LegalBenchRAGLoader, LegalBenchSnippet


def calculate_snippet_overlap(
    retrieved_span: Tuple[int, int],
    ground_truth_span: Tuple[int, int],
    min_iou: float = 0.5
) -> bool:
    """
    Calculate whether two character spans overlap sufficiently.

    Uses Intersection over Union (IoU) metric.

    Args:
        retrieved_span: (start, end) of retrieved chunk
        ground_truth_span: (start, end) of ground truth snippet
        min_iou: Minimum IoU threshold to consider a match

    Returns:
        True if overlap meets threshold
    """
    r_start, r_end = retrieved_span
    gt_start, gt_end = ground_truth_span

    # Calculate intersection
    intersection_start = max(r_start, gt_start)
    intersection_end = min(r_end, gt_end)
    intersection_length = max(0, intersection_end - intersection_start)

    # Calculate union
    union_start = min(r_start, gt_start)
    union_end = max(r_end, gt_end)
    union_length = union_end - union_start

    # IoU
    if union_length == 0:
        return False

    iou = intersection_length / union_length
    return iou >= min_iou


def calculate_snippet_precision_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_snippets: List[Dict[str, Any]],
    k: int,
    min_iou: float = 0.5
) -> float:
    """
    Calculate snippet-level Precision@k.

    Snippet Precision@k = (# retrieved chunks matching ground truth snippets) / k

    Args:
        retrieved_chunks: List of retrieved chunks with file_path and span
        ground_truth_snippets: List of ground truth snippets with file_path and span
        k: Cutoff rank
        min_iou: Minimum IoU for snippet match

    Returns:
        Snippet Precision@k score
    """
    if k == 0 or not ground_truth_snippets:
        return 0.0

    top_k_chunks = retrieved_chunks[:k]
    num_matches = 0

    for chunk in top_k_chunks:
        # Check top-level first, then check inside metadata dict
        chunk_file = chunk.get('file_path') or chunk.get('source') or \
                     chunk.get('metadata', {}).get('file_path') or \
                     chunk.get('metadata', {}).get('source', '')
        chunk_span = chunk.get('span', (chunk.get('start_char', 0), chunk.get('end_char', 0)))

        # Check if this chunk matches any ground truth snippet
        for gt_snippet in ground_truth_snippets:
            gt_file = gt_snippet['file_path']
            gt_span = tuple(gt_snippet['span'])

            # Must be from same document
            if chunk_file == gt_file:
                if calculate_snippet_overlap(chunk_span, gt_span, min_iou):
                    num_matches += 1
                    break  # Count each chunk once

    return num_matches / k


def calculate_snippet_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_snippets: List[Dict[str, Any]],
    k: int,
    min_iou: float = 0.5
) -> float:
    """
    Calculate snippet-level Recall@k.

    Snippet Recall@k = (# ground truth snippets found in top k) / (total # ground truth snippets)

    Args:
        retrieved_chunks: List of retrieved chunks with file_path and span
        ground_truth_snippets: List of ground truth snippets with file_path and span
        k: Cutoff rank
        min_iou: Minimum IoU for snippet match

    Returns:
        Snippet Recall@k score
    """
    if not ground_truth_snippets:
        return 0.0

    top_k_chunks = retrieved_chunks[:k]
    snippets_found = set()

    for i, gt_snippet in enumerate(ground_truth_snippets):
        gt_file = gt_snippet['file_path']
        gt_span = tuple(gt_snippet['span'])

        # Check if any retrieved chunk matches this ground truth snippet
        for chunk in top_k_chunks:
            # Check top-level first, then check inside metadata dict
            chunk_file = chunk.get('file_path') or chunk.get('source') or \
                         chunk.get('metadata', {}).get('file_path') or \
                         chunk.get('metadata', {}).get('source', '')
            chunk_span = chunk.get('span', (chunk.get('start_char', 0), chunk.get('end_char', 0)))

            if chunk_file == gt_file:
                if calculate_snippet_overlap(chunk_span, gt_span, min_iou):
                    snippets_found.add(i)
                    break

    return len(snippets_found) / len(ground_truth_snippets)


def calculate_document_precision_at_k(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int
) -> float:
    """
    Calculate document-level Precision@k.

    Args:
        retrieved_docs: List of retrieved document IDs/paths (ranked)
        relevant_docs: List of ground truth relevant document IDs/paths
        k: Cutoff rank

    Returns:
        Document Precision@k score
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)

    num_relevant = len(retrieved_at_k & relevant_set)
    return num_relevant / k


def calculate_document_recall_at_k(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int
) -> float:
    """
    Calculate document-level Recall@k.

    Args:
        retrieved_docs: List of retrieved document IDs/paths (ranked)
        relevant_docs: List of ground truth relevant document IDs/paths
        k: Cutoff rank

    Returns:
        Document Recall@k score
    """
    if not relevant_docs:
        return 0.0

    retrieved_at_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)

    num_relevant = len(retrieved_at_k & relevant_set)
    return len(relevant_docs) > 0 and num_relevant / len(relevant_docs) or 0.0


def evaluate_legalbench_rag(
    retriever: LegalRetriever,
    loader: LegalBenchRAGLoader,
    k_values: List[int] = [1, 2, 4, 8, 16, 32, 64],
    min_iou: float = 0.5,
    evaluate_snippets: bool = True
) -> Dict[str, Any]:
    """
    Evaluate retrieval system on LegalBench-RAG benchmark.

    Args:
        retriever: LegalRetriever instance
        loader: LegalBenchRAGLoader with loaded queries
        k_values: List of k values for metrics (paper uses 1,2,4,8,16,32,64)
        min_iou: Minimum IoU for snippet matching
        evaluate_snippets: Whether to compute snippet-level metrics

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating on {len(loader.queries)} LegalBench-RAG queries...")

    # Initialize metrics
    doc_precision_at_k = {k: [] for k in k_values}
    doc_recall_at_k = {k: [] for k in k_values}
    snippet_precision_at_k = {k: [] for k in k_values}
    snippet_recall_at_k = {k: [] for k in k_values}

    # Track per-dataset metrics
    datasets = ['ContractNLI', 'CUAD', 'MAUD', 'PrivacyQA']
    per_dataset_metrics = {
        ds: {
            'doc_precision': {k: [] for k in k_values},
            'doc_recall': {k: [] for k in k_values},
            'snippet_precision': {k: [] for k in k_values},
            'snippet_recall': {k: [] for k in k_values},
            'count': 0
        }
        for ds in datasets
    }

    # Evaluate each query
    for query_obj in tqdm(loader.queries, desc="Evaluating queries"):
        query = query_obj.query
        dataset_source = query_obj.dataset_source

        # Ground truth
        relevant_docs = query_obj.get_document_ids()
        ground_truth_snippets = [
            {
                'file_path': s.file_path,
                'span': s.span,
                'text': s.answer
            }
            for s in query_obj.snippets
        ]

        # Retrieve documents
        max_k = max(k_values)
        results = retriever.retrieve(query, top_k=max_k)

        # Extract retrieved document IDs and spans
        retrieved_docs = []
        retrieved_chunks = []
        for doc in results:
            # Check top-level first, then check inside metadata dict
            doc_id = doc.get('source') or doc.get('file_path') or \
                     doc.get('metadata', {}).get('source') or \
                     doc.get('metadata', {}).get('file_path', '')
            retrieved_docs.append(doc_id)

            # Extract span information if available
            # Note: If positions aren't tracked, both default to 0 (indicating no position info)
            # rather than creating false spans with len(text)
            chunk_info = {
                'file_path': doc_id,
                'span': (doc.get('start_char', 0), doc.get('end_char', 0)),
                'text': doc.get('text', ''),
                'metadata': doc.get('metadata', {})  # Preserve for debugging
            }
            retrieved_chunks.append(chunk_info)

        # Calculate document-level metrics
        for k in k_values:
            doc_prec = calculate_document_precision_at_k(retrieved_docs, relevant_docs, k)
            doc_rec = calculate_document_recall_at_k(retrieved_docs, relevant_docs, k)

            doc_precision_at_k[k].append(doc_prec)
            doc_recall_at_k[k].append(doc_rec)

            # Track per-dataset
            if dataset_source in per_dataset_metrics:
                per_dataset_metrics[dataset_source]['doc_precision'][k].append(doc_prec)
                per_dataset_metrics[dataset_source]['doc_recall'][k].append(doc_rec)

        # Calculate snippet-level metrics
        if evaluate_snippets:
            for k in k_values:
                snip_prec = calculate_snippet_precision_at_k(
                    retrieved_chunks, ground_truth_snippets, k, min_iou
                )
                snip_rec = calculate_snippet_recall_at_k(
                    retrieved_chunks, ground_truth_snippets, k, min_iou
                )

                snippet_precision_at_k[k].append(snip_prec)
                snippet_recall_at_k[k].append(snip_rec)

                # Track per-dataset
                if dataset_source in per_dataset_metrics:
                    per_dataset_metrics[dataset_source]['snippet_precision'][k].append(snip_prec)
                    per_dataset_metrics[dataset_source]['snippet_recall'][k].append(snip_rec)

        # Update count
        if dataset_source in per_dataset_metrics:
            per_dataset_metrics[dataset_source]['count'] += 1

    # Aggregate overall metrics
    results = {
        'num_queries': len(loader.queries),
        'k_values': k_values,
        'document_precision@k': {k: float(np.mean(doc_precision_at_k[k])) for k in k_values},
        'document_recall@k': {k: float(np.mean(doc_recall_at_k[k])) for k in k_values},
    }

    if evaluate_snippets:
        results['snippet_precision@k'] = {k: float(np.mean(snippet_precision_at_k[k])) for k in k_values}
        results['snippet_recall@k'] = {k: float(np.mean(snippet_recall_at_k[k])) for k in k_values}
        results['min_iou'] = min_iou

    # Aggregate per-dataset metrics
    results['per_dataset'] = {}
    for ds_name, ds_metrics in per_dataset_metrics.items():
        if ds_metrics['count'] > 0:
            results['per_dataset'][ds_name] = {
                'count': ds_metrics['count'],
                'document_precision@k': {k: float(np.mean(ds_metrics['doc_precision'][k])) if ds_metrics['doc_precision'][k] else 0.0 for k in k_values},
                'document_recall@k': {k: float(np.mean(ds_metrics['doc_recall'][k])) if ds_metrics['doc_recall'][k] else 0.0 for k in k_values},
            }
            if evaluate_snippets:
                results['per_dataset'][ds_name]['snippet_precision@k'] = {k: float(np.mean(ds_metrics['snippet_precision'][k])) if ds_metrics['snippet_precision'][k] else 0.0 for k in k_values}
                results['per_dataset'][ds_name]['snippet_recall@k'] = {k: float(np.mean(ds_metrics['snippet_recall'][k])) if ds_metrics['snippet_recall'][k] else 0.0 for k in k_values}

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in formatted tables."""
    print("\n" + "=" * 100)
    print("LEGALBENCH-RAG EVALUATION RESULTS")
    print("=" * 100)
    print(f"\nNumber of queries evaluated: {results['num_queries']}\n")

    # Overall metrics
    print("OVERALL METRICS")
    print("-" * 100)

    print("\nDocument-level Precision@k:")
    for k in results['k_values']:
        value = results['document_precision@k'][k]
        print(f"  P@{k:2d}: {value:6.2%}")

    print("\nDocument-level Recall@k:")
    for k in results['k_values']:
        value = results['document_recall@k'][k]
        print(f"  R@{k:2d}: {value:6.2%}")

    if 'snippet_precision@k' in results:
        print(f"\nSnippet-level Precision@k (IoU >= {results['min_iou']}):")
        for k in results['k_values']:
            value = results['snippet_precision@k'][k]
            print(f"  P@{k:2d}: {value:6.2%}")

        print(f"\nSnippet-level Recall@k (IoU >= {results['min_iou']}):")
        for k in results['k_values']:
            value = results['snippet_recall@k'][k]
            print(f"  R@{k:2d}: {value:6.2%}")

    # Per-dataset metrics
    if 'per_dataset' in results:
        print("\n" + "=" * 100)
        print("PER-DATASET BREAKDOWN")
        print("=" * 100)

        for ds_name, ds_results in results['per_dataset'].items():
            print(f"\n{ds_name} ({ds_results['count']} queries)")
            print("-" * 100)

            print("  Document Precision@k: ", end="")
            print(" | ".join([f"@{k}:{ds_results['document_precision@k'][k]:5.2%}" for k in [1, 4, 16, 64]]))

            print("  Document Recall@k:    ", end="")
            print(" | ".join([f"@{k}:{ds_results['document_recall@k'][k]:5.2%}" for k in [1, 4, 16, 64]]))

            if 'snippet_precision@k' in ds_results:
                print("  Snippet Precision@k:  ", end="")
                print(" | ".join([f"@{k}:{ds_results['snippet_precision@k'][k]:5.2%}" for k in [1, 4, 16, 64]]))

                print("  Snippet Recall@k:     ", end="")
                print(" | ".join([f"@{k}:{ds_results['snippet_recall@k'][k]:5.2%}" for k in [1, 4, 16, 64]]))

    print("\n" + "=" * 100)

    # Comparison to paper baselines
    print("\nPAPER BASELINE COMPARISON (from Table 5 - RCTS method):")
    print("  PrivacyQA:   Precision@1: 14.38% | Recall@64: 84.19%")
    print("  ContractNLI: Precision@1:  6.63% | Recall@64: 61.72%")
    print("  MAUD:        Precision@1:  2.65% | Recall@64: 28.28%")
    print("  CUAD:        Precision@1:  1.97% | Recall@64: 74.70%")
    print("  Overall:     Precision@1:  6.41% | Recall@64: 62.22%")
    print("=" * 100)


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on LegalBench-RAG Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/legalbench_config.yaml",
        help="Path to LegalBench-RAG configuration"
    )
    parser.add_argument(
        "--retrieval-config",
        type=str,
        default="configs/retrieval_config.yaml",
        help="Path to retrieval system configuration"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Directory with FAISS index for LegalBench-RAG corpus"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/legalbench_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--use-mini",
        action="store_true",
        help="Use LegalBench-RAG-mini (776 queries)"
    )
    parser.add_argument(
        "--no-snippets",
        action="store_true",
        help="Disable snippet-level evaluation (faster)"
    )

    args = parser.parse_args()

    # Load LegalBench-RAG config
    with open(args.config, 'r') as f:
        lb_config = yaml.safe_load(f)

    # Load retrieval system
    print("Loading retrieval system...")
    retriever = load_retriever_from_config(args.retrieval_config)
    retriever.load_index(args.index_dir)
    print(f"Index loaded: {retriever.get_num_documents()} documents")

    # Load LegalBench-RAG dataset
    print("Loading LegalBench-RAG dataset...")
    loader = LegalBenchRAGLoader(
        corpus_dir=lb_config['corpus_dir'],
        queries_file=lb_config['queries_file'],
        use_mini=args.use_mini
    )
    loader.load_queries()

    # Show corpus statistics
    stats = loader.get_corpus_statistics()
    print("\nCorpus Statistics:")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Queries: {stats['num_queries']}")
    print(f"  Version: {stats['version']}")
    print(f"  By dataset: {stats['queries_by_dataset']}")

    # Evaluate
    results = evaluate_legalbench_rag(
        retriever=retriever,
        loader=loader,
        k_values=lb_config.get('k_values', [1, 2, 4, 8, 16, 32, 64]),
        min_iou=lb_config.get('min_iou', 0.5),
        evaluate_snippets=not args.no_snippets
    )

    # Print results
    print_results(results)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
