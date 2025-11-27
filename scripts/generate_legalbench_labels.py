"""
Generate training labels for LegalBench mini dataset using OpenAI GPT-5.1.

This script:
1. Loads LegalBench mini queries (776 samples)
2. Uses existing retriever to get passages for each query
3. Uses ground truth snippets as "answers"
4. Generates reflection token labels using GPT-5.1 (primary) with Qwen fallback
5. Saves labeled data for training Self-RAG models
"""

import sys
from pathlib import Path
import json
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.generate_labels import LabelGenerator
from retrieval.retriever import LegalRetriever
from retrieval.embedding import EmbeddingModel
from retrieval.indexing import VectorIndex


def create_training_examples(
    queries_file: str = "data/legalbench-rag/queries.json",
    retriever_index_path: str = "data/legalbench_embeddings",
    num_samples: int = 776,  # Mini dataset size
    output_dir: str = "data/training",
    use_openai: bool = True,
    openai_model: str = "gpt-5.1",
    reasoning_effort: str = "auto",
    use_local_llm: bool = True,
    local_model: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """
    Create training examples with reflection token labels.

    Args:
        queries_file: Path to LegalBench queries
        retriever_index_path: Path to FAISS index
        num_samples: Number of queries to process
        output_dir: Output directory for labeled data
        use_openai: Whether to use OpenAI API (primary)
        openai_model: OpenAI model to use
        reasoning_effort: Reasoning effort for GPT-5.1
        use_local_llm: Whether to use local LLM (fallback)
        local_model: Local LLM model name
    """
    print("=" * 80)
    print("LegalBench Label Generation for Self-RAG Training")
    print("=" * 80)
    print()

    # Load queries
    print(f"Loading queries from {queries_file}...")
    with open(queries_file, 'r') as f:
        data = json.load(f)

    queries = data['tests'][:num_samples]
    print(f"Loaded {len(queries)} queries")
    print()

    # Initialize retriever
    print(f"Loading retriever from {retriever_index_path}...")

    # Load embedding model
    embedding_model = EmbeddingModel(model_name="sentence-transformers/all-mpnet-base-v2")

    # Initialize retriever
    retriever = LegalRetriever(
        embedding_model=embedding_model,
        top_k=3
    )

    # Load pre-built index
    retriever.load_index(retriever_index_path)
    print(f"Retriever loaded successfully!")
    print()

    # Initialize label generator
    print("Initializing label generator...")
    generator = LabelGenerator(
        use_openai=use_openai,
        use_local_llm=use_local_llm,
        model=openai_model,
        reasoning_effort=reasoning_effort,
        local_model=local_model,
    )
    print()

    # Process queries
    print(f"Processing {len(queries)} queries...")
    print()

    labeled_examples = []

    for i, query_data in enumerate(tqdm(queries, desc="Generating labels")):
        query = query_data['query']

        # Get ground truth snippet
        if query_data['snippets'] and len(query_data['snippets']) > 0:
            gt_snippet = query_data['snippets'][0]
            answer = gt_snippet.get('answer', '')
        else:
            answer = ""

        # Retrieve passages
        try:
            retrieved = retriever.retrieve(query, top_k=3)

            if retrieved:
                # Use top retrieved passage
                passage = retrieved[0]['text']
                passage_source = retrieved[0].get('source', 'unknown')
            else:
                passage = ""
                passage_source = "none"
        except Exception as e:
            print(f"\nWarning: Retrieval failed for query {i}: {e}")
            passage = ""
            passage_source = "error"

        # Generate labels
        try:
            labels = generator.generate_all_labels(
                question=query,
                passage=passage if passage else None,
                answer=answer if answer else None,
            )
        except Exception as e:
            print(f"\nError generating labels for query {i}: {e}")
            continue

        # Create labeled example
        labeled_example = {
            'query_id': i,
            'question': query,
            'passage': passage,
            'passage_source': passage_source,
            'answer': answer,
            'ground_truth_snippet': query_data['snippets'][0] if query_data['snippets'] else None,
            'dataset_source': query_data.get('dataset_source', 'unknown'),
            'reflection_tokens': labels,
        }

        labeled_examples.append(labeled_example)

    # Save labeled data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / 'legalbench_training_labels.json'

    with open(output_file, 'w') as f:
        json.dump(labeled_examples, f, indent=2)

    print()
    print("=" * 80)
    print(f"✓ Generated labels for {len(labeled_examples)} examples")
    print(f"✓ Saved to {output_file}")
    print("=" * 80)

    # Print label distribution
    print()
    print("Label Distribution:")
    print("-" * 80)

    token_types = ['retrieve', 'isrel', 'issup', 'isuse']

    for token_type in token_types:
        if any(token_type in ex['reflection_tokens'] for ex in labeled_examples):
            values = [ex['reflection_tokens'].get(token_type, 'N/A') for ex in labeled_examples]
            unique_values = set(v for v in values if v != 'N/A')

            print(f"\n{token_type.upper()}:")
            for val in unique_values:
                count = values.count(val)
                pct = (count / len(values)) * 100
                print(f"  {val}: {count} ({pct:.1f}%)")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training labels for LegalBench mini dataset"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="data/legalbench-rag/queries.json",
        help="Path to LegalBench queries file",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/legalbench_embeddings",
        help="Path to retriever FAISS index",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=776,
        help="Number of samples to process (776 = mini, 6858 = full)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory for labeled data",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        default=True,
        help="Use OpenAI API (GPT-5.1) as primary (default: True)",
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Disable OpenAI and use only local LLM",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5.1",
        help="OpenAI model to use (default: gpt-5.1)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="auto",
        choices=["auto", "none", "low", "medium", "high"],
        help="Reasoning effort for GPT-5.1 (default: auto)",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Local LLM model for fallback",
    )
    parser.add_argument(
        "--no-local-llm",
        action="store_true",
        help="Disable local LLM fallback",
    )

    args = parser.parse_args()

    create_training_examples(
        queries_file=args.queries,
        retriever_index_path=args.index,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        use_openai=args.use_openai and not args.no_openai,
        openai_model=args.openai_model,
        reasoning_effort=args.reasoning_effort,
        use_local_llm=not args.no_local_llm,
        local_model=args.local_model,
    )
