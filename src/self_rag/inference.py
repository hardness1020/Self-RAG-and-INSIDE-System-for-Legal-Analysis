"""
Inference Module

Complete Self-RAG inference pipeline combining:
1. Legal document retrieval
2. Self-RAG generator with reflection tokens
3. Adaptive retrieval based on model decisions

Provides end-to-end interface for querying the system.
"""

from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path
import yaml
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retriever import LegalRetriever, load_retriever_from_config
from self_rag.generator import SelfRAGGenerator, load_generator_from_config
from self_rag.reflection_tokens import ReflectionTokenizer, RetrieveToken


class SelfRAGPipeline:
    """
    Complete Self-RAG pipeline for legal question answering.

    Combines retrieval and generation with self-verification.
    """

    def __init__(
        self,
        generator: SelfRAGGenerator,
        retriever: LegalRetriever,
        adaptive_retrieval: bool = True,
        max_retrieval_steps: int = 3,
    ):
        """
        Initialize Self-RAG pipeline.

        Args:
            generator: Self-RAG generator model
            retriever: Legal document retriever
            adaptive_retrieval: Whether to use adaptive retrieval
            max_retrieval_steps: Maximum number of retrieval steps
        """
        self.generator = generator
        self.retriever = retriever
        self.adaptive_retrieval = adaptive_retrieval
        self.max_retrieval_steps = max_retrieval_steps

    def answer_question(
        self,
        question: str,
        include_retrieval: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Answer a legal question using Self-RAG.

        Args:
            question: Question to answer
            include_retrieval: Whether to retrieve supporting passages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with answer, reflection tokens, and retrieved passages
        """
        result = {
            'question': question,
            'answer': '',
            'reflection': {},
            'retrieved_passages': [],
            'retrieval_history': [],
        }

        # Initial retrieval (if not using adaptive retrieval)
        if include_retrieval and not self.adaptive_retrieval:
            retrieved_docs = self.retriever.retrieve(question, top_k=3)
            result['retrieved_passages'] = retrieved_docs

            # Format prompt with retrieved passage
            if retrieved_docs:
                best_passage = retrieved_docs[0]['text']
                prompt = f"Question: {question}\nPassage: {best_passage}\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        # Generate with adaptive retrieval if enabled
        if self.adaptive_retrieval and include_retrieval:
            response, retrieval_history = self.generator.generate_with_retrieval(
                question=question,
                retriever=self.retriever,
                max_new_tokens=max_new_tokens,
                max_retrieval_steps=self.max_retrieval_steps,
            )
            result['retrieval_history'] = retrieval_history
        else:
            # Standard generation
            response = self.generator.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        # Parse response
        parsed = self.generator.parse_response(response)
        result['answer'] = parsed['text']
        result['reflection'] = parsed['reflection']
        result['raw_response'] = parsed['raw_response']

        # Score response
        result['score'] = self.generator.score_response(response)

        return result

    def answer_batch(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.

        Args:
            questions: List of questions
            **kwargs: Arguments passed to answer_question

        Returns:
            List of result dictionaries
        """
        results = []
        for question in questions:
            result = self.answer_question(question, **kwargs)
            results.append(result)
        return results

    def format_response(
        self,
        result: Dict[str, Any],
        include_passages: bool = True,
        include_reflection: bool = True,
    ) -> str:
        """
        Format result as human-readable string.

        Args:
            result: Result dictionary from answer_question
            include_passages: Whether to include retrieved passages
            include_reflection: Whether to include reflection tokens

        Returns:
            Formatted string
        """
        output = []

        output.append(f"Question: {result['question']}\n")
        output.append(f"Answer: {result['answer']}\n")

        if include_reflection:
            output.append("\nSelf-Evaluation:")
            reflection = result['reflection']
            for token_type, token_value in reflection.items():
                if token_value:
                    output.append(f"  {token_type.upper()}: {token_value}")

            if 'score' in result:
                output.append(f"  Overall Score: {result['score']:.2f}")

        if include_passages and result.get('retrieved_passages'):
            output.append("\n\nRetrieved Passages:")
            for i, passage in enumerate(result['retrieved_passages'][:3]):
                output.append(f"\n{i+1}. (Score: {passage['score']:.3f})")
                output.append(f"   {passage['text'][:200]}...")

        if result.get('retrieval_history'):
            output.append("\n\nRetrieval History:")
            for step in result['retrieval_history']:
                output.append(f"  Step {step['step']}: Retrieved {len(step['documents'])} documents")

        return "\n".join(output)


def load_pipeline_from_config(
    retrieval_config_path: str,
    generator_config_path: str,
    retriever_index_dir: Optional[str] = None,
    generator_weights_path: Optional[str] = None,
) -> SelfRAGPipeline:
    """
    Load complete pipeline from configuration files.

    Args:
        retrieval_config_path: Path to retrieval config YAML
        generator_config_path: Path to generator config YAML
        retriever_index_dir: Directory with FAISS index (optional)
        generator_weights_path: Path to generator LoRA weights (optional)

    Returns:
        Configured SelfRAGPipeline
    """
    print("Loading Self-RAG Pipeline...\n")

    # Load retriever
    print("1. Loading retriever...")
    retriever = load_retriever_from_config(retrieval_config_path)

    if retriever_index_dir:
        print(f"   Loading index from {retriever_index_dir}")
        retriever.load_index(retriever_index_dir)
        print(f"   Index loaded: {retriever.get_num_documents()} documents")
    else:
        print("   No index loaded. Call retriever.index_documents() to build index.")

    # Load generator
    print("\n2. Loading generator...")
    generator = load_generator_from_config(generator_config_path)
    generator.load_model(lora_weights_path=generator_weights_path)

    # Load generator config for adaptive retrieval settings
    with open(generator_config_path, 'r') as f:
        gen_config = yaml.safe_load(f)

    inference_config = gen_config.get('inference', {})
    adaptive_retrieval = inference_config.get('adaptive_retrieval', True)

    # Create pipeline
    pipeline = SelfRAGPipeline(
        generator=generator,
        retriever=retriever,
        adaptive_retrieval=adaptive_retrieval,
        max_retrieval_steps=3,
    )

    print("\nPipeline loaded successfully!")
    return pipeline


def main():
    """
    Command-line interface for Self-RAG inference.
    """
    parser = argparse.ArgumentParser(description="Self-RAG Legal Q&A System")
    parser.add_argument(
        "--retrieval-config",
        type=str,
        default="configs/retrieval_config.yaml",
        help="Path to retrieval configuration",
    )
    parser.add_argument(
        "--generator-config",
        type=str,
        default="configs/generator_config.yaml",
        help="Path to generator configuration",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/embeddings",
        help="Directory with FAISS index",
    )
    parser.add_argument(
        "--generator-weights",
        type=str,
        default=None,
        help="Path to generator LoRA weights",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to answer (for single query mode)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )

    args = parser.parse_args()

    # Load pipeline
    pipeline = load_pipeline_from_config(
        retrieval_config_path=args.retrieval_config,
        generator_config_path=args.generator_config,
        retriever_index_dir=args.index_dir,
        generator_weights_path=args.generator_weights,
    )

    # Single query mode
    if args.query:
        print("\n" + "=" * 80)
        result = pipeline.answer_question(args.query)
        print(pipeline.format_response(result))
        print("=" * 80)

    # Interactive mode
    elif args.interactive:
        print("\n" + "=" * 80)
        print("Self-RAG Legal Q&A System - Interactive Mode")
        print("Type 'quit' or 'exit' to exit")
        print("=" * 80 + "\n")

        while True:
            question = input("Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not question:
                continue

            print("\nGenerating answer...\n")
            result = pipeline.answer_question(question)
            print(pipeline.format_response(result))
            print("\n" + "-" * 80 + "\n")

    else:
        # Example queries
        print("\n" + "=" * 80)
        print("Running example queries...\n")

        example_questions = [
            "What are the elements of negligence?",
            "What is the standard for breach of duty?",
            "What types of damages can be recovered in negligence cases?",
        ]

        for i, question in enumerate(example_questions, 1):
            print(f"\nExample {i}:")
            print("=" * 80)
            result = pipeline.answer_question(question)
            print(pipeline.format_response(result))
            print("\n")


if __name__ == "__main__":
    # For testing without command line args
    print("Self-RAG Inference Pipeline")
    print("=" * 80)
    print("\nThis module provides:")
    print("1. Complete Self-RAG pipeline for legal Q&A")
    print("2. Adaptive retrieval with self-verification")
    print("3. Command-line and interactive interfaces")
    print("\nUsage:")
    print("  python -m src.self_rag.inference --query 'Your question here'")
    print("  python -m src.self_rag.inference --interactive")
    print("\nFor full functionality, train models first using:")
    print("  python -m src.training.train_critic_qlora")
    print("  python -m src.training.train_generator_qlora")
