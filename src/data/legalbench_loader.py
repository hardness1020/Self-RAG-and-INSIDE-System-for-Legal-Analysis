"""
LegalBench-RAG Dataset Loader

Loads and processes the LegalBench-RAG benchmark dataset for retrieval evaluation.
Based on: "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain"

Dataset available at: https://github.com/zeroentropy-cc/legalbenchrag
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LegalBenchSnippet:
    """Represents a ground truth snippet from LegalBench-RAG."""
    file_path: str
    span: Tuple[int, int]  # (start_char, end_char)
    answer: str

    def __post_init__(self):
        """Validate snippet data."""
        if len(self.span) != 2:
            raise ValueError(f"Span must be (start, end) tuple, got {self.span}")
        if self.span[0] < 0 or self.span[1] < self.span[0]:
            raise ValueError(f"Invalid span: {self.span}")


@dataclass
class LegalBenchQuery:
    """Represents a query from LegalBench-RAG."""
    query: str
    snippets: List[LegalBenchSnippet]
    dataset_source: Optional[str] = None  # ContractNLI, CUAD, MAUD, or PrivacyQA

    def get_document_ids(self) -> List[str]:
        """Get unique document file paths."""
        return list(set(snippet.file_path for snippet in self.snippets))


class LegalBenchRAGLoader:
    """
    Loader for LegalBench-RAG dataset.

    Supports both full dataset (6,858 queries) and mini version (776 queries).
    """

    def __init__(
        self,
        corpus_dir: str,
        queries_file: str,
        use_mini: bool = False
    ):
        """
        Initialize LegalBench-RAG loader.

        Args:
            corpus_dir: Directory containing legal document corpus
            queries_file: Path to queries JSON file
            use_mini: Whether to use LegalBench-RAG-mini (776 queries)
        """
        self.corpus_dir = Path(corpus_dir)
        self.queries_file = Path(queries_file)
        self.use_mini = use_mini

        # Verify paths exist
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
        if not self.queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")

        self.queries: List[LegalBenchQuery] = []
        self.corpus_cache: Dict[str, str] = {}

    def load_queries(self) -> List[LegalBenchQuery]:
        """
        Load queries from JSON file.

        Expected format (from Appendix A of paper):
        {
            "tests": [
                {
                    "query": "Consider the Non-Disclosure Agreement...",
                    "snippets": [
                        {
                            "file_path": "contractnli/CopAcc_NDA.txt",
                            "span": [11461, 11963],
                            "answer": "Any and all proprietary rights..."
                        }
                    ]
                },
                ...
            ]
        }

        Returns:
            List of LegalBenchQuery objects
        """
        with open(self.queries_file, 'r') as f:
            data = json.load(f)

        # Handle different possible JSON structures
        if isinstance(data, dict) and 'tests' in data:
            query_list = data['tests']
        elif isinstance(data, list):
            query_list = data
        else:
            raise ValueError(f"Unexpected JSON format in {self.queries_file}")

        queries = []
        for query_data in query_list:
            snippets = []
            for snippet_data in query_data.get('snippets', []):
                snippet = LegalBenchSnippet(
                    file_path=snippet_data['file_path'],
                    span=tuple(snippet_data['span']),
                    answer=snippet_data.get('answer', '')
                )
                snippets.append(snippet)

            query = LegalBenchQuery(
                query=query_data['query'],
                snippets=snippets,
                dataset_source=self._infer_dataset_source(query_data.get('query', ''))
            )
            queries.append(query)

        # If using mini version, sample uniformly across datasets
        if self.use_mini and len(queries) > 776:
            queries = self._sample_mini_version(queries)

        self.queries = queries
        print(f"Loaded {len(queries)} queries from LegalBench-RAG")
        return queries

    def _infer_dataset_source(self, query: str) -> str:
        """Infer which subdataset this query came from based on content."""
        query_lower = query.lower()

        if 'privacy' in query_lower or 'data collection' in query_lower:
            return 'PrivacyQA'
        elif 'merger' in query_lower or 'acquisition' in query_lower or 'm&a' in query_lower:
            return 'MAUD'
        elif 'contract' in query_lower and 'nda' not in query_lower:
            return 'CUAD'
        elif 'non-disclosure' in query_lower or 'nda' in query_lower:
            return 'ContractNLI'
        else:
            return 'Unknown'

    def _sample_mini_version(self, queries: List[LegalBenchQuery]) -> List[LegalBenchQuery]:
        """
        Sample 776 queries for mini version (194 from each of 4 datasets).

        Args:
            queries: Full list of queries

        Returns:
            Sampled queries for mini version
        """
        # Group by dataset
        by_dataset = {'ContractNLI': [], 'CUAD': [], 'MAUD': [], 'PrivacyQA': []}
        for query in queries:
            source = query.dataset_source
            if source in by_dataset:
                by_dataset[source].append(query)

        # Sample 194 from each
        import random
        random.seed(42)  # For reproducibility

        sampled = []
        for dataset, dataset_queries in by_dataset.items():
            if len(dataset_queries) >= 194:
                sampled.extend(random.sample(dataset_queries, 194))
            else:
                sampled.extend(dataset_queries)

        return sampled

    def load_document(self, file_path: str) -> str:
        """
        Load document text from corpus.

        Args:
            file_path: Relative path within corpus directory

        Returns:
            Document text content
        """
        if file_path in self.corpus_cache:
            return self.corpus_cache[file_path]

        full_path = self.corpus_dir / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Document not found: {full_path}")

        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        self.corpus_cache[file_path] = text
        return text

    def extract_snippet_text(self, snippet: LegalBenchSnippet) -> str:
        """
        Extract the ground truth text from a snippet.

        Args:
            snippet: LegalBenchSnippet object

        Returns:
            Text at the specified character span
        """
        doc_text = self.load_document(snippet.file_path)
        start, end = snippet.span
        return doc_text[start:end]

    def convert_to_retrieval_format(
        self,
        include_snippets: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convert LegalBench-RAG queries to standard retrieval evaluation format.

        Args:
            include_snippets: Whether to include character-level snippet information

        Returns:
            List of queries in format compatible with retrieval_eval.py
            Format:
            [
                {
                    "query": "What are the elements...",
                    "relevant_doc_ids": ["contractnli/file1.txt", ...],
                    "snippets": [  # optional, for snippet-level eval
                        {"file_path": "...", "span": [start, end], "text": "..."}
                    ]
                }
            ]
        """
        converted = []

        for query_obj in self.queries:
            query_dict = {
                'query': query_obj.query,
                'relevant_doc_ids': query_obj.get_document_ids(),
                'dataset_source': query_obj.dataset_source
            }

            if include_snippets:
                query_dict['snippets'] = [
                    {
                        'file_path': s.file_path,
                        'span': list(s.span),
                        'text': s.answer
                    }
                    for s in query_obj.snippets
                ]

            converted.append(query_dict)

        return converted

    def get_corpus_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the LegalBench-RAG corpus.

        Returns:
            Dictionary with corpus statistics
        """
        # Count files in corpus
        all_files = list(self.corpus_dir.rglob('*.txt'))
        total_chars = sum(len(self.load_document(str(f.relative_to(self.corpus_dir))))
                         for f in all_files)

        # Get query statistics
        datasets = {}
        for query in self.queries:
            source = query.dataset_source
            if source not in datasets:
                datasets[source] = 0
            datasets[source] += 1

        return {
            'num_documents': len(all_files),
            'total_characters': total_chars,
            'num_queries': len(self.queries),
            'queries_by_dataset': datasets,
            'version': 'mini' if self.use_mini else 'full'
        }


def load_legalbench_rag(
    corpus_dir: str,
    queries_file: str,
    use_mini: bool = False,
    output_format: str = 'standard'
) -> List[Dict[str, Any]]:
    """
    Convenience function to load LegalBench-RAG dataset.

    Args:
        corpus_dir: Directory containing legal document corpus
        queries_file: Path to queries JSON file
        use_mini: Whether to use mini version (776 queries)
        output_format: 'standard' (for retrieval_eval) or 'raw' (LegalBenchQuery objects)

    Returns:
        List of queries in requested format
    """
    loader = LegalBenchRAGLoader(corpus_dir, queries_file, use_mini)
    loader.load_queries()

    if output_format == 'standard':
        return loader.convert_to_retrieval_format()
    elif output_format == 'raw':
        return loader.queries
    else:
        raise ValueError(f"Unknown output_format: {output_format}")


if __name__ == "__main__":
    # Example usage
    print("LegalBench-RAG Dataset Loader")
    print("=" * 80)
    print("\nThis module loads the LegalBench-RAG benchmark dataset.")
    print("\nDataset structure:")
    print("  - corpus/: Directory with legal documents (.txt files)")
    print("  - queries.json: Query-answer pairs with character spans")
    print("\nDataset statistics (from paper):")
    print("  - Full version: 6,858 queries, 714 documents, 79M+ characters")
    print("  - Mini version: 776 queries (194 per dataset)")
    print("  - 4 subdatasets: ContractNLI, CUAD, MAUD, PrivacyQA")
    print("\nUsage:")
    print("  from src.data.legalbench_loader import load_legalbench_rag")
    print("  queries = load_legalbench_rag(")
    print("      corpus_dir='data/legalbench-rag/corpus',")
    print("      queries_file='data/legalbench-rag/queries.json',")
    print("      use_mini=True")
    print("  )")
    print("\nDownload dataset from:")
    print("  https://github.com/zeroentropy-cc/legalbenchrag")
    print("=" * 80)
