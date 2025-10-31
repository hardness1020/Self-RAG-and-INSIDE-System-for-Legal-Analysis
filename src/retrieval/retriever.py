"""
Retriever Module

High-level interface for the complete retrieval pipeline.
Combines chunking, embedding, and FAISS indexing for legal document retrieval.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import yaml
from pathlib import Path

from src.retrieval.chunking import DocumentChunker, load_chunker_from_config
from src.retrieval.embedding import EmbeddingModel, load_embedding_model_from_config
from src.retrieval.indexing import VectorIndex, create_index_from_config


class LegalRetriever:
    """
    Complete retrieval pipeline for legal documents.

    Integrates document chunking, embedding generation, and vector search.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: Optional[VectorIndex] = None,
        chunker: Optional[DocumentChunker] = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ):
        """
        Initialize retriever.

        Args:
            embedding_model: Embedding model for encoding text
            vector_index: Vector index for similarity search (optional)
            chunker: Document chunker (optional)
            top_k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold
        """
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.chunker = chunker
        self.top_k = top_k
        self.min_similarity = min_similarity

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        chunk_documents: bool = True,
    ):
        """
        Index documents for retrieval.

        Args:
            documents: List of document dictionaries with text
            text_field: Field containing document text
            chunk_documents: Whether to chunk documents before indexing
        """
        # Chunk documents if requested
        if chunk_documents and self.chunker is not None:
            print(f"Chunking {len(documents)} documents...")
            chunks = self.chunker.chunk_documents(documents)
            print(f"Created {len(chunks)} chunks")
        else:
            chunks = documents

        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} text segments...")
        texts = [chunk[text_field] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Create or update index
        if self.vector_index is None:
            embedding_dim = self.embedding_model.get_embedding_dim()
            self.vector_index = VectorIndex(
                embedding_dim=embedding_dim,
                index_type="IndexFlatIP",
            )

        # Add to index
        self.vector_index.add(embeddings, chunks)

        print(f"Indexing complete. Total documents in index: {self.vector_index.get_num_documents()}")

    def retrieve(
        self,
        query: Union[str, List[str]],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        return_embeddings: bool = False,
    ) -> Union[List[List[Dict[str, Any]]], tuple]:
        """
        Retrieve relevant documents for query.

        Args:
            query: Query string or list of query strings
            top_k: Number of documents to retrieve (overrides instance setting)
            min_similarity: Minimum similarity threshold (overrides instance setting)
            return_embeddings: Whether to return query embeddings

        Returns:
            List of retrieved documents (one list per query)
            If return_embeddings=True, returns (results, query_embeddings)
        """
        if self.vector_index is None:
            raise ValueError("No index available. Call index_documents() first.")

        if top_k is None:
            top_k = self.top_k

        if min_similarity is None:
            min_similarity = self.min_similarity

        # Handle single query
        if isinstance(query, str):
            query = [query]
            single_query = True
        else:
            single_query = False

        # Encode queries
        query_embeddings = self.embedding_model.encode(query, show_progress_bar=False)

        # Search index
        results = self.vector_index.search_with_documents(query_embeddings, k=top_k)

        # Filter by minimum similarity
        filtered_results = []
        for query_results in results:
            filtered = [
                doc for doc in query_results
                if doc['score'] >= min_similarity
            ]
            filtered_results.append(filtered)

        # Return single result list for single query
        if single_query:
            filtered_results = filtered_results[0]

        if return_embeddings:
            return filtered_results, query_embeddings
        else:
            return filtered_results

    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with optional reranking (placeholder for future implementation).

        Args:
            query: Query string
            top_k: Final number of documents to return
            retrieve_k: Number of documents to retrieve before reranking

        Returns:
            List of retrieved and reranked documents
        """
        if retrieve_k is None:
            retrieve_k = top_k * 2

        # Initial retrieval
        results = self.retrieve(query, top_k=retrieve_k)

        # TODO: Implement reranking logic here if needed
        # For now, just return top_k results

        return results[:top_k]

    def save_index(self, output_dir: str):
        """
        Save vector index to disk.

        Args:
            output_dir: Directory to save index files
        """
        if self.vector_index is None:
            raise ValueError("No index to save")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        index_path = Path(output_dir) / "faiss_index.faiss"
        documents_path = Path(output_dir) / "documents.pkl"

        self.vector_index.save(str(index_path), str(documents_path))

    def load_index(self, index_dir: str):
        """
        Load vector index from disk.

        Args:
            index_dir: Directory containing index files
        """
        index_path = Path(index_dir) / "faiss_index.faiss"
        documents_path = Path(index_dir) / "documents.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if self.vector_index is None:
            embedding_dim = self.embedding_model.get_embedding_dim()
            self.vector_index = VectorIndex(embedding_dim=embedding_dim)

        self.vector_index.load(str(index_path), str(documents_path))

    def get_num_documents(self) -> int:
        """Get number of documents in index."""
        if self.vector_index is None:
            return 0
        return self.vector_index.get_num_documents()


def load_retriever_from_config(config_path: str) -> LegalRetriever:
    """
    Load retriever from configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured LegalRetriever instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load embedding model
    embedding_model = load_embedding_model_from_config(config)

    # Load chunker
    chunker = None
    if 'chunking' in config:
        chunking_config = config['chunking']
        chunker = DocumentChunker(chunking_config)

    # Get retrieval parameters
    retrieval_config = config.get('retrieval', {})
    top_k = retrieval_config.get('top_k', 5)
    min_similarity = retrieval_config.get('min_similarity', 0.0)

    # Create retriever
    retriever = LegalRetriever(
        embedding_model=embedding_model,
        vector_index=None,  # Will be created when indexing or loaded
        chunker=chunker,
        top_k=top_k,
        min_similarity=min_similarity,
    )

    return retriever


if __name__ == "__main__":
    # Example usage
    print("Creating test retriever...")

    # Create embedding model
    embedding_model = EmbeddingModel(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=8,
    )

    # Create chunker
    chunker_config = {
        'chunk_size': 256,
        'chunk_overlap': 30,
    }
    chunker = DocumentChunker(chunker_config)

    # Create retriever
    retriever = LegalRetriever(
        embedding_model=embedding_model,
        chunker=chunker,
        top_k=3,
    )

    # Test documents
    documents = [
        {
            "text": """
            To establish negligence, the plaintiff must prove four elements:
            (1) duty of care, (2) breach of that duty, (3) causation, and (4) damages.
            Each element must be proven by a preponderance of the evidence.
            """,
            "source": "negligence_law.txt",
            "title": "Elements of Negligence",
        },
        {
            "text": """
            A breach of duty occurs when the defendant fails to exercise reasonable care.
            The standard is what a reasonable person would do in similar circumstances.
            This is an objective standard, not based on the defendant's subjective beliefs.
            """,
            "source": "negligence_law.txt",
            "title": "Breach of Duty",
        },
        {
            "text": """
            Damages in negligence cases must be actual and quantifiable.
            Nominal damages are not sufficient. The plaintiff must show real harm
            that can be compensated through monetary awards.
            """,
            "source": "damages.txt",
            "title": "Damages Requirements",
        },
    ]

    # Index documents
    print("\nIndexing documents...")
    retriever.index_documents(documents)

    # Test retrieval
    query = "What are the requirements for proving negligence?"
    print(f"\nQuery: {query}")

    results = retriever.retrieve(query)

    print(f"\nRetrieved {len(results)} documents:")
    for i, doc in enumerate(results):
        print(f"\n{i+1}. Score: {doc['score']:.4f}")
        print(f"   Title: {doc.get('title', 'N/A')}")
        print(f"   Text: {doc['text'][:100]}...")
