"""
Indexing Module

Creates and manages FAISS vector indexes for efficient similarity search.
Optimized for CPU inference with legal document retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import pickle
import os
from pathlib import Path


class VectorIndex:
    """
    FAISS vector index wrapper for document retrieval.

    Supports various index types optimized for CPU usage.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IndexFlatIP",
        metric: str = "inner_product",
        use_gpu: bool = False,
    ):
        """
        Initialize vector index.

        Args:
            embedding_dim: Dimensionality of embeddings
            index_type: Type of FAISS index ('IndexFlatIP', 'IndexFlatL2', 'IndexIVFFlat')
            metric: Distance metric ('inner_product' or 'l2')
            use_gpu: Whether to use GPU (if available)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu
        self.index = None
        self.documents = []  # Store document metadata

        self._create_index()

    def _create_index(self):
        """Create FAISS index based on configuration."""
        if self.index_type == "IndexFlatIP":
            # Flat index with inner product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        elif self.index_type == "IndexFlatL2":
            # Flat index with L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        elif self.index_type == "IndexIVFFlat":
            # Inverted file index (more memory efficient for large datasets)
            # Requires training
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            print("Index moved to GPU")
        else:
            print("Using CPU index")

        print(f"Created {self.index_type} index with dimension {self.embedding_dim}")

    def add(
        self,
        embeddings: np.ndarray,
        documents: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add embeddings to the index.

        Args:
            embeddings: Embeddings array of shape (n, embedding_dim)
            documents: Optional list of document metadata dictionaries
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Train index if needed (for IVF indexes)
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings)
            print("Index trained")

        # Add embeddings
        self.index.add(embeddings)

        # Store documents
        if documents is not None:
            self.documents.extend(documents)
        else:
            # Create dummy documents
            start_id = len(self.documents)
            self.documents.extend([
                {"id": start_id + i}
                for i in range(len(embeddings))
            ])

        print(f"Added {len(embeddings)} embeddings to index")
        print(f"Total documents in index: {self.index.ntotal}")

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            query_embeddings: Query embeddings of shape (n_queries, embedding_dim)
            k: Number of nearest neighbors to retrieve

        Returns:
            Tuple of (distances, indices) arrays
        """
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        # Ensure 2D array
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

    def search_with_documents(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search and return documents with metadata.

        Args:
            query_embeddings: Query embeddings
            k: Number of results per query

        Returns:
            List of result lists, where each result contains document and score
        """
        distances, indices = self.search(query_embeddings, k)

        results = []
        for query_dists, query_indices in zip(distances, indices):
            query_results = []
            for dist, idx in zip(query_dists, query_indices):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(dist)
                    doc['index_id'] = int(idx)
                    query_results.append(doc)

            results.append(query_results)

        return results

    def save(self, index_path: str, documents_path: Optional[str] = None):
        """
        Save index and documents to disk.

        Args:
            index_path: Path to save FAISS index (.faiss file)
            documents_path: Path to save documents (.pkl file)
        """
        # Save FAISS index
        if self.use_gpu:
            # Move to CPU before saving
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, index_path)
        else:
            faiss.write_index(self.index, index_path)

        print(f"Index saved to {index_path}")

        # Save documents
        if documents_path is None:
            documents_path = index_path.replace('.faiss', '_documents.pkl')

        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"Documents saved to {documents_path}")

    def load(self, index_path: str, documents_path: Optional[str] = None):
        """
        Load index and documents from disk.

        Args:
            index_path: Path to FAISS index file
            documents_path: Path to documents pickle file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

        print(f"Index loaded from {index_path}")
        print(f"Total documents in index: {self.index.ntotal}")

        # Load documents
        if documents_path is None:
            documents_path = index_path.replace('.faiss', '_documents.pkl')

        if os.path.exists(documents_path):
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Documents loaded from {documents_path}")
        else:
            print(f"Warning: Documents file not found at {documents_path}")

    def get_num_documents(self) -> int:
        """Get the number of documents in the index."""
        return self.index.ntotal if self.index else 0


def create_index_from_config(config: Dict[str, Any], embedding_dim: int) -> VectorIndex:
    """
    Create vector index from configuration.

    Args:
        config: Configuration dictionary
        embedding_dim: Embedding dimensionality

    Returns:
        Configured VectorIndex instance
    """
    indexing_config = config.get('indexing', {})

    return VectorIndex(
        embedding_dim=embedding_dim,
        index_type=indexing_config.get('index_type', 'IndexFlatIP'),
        metric=indexing_config.get('metric', 'inner_product'),
        use_gpu=indexing_config.get('use_gpu', False),
    )


def build_index_from_documents(
    documents: List[Dict[str, Any]],
    embeddings: np.ndarray,
    output_dir: str,
    config: Dict[str, Any],
) -> VectorIndex:
    """
    Build and save vector index from documents and embeddings.

    Args:
        documents: List of document dictionaries
        embeddings: Document embeddings array
        output_dir: Directory to save index
        config: Configuration dictionary

    Returns:
        Built VectorIndex instance
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create index
    embedding_dim = embeddings.shape[1]
    index = create_index_from_config(config, embedding_dim)

    # Add documents
    index.add(embeddings, documents)

    # Save index
    index_path = os.path.join(output_dir, "faiss_index.faiss")
    documents_path = os.path.join(output_dir, "documents.pkl")
    index.save(index_path, documents_path)

    return index


if __name__ == "__main__":
    # Example usage
    print("Creating test index...")

    # Create dummy embeddings and documents
    embedding_dim = 384
    n_docs = 100

    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    documents = [
        {
            "id": i,
            "text": f"This is test document {i}",
            "source": "test_corpus",
        }
        for i in range(n_docs)
    ]

    # Create index
    index = VectorIndex(embedding_dim=embedding_dim, index_type="IndexFlatIP")
    index.add(embeddings, documents)

    # Test search
    query_embedding = np.random.randn(1, embedding_dim).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    results = index.search_with_documents(query_embedding, k=5)

    print("\nSearch results:")
    for i, doc in enumerate(results[0]):
        print(f"{i+1}. Score: {doc['score']:.4f}, Text: {doc['text']}")

    # Test save/load
    print("\nTesting save/load...")
    index.save("test_index.faiss")

    index2 = VectorIndex(embedding_dim=embedding_dim)
    index2.load("test_index.faiss")

    print(f"Loaded index has {index2.get_num_documents()} documents")
