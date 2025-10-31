"""
Embedding Module

Generates dense vector embeddings for text using sentence transformers.
Optimized for legal document retrieval.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


class EmbeddingModel:
    """
    Wrapper for sentence transformer embedding models.

    Supports various embedding models with CPU/GPU inference and batch processing.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name or path
            device: Device for inference ('cpu' or 'cuda')
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings to unit length
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Load model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Model loaded on {device}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array

        Returns:
            Embeddings as numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings

    def encode_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode documents with metadata.

        Args:
            documents: List of document dictionaries
            text_field: Field name containing text to encode
            show_progress_bar: Whether to show progress bar

        Returns:
            Embeddings as numpy array
        """
        texts = [doc[text_field] for doc in documents]
        return self.encode(texts, show_progress_bar=show_progress_bar)

    def encode_batched(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode texts in batches with manual batching control.

        Useful for processing very large datasets with progress tracking.

        Args:
            texts: List of texts to encode
            batch_size: Batch size (uses instance batch_size if None)

        Returns:
            Embeddings as numpy array
        """
        if batch_size is None:
            batch_size = self.batch_size

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str,
    ) -> None:
        """
        Save embeddings to disk.

        Args:
            embeddings: Embeddings array
            output_path: Path to save embeddings (.npy file)
        """
        np.save(output_path, embeddings)
        print(f"Embeddings saved to {output_path}")

    def load_embeddings(self, input_path: str) -> np.ndarray:
        """
        Load embeddings from disk.

        Args:
            input_path: Path to embeddings file (.npy)

        Returns:
            Loaded embeddings array
        """
        embeddings = np.load(input_path)
        print(f"Embeddings loaded from {input_path}")
        print(f"Shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.embedding_dim

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute similarity between embeddings.

        Args:
            embeddings1: First set of embeddings (n, dim)
            embeddings2: Second set of embeddings (m, dim)
            metric: Similarity metric ('cosine' or 'dot')

        Returns:
            Similarity matrix (n, m)
        """
        if metric == "cosine":
            # Normalize if not already normalized
            if not self.normalize_embeddings:
                embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
                embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

            return np.matmul(embeddings1, embeddings2.T)

        elif metric == "dot":
            return np.matmul(embeddings1, embeddings2.T)

        else:
            raise ValueError(f"Unknown metric: {metric}")


def load_embedding_model_from_config(config: Dict[str, Any]) -> EmbeddingModel:
    """
    Load embedding model from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured EmbeddingModel instance
    """
    embedding_config = config.get('embedding', {})

    return EmbeddingModel(
        model_name=embedding_config.get('model_name', 'sentence-transformers/all-mpnet-base-v2'),
        device=embedding_config.get('device', 'cpu'),
        batch_size=embedding_config.get('batch_size', 32),
        normalize_embeddings=embedding_config.get('normalize_embeddings', True),
    )


def load_embedding_model_from_config_file(config_path: str) -> EmbeddingModel:
    """
    Load embedding model from YAML configuration file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configured EmbeddingModel instance
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return load_embedding_model_from_config(config)


if __name__ == "__main__":
    # Example usage
    model = EmbeddingModel(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for testing
        device="cpu",
        batch_size=8,
    )

    # Test encoding
    texts = [
        "The plaintiff must prove negligence by a preponderance of the evidence.",
        "A breach of duty occurs when the defendant fails to meet the standard of care.",
        "Damages must be quantifiable and directly caused by the defendant's actions.",
    ]

    print("\nEncoding test texts...")
    embeddings = model.encode(texts)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0, :5]}")

    # Test similarity
    query = "What are the requirements for proving negligence?"
    query_embedding = model.encode(query)

    similarities = model.similarity(query_embedding.reshape(1, -1), embeddings)
    print(f"\nSimilarities to query:")
    for i, sim in enumerate(similarities[0]):
        print(f"  Text {i+1}: {sim:.4f}")
