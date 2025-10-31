"""
Intent Detection for Adaptive Retrieval

Classifies query intent to enable intent-aware retrieval strategies.
Different intents require different retrieval approaches:
- Factual: High precision, specific document retrieval
- Exploratory: Diverse results, broader coverage
- Comparative: Contrasting documents for analysis
- Procedural: Step-by-step instructions
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from enum import Enum
from transformers import AutoTokenizer, AutoModel
import numpy as np


class QueryIntent(Enum):
    """Query intent types for legal domain."""
    FACTUAL = "factual"  # Seeking specific facts or definitions
    EXPLORATORY = "exploratory"  # Broad exploration of a topic
    COMPARATIVE = "comparative"  # Comparing cases, laws, or concepts
    PROCEDURAL = "procedural"  # How-to or process-oriented queries
    UNKNOWN = "unknown"  # Cannot determine intent


class IntentDetector:
    """
    Detects query intent using pattern matching and optional ML classification.

    Supports both rule-based and model-based intent detection.

    Args:
        method: Detection method ('rules', 'model', 'hybrid')
        model_name: Hugging Face model for classification (if method includes 'model')
        device: Device for computation
    """

    def __init__(
        self,
        method: str = 'rules',
        model_name: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.method = method
        self.device = device

        # Rule-based patterns
        self.patterns = {
            QueryIntent.FACTUAL: [
                "what is", "define", "definition of", "meaning of",
                "who is", "when did", "where is", "which",
                "how many", "how much", "is it true"
            ],
            QueryIntent.EXPLORATORY: [
                "tell me about", "explain", "describe", "overview",
                "discuss", "explore", "summarize", "what are all",
                "give me information"
            ],
            QueryIntent.COMPARATIVE: [
                "compare", "difference between", "versus", "vs",
                "contrast", "how does X differ from", "similar to",
                "rather than", "instead of"
            ],
            QueryIntent.PROCEDURAL: [
                "how to", "how do i", "how can i", "steps to",
                "procedure for", "process of", "guide to",
                "instructions", "tutorial"
            ]
        }

        # Load model if needed
        if method in ['model', 'hybrid'] and model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()

            # Classification head (will be trained)
            self.classifier = nn.Linear(
                self.model.config.hidden_size,
                len(QueryIntent) - 1  # Exclude UNKNOWN
            ).to(device)
        else:
            self.model = None
            self.classifier = None

    def detect_intent_rules(self, query: str) -> QueryIntent:
        """
        Detect intent using rule-based pattern matching.

        Args:
            query: Query string

        Returns:
            Detected QueryIntent
        """
        query_lower = query.lower()

        # Score each intent based on pattern matches
        scores = {intent: 0 for intent in QueryIntent if intent != QueryIntent.UNKNOWN}

        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    scores[intent] += 1

        # Return intent with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return QueryIntent.UNKNOWN

    def detect_intent_model(self, query: str) -> QueryIntent:
        """
        Detect intent using ML model.

        Args:
            query: Query string

        Returns:
            Detected QueryIntent
        """
        if self.model is None or self.classifier is None:
            raise ValueError("Model not loaded. Initialize with method='model' or 'hybrid'")

        # Tokenize query
        inputs = self.tokenizer(
            query,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]

            # Classify
            logits = self.classifier(embedding)
            predicted_idx = torch.argmax(logits, dim=1).item()

        # Map index to intent
        intents = [intent for intent in QueryIntent if intent != QueryIntent.UNKNOWN]
        return intents[predicted_idx]

    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect query intent using configured method.

        Args:
            query: Query string

        Returns:
            Detected QueryIntent
        """
        if self.method == 'rules':
            return self.detect_intent_rules(query)
        elif self.method == 'model':
            return self.detect_intent_model(query)
        elif self.method == 'hybrid':
            # Use rules first, fallback to model if uncertain
            rule_intent = self.detect_intent_rules(query)
            if rule_intent == QueryIntent.UNKNOWN:
                return self.detect_intent_model(query)
            return rule_intent
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def detect_batch(self, queries: List[str]) -> List[QueryIntent]:
        """
        Detect intent for a batch of queries.

        Args:
            queries: List of query strings

        Returns:
            List of detected QueryIntent
        """
        return [self.detect_intent(query) for query in queries]

    def get_confidence(self, query: str) -> Dict[QueryIntent, float]:
        """
        Get confidence scores for each intent.

        Args:
            query: Query string

        Returns:
            Dictionary mapping QueryIntent to confidence score
        """
        if self.method == 'rules':
            # Rule-based confidence based on pattern matches
            query_lower = query.lower()
            scores = {intent: 0 for intent in QueryIntent if intent != QueryIntent.UNKNOWN}

            for intent, patterns in self.patterns.items():
                for pattern in patterns:
                    if pattern in query_lower:
                        scores[intent] += 1

            # Normalize to probabilities
            total = sum(scores.values())
            if total > 0:
                return {intent: score / total for intent, score in scores.items()}
            else:
                return {intent: 1.0 / len(scores) for intent in scores}

        elif self.method in ['model', 'hybrid']:
            if self.model is None:
                return self.get_confidence_rules(query)

            # Model-based confidence
            inputs = self.tokenizer(
                query,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(embedding)
                probs = torch.softmax(logits, dim=1)[0]

            intents = [intent for intent in QueryIntent if intent != QueryIntent.UNKNOWN]
            return {intent: float(probs[i]) for i, intent in enumerate(intents)}

    def train_classifier(
        self,
        train_queries: List[str],
        train_labels: List[QueryIntent],
        val_queries: Optional[List[str]] = None,
        val_labels: Optional[List[QueryIntent]] = None,
        epochs: int = 3,
        lr: float = 1e-4,
        batch_size: int = 16
    ):
        """
        Train the intent classification model.

        Args:
            train_queries: Training queries
            train_labels: Training labels
            val_queries: Validation queries (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
        """
        if self.model is None or self.classifier is None:
            raise ValueError("Model not loaded. Initialize with method='model' or 'hybrid'")

        # Prepare labels
        intent_to_idx = {
            intent: i for i, intent in enumerate(QueryIntent) if intent != QueryIntent.UNKNOWN
        }
        train_label_ids = [intent_to_idx[label] for label in train_labels]

        # Create optimizer
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            correct = 0
            total = 0

            # Mini-batch training
            for i in range(0, len(train_queries), batch_size):
                batch_queries = train_queries[i:i + batch_size]
                batch_labels = train_label_ids[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_queries,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]

                logits = self.classifier(embeddings)
                labels = torch.tensor(batch_labels).to(self.device)

                # Compute loss
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += len(batch_labels)

            # Epoch metrics
            avg_loss = total_loss / (len(train_queries) // batch_size + 1)
            accuracy = correct / total
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Validation
            if val_queries and val_labels:
                val_accuracy = self.evaluate(val_queries, val_labels)
                print(f"Validation Accuracy: {val_accuracy:.4f}")

    def evaluate(
        self,
        queries: List[str],
        labels: List[QueryIntent]
    ) -> float:
        """
        Evaluate classifier accuracy.

        Args:
            queries: Evaluation queries
            labels: Ground truth labels

        Returns:
            Accuracy score
        """
        predictions = self.detect_batch(queries)
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        return correct / len(labels)


def get_retrieval_strategy(intent: QueryIntent) -> Dict[str, any]:
    """
    Map query intent to retrieval strategy parameters.

    Args:
        intent: Detected query intent

    Returns:
        Dictionary of retrieval strategy parameters
    """
    strategies = {
        QueryIntent.FACTUAL: {
            'top_k': 3,
            'diversity': 0.0,
            'rerank_method': 'relevance',
            'description': 'High precision, focused retrieval'
        },
        QueryIntent.EXPLORATORY: {
            'top_k': 10,
            'diversity': 0.7,
            'rerank_method': 'diversity',
            'description': 'Broad coverage, diverse results'
        },
        QueryIntent.COMPARATIVE: {
            'top_k': 6,
            'diversity': 0.5,
            'rerank_method': 'contrast',
            'description': 'Contrasting documents for comparison'
        },
        QueryIntent.PROCEDURAL: {
            'top_k': 5,
            'diversity': 0.3,
            'rerank_method': 'sequential',
            'description': 'Step-by-step relevant documents'
        },
        QueryIntent.UNKNOWN: {
            'top_k': 5,
            'diversity': 0.3,
            'rerank_method': 'relevance',
            'description': 'Default balanced retrieval'
        }
    }

    return strategies.get(intent, strategies[QueryIntent.UNKNOWN])


def analyze_query_characteristics(query: str) -> Dict[str, any]:
    """
    Analyze query characteristics that influence retrieval.

    Args:
        query: Query string

    Returns:
        Dictionary of query characteristics
    """
    words = query.split()
    chars = len(query)

    # Check for question words
    question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
    has_question_word = any(word in query.lower() for word in question_words)

    # Check for legal-specific terms
    legal_terms = ['law', 'case', 'statute', 'regulation', 'liability', 'contract']
    has_legal_term = any(term in query.lower() for term in legal_terms)

    return {
        'length_words': len(words),
        'length_chars': chars,
        'has_question_word': has_question_word,
        'has_legal_term': has_legal_term,
        'complexity': min(1.0, len(words) / 20.0),  # Normalized complexity
        'specificity': 1.0 - (0.1 * query.lower().count('all'))  # Rough estimate
    }
