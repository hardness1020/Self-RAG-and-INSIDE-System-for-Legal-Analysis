"""
Unified Hallucination Detection Interface

Combines EigenScore-based detection and feature clipping to provide
comprehensive hallucination detection for Self-RAG systems.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedModel

from .eigenscore import EigenScore, EigenScoreAggregator, AdaptiveEigenScore
from .feature_clipping import FeatureClipper, ClippedGenerator, detect_overconfident_hallucination
from .internal_states import InternalStatesExtractor
from .intent_detector import IntentDetector, QueryIntent


class HallucinationDetector:
    """
    Unified interface for hallucination detection using INSIDE methods.

    Combines:
    1. EigenScore: Semantic consistency via internal embeddings
    2. Feature Clipping: Overconfident hallucination detection
    3. Intent-aware thresholds: Adaptive detection based on query type

    Args:
        model: Language model
        tokenizer: Tokenizer
        target_layers: Layers to extract internal states from
        eigenscore_threshold: Base threshold for EigenScore
        clipping_percentile: Percentile for feature clipping
        use_adaptive_threshold: Whether to use intent-aware adaptive thresholds
        device: Device for computation
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        target_layers: Optional[List[int]] = None,
        eigenscore_threshold: float = 5.0,
        clipping_percentile: float = 95.0,
        use_adaptive_threshold: bool = True,
        device: str = 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize components
        self.states_extractor = InternalStatesExtractor(
            model=model,
            target_layers=target_layers,
            device=device
        )

        self.eigenscore_calculator = EigenScore(normalize=True)

        self.eigenscore_aggregator = EigenScoreAggregator(
            aggregation_method='mean',
            threshold=eigenscore_threshold
        )

        self.feature_clipper = FeatureClipper(
            percentile=clipping_percentile,
            symmetric=True,
            layer_specific=True
        )

        self.clipped_generator = ClippedGenerator(
            model=model,
            tokenizer=tokenizer,
            clipper=self.feature_clipper,
            target_layers=self.states_extractor.target_layers,
            device=device
        )

        self.adaptive_eigenscore = AdaptiveEigenScore(
            base_threshold=eigenscore_threshold
        ) if use_adaptive_threshold else None

        self.intent_detector = IntentDetector(method='rules')

    def detect_from_generations(
        self,
        query: str,
        generations: List[str],
        use_clipping: bool = True,
        return_details: bool = False
    ) -> Dict[str, any]:
        """
        Detect hallucinations from multiple generations of a query.

        Args:
            query: Input query
            generations: List of generated responses
            use_clipping: Whether to use feature clipping analysis
            return_details: Whether to return detailed information

        Returns:
            Dictionary with detection results
        """
        # Detect query intent
        intent = self.intent_detector.detect_intent(query)

        # Extract embeddings from generations
        embeddings = self.states_extractor.extract_from_generations(
            generations=generations,
            tokenizer=self.tokenizer,
            split_sentences=True
        )

        # Get embeddings from first target layer
        layer_idx = list(embeddings.keys())[0]
        emb = embeddings[layer_idx]

        # Compute EigenScore
        eigenscore_details = self.eigenscore_calculator.compute(emb, return_details=True)
        eigenscore = eigenscore_details['score']

        # Adaptive threshold based on query
        if self.adaptive_eigenscore:
            query_tokens = self.tokenizer.encode(query)
            is_hallucination, eigenscore, threshold = self.adaptive_eigenscore.detect_hallucination_adaptive(
                eigenscore=eigenscore,
                query_length=len(query_tokens),
                model_confidence=None  # TODO: Add confidence from generation
            )
        else:
            threshold = self.eigenscore_aggregator.threshold
            is_hallucination = eigenscore > threshold if threshold else False

        # Feature clipping analysis (if requested)
        clipping_results = {}
        if use_clipping and len(generations) > 0:
            # Compare with clipped generations
            clipped_gens = self.clipped_generator.generate_with_clipping(
                input_text=query,
                num_return_sequences=min(3, len(generations))
            )

            is_overconfident, sensitivity = detect_overconfident_hallucination(
                clipped_texts=clipped_gens,
                unclipped_texts=generations[:len(clipped_gens)]
            )

            clipping_results = {
                'is_overconfident': is_overconfident,
                'clipping_sensitivity': sensitivity,
                'clipped_generations': clipped_gens if return_details else None
            }

        # Combined detection
        combined_hallucination = is_hallucination or clipping_results.get('is_overconfident', False)

        result = {
            'is_hallucination': combined_hallucination,
            'eigenscore': eigenscore,
            'eigenscore_threshold': threshold,
            'query_intent': intent.value,
            'num_generations': len(generations)
        }

        if use_clipping:
            result.update(clipping_results)

        if return_details:
            result['eigenscore_details'] = eigenscore_details
            result['generations'] = generations

        return result

    def detect_from_text(
        self,
        query: str,
        generated_text: str,
        num_samples: int = 3,
        return_details: bool = False
    ) -> Dict[str, any]:
        """
        Detect hallucination from a single generated text.

        Generates multiple samples to compute EigenScore.

        Args:
            query: Input query
            generated_text: Generated response
            num_samples: Number of additional samples to generate
            return_details: Whether to return detailed information

        Returns:
            Dictionary with detection results
        """
        # Generate multiple samples for robust detection
        inputs = self.tokenizer(query, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=0.8
            )

        additional_generations = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        # Include original text
        all_generations = [generated_text] + additional_generations

        # Run detection
        return self.detect_from_generations(
            query=query,
            generations=all_generations,
            return_details=return_details
        )

    def batch_detect(
        self,
        queries: List[str],
        generations_list: List[List[str]],
        use_clipping: bool = False  # Clipping is expensive, off by default for batches
    ) -> List[Dict[str, any]]:
        """
        Detect hallucinations for a batch of query-generation pairs.

        Args:
            queries: List of queries
            generations_list: List of generation lists (one per query)
            use_clipping: Whether to use feature clipping

        Returns:
            List of detection results
        """
        results = []

        for query, generations in zip(queries, generations_list):
            result = self.detect_from_generations(
                query=query,
                generations=generations,
                use_clipping=use_clipping,
                return_details=False
            )
            results.append(result)

        return results

    def calibrate(
        self,
        factual_examples: List[Tuple[str, List[str]]],
        hallucinated_examples: List[Tuple[str, List[str]]],
        percentile: float = 10.0
    ):
        """
        Calibrate detection thresholds using labeled examples.

        Args:
            factual_examples: List of (query, generations) for factual responses
            hallucinated_examples: List of (query, generations) for hallucinated responses
            percentile: Percentile for threshold calibration
        """
        factual_embeddings = []
        hallucinated_embeddings = []

        # Extract embeddings from factual examples
        for query, generations in factual_examples:
            embeddings = self.states_extractor.extract_from_generations(
                generations=generations,
                tokenizer=self.tokenizer,
                split_sentences=True
            )
            layer_idx = list(embeddings.keys())[0]
            factual_embeddings.append(embeddings[layer_idx])

        # Extract embeddings from hallucinated examples
        for query, generations in hallucinated_examples:
            embeddings = self.states_extractor.extract_from_generations(
                generations=generations,
                tokenizer=self.tokenizer,
                split_sentences=True
            )
            layer_idx = list(embeddings.keys())[0]
            hallucinated_embeddings.append(embeddings[layer_idx])

        # Calibrate threshold
        from .eigenscore import calibrate_threshold

        threshold = calibrate_threshold(
            hallucinated_embeddings=hallucinated_embeddings,
            factual_embeddings=factual_embeddings,
            percentile=percentile
        )

        # Update threshold
        self.eigenscore_aggregator.threshold = threshold
        if self.adaptive_eigenscore:
            self.adaptive_eigenscore.base_threshold = threshold

        print(f"Calibrated EigenScore threshold: {threshold:.2f}")

        return threshold

    def calibrate_clipper(
        self,
        calibration_texts: List[str]
    ):
        """
        Calibrate feature clipper on example texts.

        Args:
            calibration_texts: List of texts for calibration
        """
        self.feature_clipper.calibrate_on_data(
            model=self.model,
            calibration_texts=calibration_texts,
            tokenizer=self.tokenizer,
            target_layers=self.states_extractor.target_layers,
            device=self.device
        )

    def get_statistics(
        self,
        queries: List[str],
        generations_list: List[List[str]]
    ) -> Dict[str, any]:
        """
        Compute statistics over a dataset.

        Args:
            queries: List of queries
            generations_list: List of generation lists

        Returns:
            Dictionary with statistics
        """
        results = self.batch_detect(queries, generations_list)

        eigenscores = [r['eigenscore'] for r in results]
        hallucination_flags = [r['is_hallucination'] for r in results]

        # Intent distribution
        intents = [r['query_intent'] for r in results]
        intent_counts = {intent: intents.count(intent) for intent in set(intents)}

        return {
            'total_queries': len(queries),
            'hallucination_rate': sum(hallucination_flags) / len(hallucination_flags),
            'mean_eigenscore': np.mean(eigenscores),
            'std_eigenscore': np.std(eigenscores),
            'min_eigenscore': np.min(eigenscores),
            'max_eigenscore': np.max(eigenscores),
            'intent_distribution': intent_counts
        }


def create_detector(
    model: PreTrainedModel,
    tokenizer,
    config: Optional[Dict[str, any]] = None,
    device: str = 'cpu'
) -> HallucinationDetector:
    """
    Factory function to create a HallucinationDetector with configuration.

    Args:
        model: Language model
        tokenizer: Tokenizer
        config: Configuration dictionary
        device: Device for computation

    Returns:
        Configured HallucinationDetector
    """
    if config is None:
        config = {}

    return HallucinationDetector(
        model=model,
        tokenizer=tokenizer,
        target_layers=config.get('target_layers', None),
        eigenscore_threshold=config.get('eigenscore_threshold', 5.0),
        clipping_percentile=config.get('clipping_percentile', 95.0),
        use_adaptive_threshold=config.get('use_adaptive_threshold', True),
        device=device
    )
