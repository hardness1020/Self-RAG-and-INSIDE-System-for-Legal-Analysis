"""
INSIDE-Enhanced Generator Module

Extends the Self-RAG generator with INSIDE capabilities:
1. Internal state extraction during generation
2. EigenScore-based hallucination detection
3. Intent token generation
4. Combined Self-RAG + INSIDE scoring

This module wraps the base SelfRAGGenerator and adds INSIDE functionality.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from .generator import SelfRAGGenerator
from .reflection_tokens import INTENTToken, ReflectionAnnotation
from ..inside.hallucination_detector import HallucinationDetector, create_detector
from ..inside.intent_detector import IntentDetector, QueryIntent


class INSIDEGenerator:
    """
    INSIDE-enhanced Self-RAG generator.

    Wraps SelfRAGGenerator and adds:
    - Intent detection and INTENT token generation
    - Internal state extraction during generation
    - EigenScore computation for hallucination detection
    - Combined scoring (reflection tokens + EigenScore)

    Args:
        base_generator: Base SelfRAGGenerator instance
        inside_config: INSIDE configuration dictionary
        device: Device for computation
    """

    def __init__(
        self,
        base_generator: SelfRAGGenerator,
        inside_config: Optional[Dict[str, Any]] = None,
        device: str = 'cpu'
    ):
        self.base_generator = base_generator
        self.device = device

        # Default INSIDE config
        if inside_config is None:
            inside_config = {
                'enabled': True,
                'extract_internal_states': True,
                'target_layers': [16],
                'eigenscore': {
                    'enabled': True,
                    'threshold': 5.0,
                    'use_adaptive_threshold': True
                },
                'combined_scoring': {
                    'use_eigenscore': True,
                    'eigenscore_weight': 0.3,
                    'reflection_weight': 0.7
                }
            }
        self.inside_config = inside_config

        # Initialize INSIDE components
        if self.inside_config['enabled']:
            # Intent detector
            self.intent_detector = IntentDetector(method='rules')

            # Hallucination detector
            if base_generator.model and base_generator.tokenizer:
                self.hallucination_detector = create_detector(
                    model=base_generator.model,
                    tokenizer=base_generator.tokenizer,
                    config={
                        'target_layers': inside_config.get('target_layers', [16]),
                        'eigenscore_threshold': inside_config['eigenscore']['threshold'],
                        'use_adaptive_threshold': inside_config['eigenscore']['use_adaptive_threshold']
                    },
                    device=device
                )
            else:
                self.hallucination_detector = None
        else:
            self.intent_detector = None
            self.hallucination_detector = None

    def generate_with_inside(
        self,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_samples: int = 1,
        detect_hallucination: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response with INSIDE enhancements.

        Args:
            query: Input query
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_samples: Number of samples for robust hallucination detection
            detect_hallucination: Whether to run hallucination detection

        Returns:
            Dictionary with:
            - response: Generated text
            - intent: Detected query intent
            - reflection_annotation: Extracted reflection tokens
            - hallucination_result: Hallucination detection result (if enabled)
            - eigenscore: EigenScore value (if computed)
            - combined_score: Combined Self-RAG + INSIDE score
        """
        # 1. Detect query intent
        intent = self.intent_detector.detect_intent(query) if self.intent_detector else QueryIntent.UNKNOWN

        # 2. Generate response(s) with base generator
        if num_samples == 1:
            response = self.base_generator.generate(
                prompt=query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            responses = [response]
        else:
            # Generate multiple samples for robust detection
            responses = []
            for _ in range(num_samples):
                response = self.base_generator.generate(
                    prompt=query,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
                responses.append(response)

        # 3. Extract reflection tokens from first response
        from .reflection_tokens import ReflectionTokenizer
        reflection_annotation = ReflectionTokenizer.extract_tokens_from_text(responses[0])

        # Add intent token
        reflection_annotation.intent = self._map_intent_to_token(intent)

        # 4. Hallucination detection (if enabled)
        hallucination_result = None
        eigenscore = None

        if detect_hallucination and self.hallucination_detector:
            hallucination_result = self.hallucination_detector.detect_from_generations(
                query=query,
                generations=responses,
                use_clipping=False,  # Disable clipping by default (expensive)
                return_details=True
            )
            eigenscore = hallucination_result.get('eigenscore')

        # 5. Combined scoring
        combined_score = self._compute_combined_score(
            reflection_annotation=reflection_annotation,
            eigenscore=eigenscore
        )

        # 6. Compile result
        result = {
            'response': responses[0],
            'all_responses': responses if num_samples > 1 else None,
            'intent': intent.value,
            'intent_token': reflection_annotation.intent.value if reflection_annotation.intent else None,
            'reflection_annotation': reflection_annotation.to_dict(),
            'hallucination_result': hallucination_result,
            'eigenscore': eigenscore,
            'combined_score': combined_score,
            'num_samples': num_samples
        }

        return result

    def generate_batch_with_inside(
        self,
        queries: List[str],
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple queries with INSIDE.

        Args:
            queries: List of query strings
            **generation_kwargs: Generation parameters

        Returns:
            List of result dictionaries
        """
        results = []
        for query in queries:
            result = self.generate_with_inside(query, **generation_kwargs)
            results.append(result)
        return results

    def _map_intent_to_token(self, intent: QueryIntent) -> INTENTToken:
        """Map QueryIntent enum to INTENTToken enum."""
        mapping = {
            QueryIntent.FACTUAL: INTENTToken.FACTUAL,
            QueryIntent.EXPLORATORY: INTENTToken.EXPLORATORY,
            QueryIntent.COMPARATIVE: INTENTToken.COMPARATIVE,
            QueryIntent.PROCEDURAL: INTENTToken.PROCEDURAL,
            QueryIntent.UNKNOWN: INTENTToken.UNKNOWN,
        }
        return mapping.get(intent, INTENTToken.UNKNOWN)

    def _compute_combined_score(
        self,
        reflection_annotation: ReflectionAnnotation,
        eigenscore: Optional[float]
    ) -> float:
        """
        Compute combined score from reflection tokens and EigenScore.

        Args:
            reflection_annotation: Reflection annotation with tokens
            eigenscore: EigenScore value (if computed)

        Returns:
            Combined score (higher = better quality)
        """
        # Get weights from config
        eigenscore_weight = self.inside_config['combined_scoring']['eigenscore_weight']
        reflection_weight = self.inside_config['combined_scoring']['reflection_weight']

        # Compute reflection score
        reflection_score = 0.0
        num_tokens = 0

        # ISSUP score (most important for hallucination)
        if reflection_annotation.issup:
            from .reflection_tokens import ISSUPToken
            if reflection_annotation.issup == ISSUPToken.FULLY_SUPPORTED:
                reflection_score += 1.0
            elif reflection_annotation.issup == ISSUPToken.PARTIALLY_SUPPORTED:
                reflection_score += 0.5
            else:  # NO_SUPPORT
                reflection_score += 0.0
            num_tokens += 1

        # ISUSE score
        if reflection_annotation.isuse:
            from .reflection_tokens import ISUSEToken
            utility_score = ISUSEToken.get_score(reflection_annotation.isuse) / 5.0
            reflection_score += utility_score
            num_tokens += 1

        # ISREL score
        if reflection_annotation.isrel:
            from .reflection_tokens import ISRELToken
            if reflection_annotation.isrel == ISRELToken.RELEVANT:
                reflection_score += 1.0
            else:
                reflection_score += 0.0
            num_tokens += 1

        # Average reflection score
        if num_tokens > 0:
            reflection_score /= num_tokens
        else:
            reflection_score = 0.5  # Neutral if no tokens

        # Normalize EigenScore (higher is better)
        if eigenscore is not None:
            # Normalize using typical range (e.g., 0-10)
            # Values > 5 are typically good, < 5 suggest hallucination
            eigenscore_normalized = min(1.0, max(0.0, eigenscore / 10.0))
        else:
            eigenscore_normalized = 0.5  # Neutral if not computed

        # Combined score
        if self.inside_config['combined_scoring']['use_eigenscore'] and eigenscore is not None:
            combined = (
                reflection_weight * reflection_score +
                eigenscore_weight * eigenscore_normalized
            )
        else:
            combined = reflection_score

        return float(combined)

    def set_inside_config(self, config: Dict[str, Any]):
        """Update INSIDE configuration."""
        self.inside_config.update(config)

    def enable_inside(self):
        """Enable INSIDE features."""
        self.inside_config['enabled'] = True

    def disable_inside(self):
        """Disable INSIDE features (use base generator only)."""
        self.inside_config['enabled'] = False

    def calibrate_hallucination_detector(
        self,
        factual_examples: List[Tuple[str, List[str]]],
        hallucinated_examples: List[Tuple[str, List[str]]],
        percentile: float = 10.0
    ):
        """
        Calibrate hallucination detection threshold on labeled data.

        Args:
            factual_examples: List of (query, generations) for factual responses
            hallucinated_examples: List of (query, generations) for hallucinated responses
            percentile: Percentile for threshold calibration
        """
        if self.hallucination_detector:
            threshold = self.hallucination_detector.calibrate(
                factual_examples=factual_examples,
                hallucinated_examples=hallucinated_examples,
                percentile=percentile
            )
            # Update config
            self.inside_config['eigenscore']['threshold'] = threshold

    @classmethod
    def from_config(
        cls,
        generator_config_path: str,
        inside_config_path: str,
        lora_weights_path: Optional[str] = None,
        device: str = 'cpu'
    ) -> 'INSIDEGenerator':
        """
        Create INSIDEGenerator from configuration files.

        Args:
            generator_config_path: Path to generator config YAML
            inside_config_path: Path to INSIDE config YAML
            lora_weights_path: Path to LoRA weights (optional)
            device: Device for computation

        Returns:
            Configured INSIDEGenerator instance
        """
        import yaml

        # Load configs
        with open(generator_config_path) as f:
            gen_config = yaml.safe_load(f)

        with open(inside_config_path) as f:
            inside_config = yaml.safe_load(f)

        # Create base generator
        base_generator = SelfRAGGenerator(
            model_name=gen_config['model']['base_model'],
            device=device,
            load_in_4bit=gen_config['quantization']['load_in_4bit']
        )

        # Load model
        base_generator.load_model(
            lora_weights_path=lora_weights_path,
            quantization_config=gen_config['quantization']
        )

        # Set reflection weights
        if 'inference' in gen_config and 'weights' in gen_config['inference']:
            weights = gen_config['inference']['weights']
            base_generator.set_reflection_weights(
                w_isrel=weights.get('w_isrel', 1.0),
                w_issup=weights.get('w_issup', 1.0),
                w_isuse=weights.get('w_isuse', 1.0)
            )

        # Create INSIDE generator
        inside_gen = cls(
            base_generator=base_generator,
            inside_config=inside_config.get('selfrag_integration', {}),
            device=device
        )

        return inside_gen


def create_inside_generator(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    lora_weights_path: Optional[str] = None,
    inside_config: Optional[Dict[str, Any]] = None,
    device: str = 'cpu'
) -> INSIDEGenerator:
    """
    Factory function to create an INSIDE-enhanced generator.

    Args:
        model_name: Base model name
        lora_weights_path: Path to LoRA weights
        inside_config: INSIDE configuration
        device: Device for computation

    Returns:
        Configured INSIDEGenerator
    """
    # Create base generator
    base_generator = SelfRAGGenerator(
        model_name=model_name,
        device=device,
        load_in_4bit=True
    )

    base_generator.load_model(lora_weights_path=lora_weights_path)

    # Create INSIDE generator
    inside_generator = INSIDEGenerator(
        base_generator=base_generator,
        inside_config=inside_config,
        device=device
    )

    return inside_generator
