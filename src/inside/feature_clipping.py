"""
Feature Clipping for Overconfident Hallucination Detection

Implements test-time feature clipping from the INSIDE paper. Truncating extreme
activations in internal states helps detect self-consistent (overconfident)
hallucinations.

Key insight: Overconfident hallucinations show extreme activations that, when
clipped, lead to significant changes in model behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from transformers import PreTrainedModel


class FeatureClipper:
    """
    Clips extreme activations in LLM hidden states at test time.

    The INSIDE paper shows that clipping extreme features helps detect
    overconfident hallucinations by revealing model uncertainty.

    Args:
        percentile: Percentile for clipping threshold (e.g., 95 = clip top/bottom 5%)
        symmetric: Whether to clip both positive and negative extremes
        layer_specific: Whether to compute thresholds per layer
    """

    def __init__(
        self,
        percentile: float = 95.0,
        symmetric: bool = True,
        layer_specific: bool = True
    ):
        self.percentile = percentile
        self.symmetric = symmetric
        self.layer_specific = layer_specific

        # Store computed thresholds
        self.thresholds = {}

    def compute_thresholds(
        self,
        activations: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Compute clipping thresholds from activation statistics.

        Args:
            activations: Activation tensor (any shape)
            layer_idx: Layer index for caching

        Returns:
            Tuple of (lower_threshold, upper_threshold)
        """
        activations_flat = activations.flatten().cpu().numpy()

        if self.symmetric:
            # Clip both extremes symmetrically
            upper_threshold = np.percentile(np.abs(activations_flat), self.percentile)
            lower_threshold = -upper_threshold
        else:
            # Clip asymmetrically
            lower_threshold = np.percentile(activations_flat, 100 - self.percentile)
            upper_threshold = np.percentile(activations_flat, self.percentile)

        # Cache thresholds if layer-specific
        if layer_idx is not None and self.layer_specific:
            self.thresholds[layer_idx] = (lower_threshold, upper_threshold)

        return lower_threshold, upper_threshold

    def clip_activations(
        self,
        activations: torch.Tensor,
        layer_idx: Optional[int] = None,
        thresholds: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Clip extreme activations.

        Args:
            activations: Activation tensor to clip
            layer_idx: Layer index (for cached thresholds)
            thresholds: Optional pre-computed thresholds (lower, upper)

        Returns:
            Clipped activations
        """
        if thresholds is None:
            # Use cached thresholds if available
            if layer_idx is not None and layer_idx in self.thresholds:
                lower, upper = self.thresholds[layer_idx]
            else:
                # Compute thresholds from current activations
                lower, upper = self.compute_thresholds(activations, layer_idx)
        else:
            lower, upper = thresholds

        # Clip activations
        clipped = torch.clamp(activations, min=lower, max=upper)

        return clipped

    def calibrate_on_data(
        self,
        model: PreTrainedModel,
        calibration_texts: List[str],
        tokenizer,
        target_layers: List[int],
        device: str = 'cpu'
    ):
        """
        Calibrate clipping thresholds on a set of texts.

        Args:
            model: Language model
            calibration_texts: List of texts for calibration
            tokenizer: Tokenizer
            target_layers: Layers to calibrate
            device: Device for computation
        """
        from .internal_states import InternalStatesExtractor

        extractor = InternalStatesExtractor(
            model=model,
            target_layers=target_layers,
            device=device
        )

        all_activations = {layer: [] for layer in target_layers}

        # Collect activations from calibration texts
        for text in calibration_texts:
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
            embeddings = extractor.extract_embeddings(tokens)

            for layer_idx, emb in embeddings.items():
                all_activations[layer_idx].append(emb)

        # Compute thresholds for each layer
        for layer_idx, embs in all_activations.items():
            if embs:
                all_embs = torch.cat(embs, dim=0)
                self.compute_thresholds(all_embs, layer_idx=layer_idx)

        print(f"Calibrated clipping thresholds for {len(target_layers)} layers")
        for layer_idx, (lower, upper) in self.thresholds.items():
            print(f"  Layer {layer_idx}: [{lower:.2f}, {upper:.2f}]")


class ClippedGenerator:
    """
    Wrapper that generates text with feature clipping applied.

    Compares generations with and without clipping to detect overconfident
    hallucinations.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        clipper: FeatureClipper,
        target_layers: List[int],
        device: str = 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.clipper = clipper
        self.target_layers = target_layers
        self.device = device

    def generate_with_clipping(
        self,
        input_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate text with feature clipping applied.

        Args:
            input_text: Input prompt
            max_length: Maximum generation length
            num_return_sequences: Number of sequences to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # Register hooks to apply clipping during generation
        hooks = []

        def create_clipping_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_state = output[0]
                    clipped = self.clipper.clip_activations(hidden_state, layer_idx)
                    # Return modified output
                    return (clipped,) + output[1:]
                else:
                    return self.clipper.clip_activations(output, layer_idx)
            return hook

        # Get model layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Could not find transformer layers")

        # Register hooks
        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(create_clipping_hook(layer_idx))
                hooks.append(hook)

        # Generate with clipping
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                **generation_kwargs
            )

        # Decode outputs
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return generated_texts

    def compare_with_without_clipping(
        self,
        input_text: str,
        max_length: int = 512,
        num_samples: int = 3,
        **generation_kwargs
    ) -> Dict[str, List[str]]:
        """
        Generate with and without clipping to detect overconfidence.

        Large differences between clipped and unclipped generations suggest
        overconfident hallucinations.

        Args:
            input_text: Input prompt
            max_length: Maximum generation length
            num_samples: Number of samples to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with 'clipped' and 'unclipped' generations
        """
        # Generate without clipping (standard)
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs_unclipped = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_samples,
                **generation_kwargs
            )

        unclipped_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs_unclipped
        ]

        # Generate with clipping
        clipped_texts = self.generate_with_clipping(
            input_text,
            max_length=max_length,
            num_return_sequences=num_samples,
            **generation_kwargs
        )

        return {
            'unclipped': unclipped_texts,
            'clipped': clipped_texts
        }


def measure_clipping_sensitivity(
    clipped_texts: List[str],
    unclipped_texts: List[str],
    metric: str = 'edit_distance'
) -> float:
    """
    Measure how much clipping changed the generations.

    Large changes suggest overconfident generation.

    Args:
        clipped_texts: Texts generated with clipping
        unclipped_texts: Texts generated without clipping
        metric: Metric to use ('edit_distance', 'token_overlap', 'semantic')

    Returns:
        Sensitivity score (higher = more sensitive to clipping = more overconfident)
    """
    if len(clipped_texts) != len(unclipped_texts):
        raise ValueError("Must have same number of clipped and unclipped texts")

    sensitivities = []

    for clipped, unclipped in zip(clipped_texts, unclipped_texts):
        if metric == 'edit_distance':
            # Normalized edit distance
            import difflib
            distance = difflib.SequenceMatcher(None, clipped, unclipped).ratio()
            sensitivity = 1.0 - distance
        elif metric == 'token_overlap':
            # Token-level Jaccard distance
            clipped_tokens = set(clipped.split())
            unclipped_tokens = set(unclipped.split())
            overlap = len(clipped_tokens & unclipped_tokens)
            union = len(clipped_tokens | unclipped_tokens)
            similarity = overlap / union if union > 0 else 0
            sensitivity = 1.0 - similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")

        sensitivities.append(sensitivity)

    return np.mean(sensitivities)


def detect_overconfident_hallucination(
    clipped_texts: List[str],
    unclipped_texts: List[str],
    sensitivity_threshold: float = 0.3
) -> Tuple[bool, float]:
    """
    Detect overconfident hallucination using clipping sensitivity.

    Args:
        clipped_texts: Texts generated with clipping
        unclipped_texts: Texts generated without clipping
        sensitivity_threshold: Threshold for detection

    Returns:
        Tuple of (is_overconfident_hallucination, sensitivity_score)
    """
    sensitivity = measure_clipping_sensitivity(clipped_texts, unclipped_texts)

    is_overconfident = sensitivity > sensitivity_threshold

    return is_overconfident, sensitivity


class AdaptiveClipper:
    """
    Adaptive feature clipper that adjusts clipping strength based on context.
    """

    def __init__(
        self,
        base_percentile: float = 95.0,
        min_percentile: float = 90.0,
        max_percentile: float = 99.0
    ):
        self.base_percentile = base_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def adjust_percentile(
        self,
        query_complexity: float,
        model_confidence: float
    ) -> float:
        """
        Adjust clipping percentile based on query and model characteristics.

        Args:
            query_complexity: Complexity score (0-1)
            model_confidence: Model confidence (0-1)

        Returns:
            Adjusted percentile
        """
        # More complex queries or lower confidence → less aggressive clipping
        adjustment = (query_complexity * 0.5 + (1 - model_confidence) * 0.5)

        adjusted_percentile = (
            self.base_percentile +
            (self.max_percentile - self.base_percentile) * adjustment
        )

        return np.clip(adjusted_percentile, self.min_percentile, self.max_percentile)
