"""
Internal States Extraction

Extract and process internal embeddings from LLM hidden layers for hallucination detection.
Based on the INSIDE paper's approach of leveraging dense semantic information.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedModel
import numpy as np


class InternalStatesExtractor:
    """
    Extracts internal hidden states from specified layers of an LLM.

    The INSIDE paper shows that middle layers contain rich semantic information
    that can be used for hallucination detection via EigenScore.

    Args:
        model: Hugging Face transformer model
        target_layers: List of layer indices to extract (default: middle layer)
        extraction_position: Position to extract from ('last', 'first', 'mean')
        device: Device to perform computations on
    """

    def __init__(
        self,
        model: PreTrainedModel,
        target_layers: Optional[List[int]] = None,
        extraction_position: str = 'last',
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.extraction_position = extraction_position

        # Default to middle layer if not specified (e.g., layer 16 for Llama-2-7B with 32 layers)
        if target_layers is None:
            num_layers = model.config.num_hidden_layers
            self.target_layers = [num_layers // 2]
        else:
            self.target_layers = target_layers

        # Storage for captured hidden states
        self.hidden_states = {}
        self.hooks = []

    def _create_hook(self, layer_idx: int):
        """Create a forward hook to capture hidden states from a specific layer."""
        def hook(module, input, output):
            # Store the hidden state (output is typically a tuple with hidden_state as first element)
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output

            self.hidden_states[layer_idx] = hidden_state.detach()

        return hook

    def register_hooks(self):
        """Register forward hooks on target layers."""
        # Clear existing hooks
        self.remove_hooks()

        # Get the transformer layers (e.g., model.model.layers for Llama)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Could not find transformer layers in model")

        # Register hooks on target layers
        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.hidden_states = {}

    def extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentences: Optional[List[str]] = None,
        sentence_boundaries: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract sentence-level embeddings from internal states.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            sentences: List of sentences (for sentence-level extraction)
            sentence_boundaries: List of (start, end) token positions for each sentence

        Returns:
            Dictionary mapping layer_idx to embeddings tensor (num_sentences, hidden_dim)
        """
        self.register_hooks()

        # Forward pass to capture hidden states
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False  # We use hooks instead
            )

        # Process captured hidden states
        embeddings = {}

        for layer_idx, hidden_state in self.hidden_states.items():
            # hidden_state shape: (batch_size, seq_len, hidden_dim)

            if sentence_boundaries is not None:
                # Extract sentence-level embeddings based on boundaries
                sentence_embeddings = []

                for start, end in sentence_boundaries:
                    if self.extraction_position == 'last':
                        # Use last token of sentence
                        emb = hidden_state[0, end - 1, :]
                    elif self.extraction_position == 'first':
                        # Use first token of sentence
                        emb = hidden_state[0, start, :]
                    elif self.extraction_position == 'mean':
                        # Use mean of all tokens in sentence
                        emb = hidden_state[0, start:end, :].mean(dim=0)
                    else:
                        raise ValueError(f"Unknown extraction position: {self.extraction_position}")

                    sentence_embeddings.append(emb)

                embeddings[layer_idx] = torch.stack(sentence_embeddings)
            else:
                # Extract based on position (for whole sequence)
                if self.extraction_position == 'last':
                    if attention_mask is not None:
                        # Get last non-padded token
                        seq_lengths = attention_mask.sum(dim=1) - 1
                        emb = hidden_state[0, seq_lengths[0], :]
                    else:
                        emb = hidden_state[0, -1, :]
                elif self.extraction_position == 'first':
                    emb = hidden_state[0, 0, :]
                elif self.extraction_position == 'mean':
                    if attention_mask is not None:
                        # Mean over non-padded tokens
                        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size())
                        emb = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                        emb = emb[0]
                    else:
                        emb = hidden_state[0].mean(dim=0)

                embeddings[layer_idx] = emb.unsqueeze(0)  # Add sentence dimension

        self.remove_hooks()

        return embeddings

    def extract_from_generations(
        self,
        generations: List[str],
        tokenizer,
        split_sentences: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from a list of generated texts.

        Args:
            generations: List of generated text strings
            tokenizer: Tokenizer for encoding text
            split_sentences: Whether to extract sentence-level embeddings

        Returns:
            Dictionary mapping layer_idx to embeddings tensor
        """
        all_embeddings = {layer_idx: [] for layer_idx in self.target_layers}

        for text in generations:
            if split_sentences:
                # Simple sentence splitting (can be improved with nltk)
                sentences = [s.strip() for s in text.split('.') if s.strip()]

                # Tokenize full text to get sentence boundaries
                full_tokens = tokenizer.encode(text, return_tensors='pt').to(self.device)

                # Approximate sentence boundaries (simple approach)
                sentence_boundaries = []
                start = 0
                for sent in sentences:
                    sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
                    end = start + len(sent_tokens)
                    sentence_boundaries.append((start, end))
                    start = end

                # Extract embeddings
                embeddings = self.extract_embeddings(
                    input_ids=full_tokens,
                    sentence_boundaries=sentence_boundaries
                )
            else:
                # Extract whole-text embedding
                tokens = tokenizer.encode(text, return_tensors='pt').to(self.device)
                embeddings = self.extract_embeddings(input_ids=tokens)

            # Accumulate embeddings
            for layer_idx, emb in embeddings.items():
                all_embeddings[layer_idx].append(emb)

        # Concatenate all embeddings
        result = {}
        for layer_idx, emb_list in all_embeddings.items():
            if emb_list:
                result[layer_idx] = torch.cat(emb_list, dim=0)

        return result


def split_into_sentences(text: str, tokenizer=None) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Split text into sentences and compute token boundaries.

    Args:
        text: Input text
        tokenizer: Tokenizer for computing token positions

    Returns:
        Tuple of (sentences, boundaries) where boundaries are (start, end) token positions
    """
    # Simple sentence splitting (can be improved with nltk)
    sentences = []
    for s in text.replace('!', '.').replace('?', '.').split('.'):
        s = s.strip()
        if s:
            sentences.append(s)

    boundaries = []
    if tokenizer is not None:
        start = 0
        for sent in sentences:
            sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
            end = start + len(sent_tokens)
            boundaries.append((start, end))
            start = end

    return sentences, boundaries


def extract_sentence_embeddings(
    model: PreTrainedModel,
    text: str,
    tokenizer,
    target_layer: Optional[int] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Convenience function to extract sentence embeddings from text.

    Args:
        model: Language model
        text: Input text
        tokenizer: Tokenizer
        target_layer: Layer to extract from (default: middle layer)
        device: Device for computation

    Returns:
        Sentence embeddings tensor (num_sentences, hidden_dim)
    """
    extractor = InternalStatesExtractor(
        model=model,
        target_layers=[target_layer] if target_layer is not None else None,
        device=device
    )

    sentences, boundaries = split_into_sentences(text, tokenizer)
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)

    embeddings = extractor.extract_embeddings(
        input_ids=tokens,
        sentence_boundaries=boundaries
    )

    # Return embeddings from first (or only) layer
    layer_idx = list(embeddings.keys())[0]
    return embeddings[layer_idx]
