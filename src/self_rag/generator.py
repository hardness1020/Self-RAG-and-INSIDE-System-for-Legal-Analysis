"""
Generator Model Module

Implements the Self-RAG generator model that produces both:
1. Task outputs (answers to questions)
2. Reflection tokens (self-evaluation)

The generator is trained on augmented data with reflection tokens
and can perform adaptive retrieval during inference.

Based on the Self-RAG paper by Asai et al. (2023).
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from datasets import Dataset
import yaml
import re

from src.self_rag.reflection_tokens import (
    RetrieveToken,
    ISRELToken,
    ISSUPToken,
    ISUSEToken,
    ReflectionTokenizer,
    ReflectionAnnotation,
)
from src.utils.device_utils import get_optimal_device


class SelfRAGGenerator:
    """
    Self-RAG generator model.

    Generates responses with embedded reflection tokens for self-evaluation.
    Supports adaptive retrieval during inference.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cpu",
        load_in_4bit: bool = True,
    ):
        """
        Initialize generator model.

        Args:
            model_name: HuggingFace model name
            device: Device for inference
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.reflection_weights = {
            'w_isrel': 1.0,
            'w_issup': 1.0,
            'w_isuse': 1.0,
        }

    def load_model(
        self,
        lora_weights_path: Optional[str] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Load base model and tokenizer, optionally with LoRA weights.

        Args:
            lora_weights_path: Path to LoRA adapter weights
            quantization_config: Quantization configuration
        """
        print(f"Loading generator model: {self.model_name}")

        # Note: 4-bit quantization disabled for macOS compatibility
        if self.load_in_4bit:
            print("Warning: 4-bit quantization not supported on macOS. Loading model in full precision.")
            self.load_in_4bit = False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Left padding for generation

        # Add reflection tokens to tokenizer
        special_tokens = ReflectionTokenizer.get_all_special_tokens()
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        # Resize token embeddings for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA weights if provided
        if lora_weights_path:
            print(f"Loading LoRA weights from {lora_weights_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)

        self.model.eval()
        print("Generator model loaded successfully")

    def prepare_for_training(
        self,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Prepare model for training with LoRA.

        Args:
            lora_config: LoRA configuration dictionary
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Setup LoRA with more target modules for generator
        if lora_config is None:
            lora_config = {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': [
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                'lora_dropout': 0.05,
                'bias': "none",
                'task_type': "CAUSAL_LM",
            }

        peft_config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def set_reflection_weights(
        self,
        w_isrel: float = 1.0,
        w_issup: float = 1.0,
        w_isuse: float = 1.0,
    ):
        """
        Set weights for reflection token scoring during inference.

        Args:
            w_isrel: Weight for relevance token
            w_issup: Weight for support token
            w_isuse: Weight for utility token
        """
        self.reflection_weights = {
            'w_isrel': w_isrel,
            'w_issup': w_issup,
            'w_isuse': w_isuse,
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate response with reflection tokens.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling

        Returns:
            Generated text with reflection tokens
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Remove prompt from output
        prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False))
        response = generated[prompt_length:]

        return response

    def generate_with_retrieval(
        self,
        question: str,
        retriever: Any,
        max_new_tokens: int = 512,
        max_retrieval_steps: int = 3,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate response with adaptive retrieval.

        Args:
            question: Question to answer
            retriever: Retriever instance for fetching passages
            max_new_tokens: Maximum tokens to generate
            max_retrieval_steps: Maximum number of retrieval steps

        Returns:
            Tuple of (generated_response, retrieval_history)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        retrieval_history = []
        current_prompt = f"Question: {question}\nAnswer:"
        full_response = ""
        retrieval_count = 0

        while retrieval_count < max_retrieval_steps:
            # Generate next segment
            segment = self.generate(
                current_prompt,
                max_new_tokens=50,  # Generate in small segments
                temperature=0.7,
            )

            # Check for retrieve token
            if RetrieveToken.YES.value in segment:
                # Perform retrieval
                retrieved_docs = retriever.retrieve(question, top_k=3)

                # Store retrieval history
                retrieval_history.append({
                    'step': retrieval_count,
                    'query': question,
                    'documents': retrieved_docs,
                })

                # Add retrieved passage to prompt
                if retrieved_docs:
                    best_passage = retrieved_docs[0]['text']
                    current_prompt += f"\nPassage: {best_passage}\n"

                retrieval_count += 1
            else:
                # Continue generation without retrieval
                full_response += segment

                # Check if generation is complete
                if self.tokenizer.eos_token in segment or len(full_response) >= max_new_tokens:
                    break

                current_prompt += segment

        return full_response, retrieval_history

    def parse_response(
        self,
        response: str,
    ) -> Dict[str, Any]:
        """
        Parse generated response to extract text and reflection tokens.

        Args:
            response: Generated response string

        Returns:
            Dictionary with parsed content and tokens
        """
        # Extract reflection tokens
        reflection = ReflectionTokenizer.extract_tokens_from_text(response)

        # Remove reflection tokens to get clean text
        clean_text = response
        for token in ReflectionTokenizer.get_all_special_tokens():
            clean_text = clean_text.replace(token, '')

        clean_text = clean_text.strip()

        return {
            'text': clean_text,
            'reflection': reflection.to_dict(),
            'raw_response': response,
        }

    def score_response(
        self,
        response: str,
        base_score: float = 1.0,
    ) -> float:
        """
        Score a response based on reflection tokens.

        Args:
            response: Generated response with reflection tokens
            base_score: Base probability score

        Returns:
            Weighted score
        """
        reflection = ReflectionTokenizer.extract_tokens_from_text(response)

        score = base_score

        # Apply reflection token weights
        if reflection.isrel == ISRELToken.RELEVANT:
            score += self.reflection_weights['w_isrel']
        elif reflection.isrel == ISRELToken.IRRELEVANT:
            score -= self.reflection_weights['w_isrel']

        if reflection.issup == ISSUPToken.FULLY_SUPPORTED:
            score += self.reflection_weights['w_issup']
        elif reflection.issup == ISSUPToken.PARTIALLY_SUPPORTED:
            score += 0.5 * self.reflection_weights['w_issup']
        elif reflection.issup == ISSUPToken.NO_SUPPORT:
            score -= self.reflection_weights['w_issup']

        if reflection.isuse:
            utility_score = ISUSEToken.get_score(reflection.isuse)
            normalized_utility = (utility_score - 3) / 2.0  # Normalize to [-1, 1]
            score += normalized_utility * self.reflection_weights['w_isuse']

        return score


def create_generator_training_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> Dataset:
    """
    Create training dataset for generator model.

    Examples should be augmented with reflection tokens from critic.

    Args:
        examples: List of training examples with question, passage, answer, reflection tokens
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset
    """
    formatted_examples = []

    for example in examples:
        question = example['question']
        passage = example.get('passage', '')
        answer = example['answer']
        reflection_tokens = example.get('reflection_tokens', {})

        # Format training example with reflection tokens
        text = f"Question: {question}\n"
        if passage:
            text += f"Passage: {passage}\n"
        text += f"Answer: {answer}"

        # Add reflection tokens
        if reflection_tokens:
            for token_type, token_value in reflection_tokens.items():
                if token_value:
                    text += f" {token_value}"

        formatted_examples.append({'text': text})

    # Convert to dataset
    dataset = Dataset.from_dict({
        'text': [ex['text'] for ex in formatted_examples],
    })

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


def load_generator_from_config(config_path: str) -> SelfRAGGenerator:
    """
    Load generator model from configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured SelfRAGGenerator instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    quantization_config = config.get('quantization', {})

    generator = SelfRAGGenerator(
        model_name=model_config.get('base_model', 'meta-llama/Llama-2-7b-hf'),
        device=get_optimal_device(prefer_gpu=True, verbose=False),
        load_in_4bit=quantization_config.get('load_in_4bit', True),
    )

    # Set reflection weights if specified
    inference_config = config.get('inference', {})
    weights = inference_config.get('weights', {})
    if weights:
        generator.set_reflection_weights(
            w_isrel=weights.get('w_isrel', 1.0),
            w_issup=weights.get('w_issup', 1.0),
            w_isuse=weights.get('w_isuse', 1.0),
        )

    return generator


if __name__ == "__main__":
    # Example usage
    print("Self-RAG Generator Example\n")
    print("=" * 80)

    print("\nExample generation workflow:")
    print("1. Load base model (Llama-2-7B) with LoRA adapters")
    print("2. Generate response with reflection tokens")
    print("3. Parse response to extract text and tokens")
    print("4. Score response based on reflection tokens")

    print("\nExample input:")
    question = "What are the elements of negligence in tort law?"

    print(f"Question: {question}")

    print("\nExpected output format:")
    print("Answer: To establish negligence, the plaintiff must prove...")
    print("[Retrieve] [Relevant] [Fully Supported] [Utility:5]")

    print("\nThe generator learns to:")
    print("- Produce accurate legal answers")
    print("- Emit appropriate reflection tokens")
    print("- Trigger retrieval when needed")
    print("- Self-evaluate response quality")
