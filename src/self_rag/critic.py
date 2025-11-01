"""
Critic Model Module

Implements the critic model for predicting reflection tokens.
The critic evaluates text and predicts appropriate reflection tokens
(Retrieve, ISREL, ISSUP, ISUSE) for use in Self-RAG training.

Based on the Self-RAG paper by Asai et al. (2023).
"""

from typing import Dict, List, Any, Optional, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from datasets import Dataset
import yaml

from src.self_rag.reflection_tokens import (
    RetrieveToken,
    ISRELToken,
    ISSUPToken,
    ISUSEToken,
    ReflectionTokenizer,
    GPT4_PROMPTS,
)
from src.utils.device_utils import get_optimal_device


class CriticModel:
    """
    Critic model for predicting reflection tokens.

    Can be used for:
    1. Training a critic model with QLoRA
    2. Inference to predict reflection tokens
    3. Generating training data for the generator model
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cpu",
        load_in_4bit: bool = True,
    ):
        """
        Initialize critic model.

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
        print(f"Loading model: {self.model_name}")

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
        self.tokenizer.padding_side = "right"

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

        print("Model loaded successfully")

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

        # Setup LoRA
        if lora_config is None:
            lora_config = {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],
                'lora_dropout': 0.05,
                'bias': "none",
                'task_type': "CAUSAL_LM",
            }

        peft_config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def predict_token(
        self,
        prompt: str,
        token_type: str,
        max_new_tokens: int = 10,
        temperature: float = 0.0,
    ) -> str:
        """
        Predict a single reflection token.

        Args:
            prompt: Input prompt
            token_type: Type of token to predict ('retrieve', 'isrel', 'issup', 'isuse')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)

        Returns:
            Predicted token string
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
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract token based on type
        if token_type == 'retrieve':
            for token in RetrieveToken:
                if token.value in generated:
                    return token.value

        elif token_type == 'isrel':
            for token in ISRELToken:
                if token.value in generated:
                    return token.value

        elif token_type == 'issup':
            for token in ISSUPToken:
                if token.value in generated:
                    return token.value

        elif token_type == 'isuse':
            for token in ISUSEToken:
                if token.value in generated:
                    return token.value

        return None

    def predict_all_tokens(
        self,
        question: str,
        passage: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Predict all relevant reflection tokens for a QA example.

        Args:
            question: Question text
            passage: Retrieved passage (optional)
            answer: Answer text (optional)

        Returns:
            Dictionary with predicted tokens
        """
        predictions = {}

        # Predict Retrieve token
        if answer:
            retrieve_prompt = GPT4_PROMPTS['retrieve'].format(
                question=question,
                answer=answer,
            )
            predictions['retrieve'] = self.predict_token(retrieve_prompt, 'retrieve')

        # Predict ISREL token
        if passage:
            isrel_prompt = GPT4_PROMPTS['isrel'].format(
                question=question,
                passage=passage,
            )
            predictions['isrel'] = self.predict_token(isrel_prompt, 'isrel')

        # Predict ISSUP token
        if passage and answer:
            issup_prompt = GPT4_PROMPTS['issup'].format(
                question=question,
                passage=passage,
                answer=answer,
            )
            predictions['issup'] = self.predict_token(issup_prompt, 'issup')

        # Predict ISUSE token
        if answer:
            isuse_prompt = GPT4_PROMPTS['isuse'].format(
                question=question,
                answer=answer,
            )
            predictions['isuse'] = self.predict_token(isuse_prompt, 'isuse')

        return predictions


def create_critic_training_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> Dataset:
    """
    Create training dataset for critic model.

    Args:
        examples: List of training examples with question, passage, answer, labels
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset
    """
    formatted_examples = []

    for example in examples:
        question = example['question']
        passage = example.get('passage', '')
        answer = example.get('answer', '')
        token_type = example['token_type']  # 'retrieve', 'isrel', 'issup', 'isuse'
        label = example['label']  # Target token

        # Format prompt based on token type
        if token_type == 'retrieve':
            prompt = GPT4_PROMPTS['retrieve'].format(
                question=question,
                answer=answer,
            )
        elif token_type == 'isrel':
            prompt = GPT4_PROMPTS['isrel'].format(
                question=question,
                passage=passage,
            )
        elif token_type == 'issup':
            prompt = GPT4_PROMPTS['issup'].format(
                question=question,
                passage=passage,
                answer=answer,
            )
        elif token_type == 'isuse':
            prompt = GPT4_PROMPTS['isuse'].format(
                question=question,
                answer=answer,
            )
        else:
            continue

        # Create full text with label
        full_text = f"{prompt} {label}"

        formatted_examples.append({
            'text': full_text,
            'prompt': prompt,
            'label': label,
        })

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


def load_critic_from_config(config_path: str) -> CriticModel:
    """
    Load critic model from configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured CriticModel instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    quantization_config = config.get('quantization', {})

    critic = CriticModel(
        model_name=model_config.get('base_model', 'meta-llama/Llama-2-7b-hf'),
        device=get_optimal_device(prefer_gpu=True, verbose=False),
        load_in_4bit=quantization_config.get('load_in_4bit', True),
    )

    return critic


if __name__ == "__main__":
    # Example usage
    print("Critic Model Example\n")
    print("=" * 80)

    # This is just a demonstration - actual usage requires model weights
    print("\nExample prediction workflow:")
    print("1. Load base model (Llama-2-7B)")
    print("2. Optionally load LoRA weights from training")
    print("3. Predict reflection tokens for QA examples")

    print("\nExample input:")
    question = "What are the elements of negligence?"
    answer = "Negligence requires duty, breach, causation, and damages."
    passage = "To establish negligence, plaintiff must prove four elements..."

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Passage: {passage[:50]}...")

    print("\nExpected predictions:")
    print("- Retrieve: [Retrieve] (needs legal knowledge)")
    print("- ISREL: [Relevant] (passage is about negligence)")
    print("- ISSUP: [Fully Supported] (answer matches passage)")
    print("- ISUSE: [Utility:5] (complete answer)")
