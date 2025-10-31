"""
Generator Model Training Script

Trains the Self-RAG generator model using QLoRA with augmented training data.
The generator learns to produce both answers and reflection tokens.

Usage:
    uv run python -m src.training.train_generator_qlora --config configs/generator_config.yaml
"""

import os
import sys
from pathlib import Path
import argparse
import json
import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from self_rag.reflection_tokens import ReflectionTokenizer
from self_rag.critic import CriticModel


def load_and_augment_data(
    data_dir: str,
    critic_model: CriticModel = None,
    num_samples: int = None
) -> Dataset:
    """
    Load training data and augment with reflection tokens from critic.

    Args:
        data_dir: Directory containing training data
        critic_model: Trained critic model for generating tokens (optional)
        num_samples: Number of samples to load

    Returns:
        HuggingFace Dataset with augmented data
    """
    data_file = os.path.join(data_dir, 'labeled_data.json')

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Training data not found at {data_file}. "
            "Run generate_labels.py first."
        )

    print(f"Loading training data from {data_file}")
    with open(data_file, 'r') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    print(f"Loaded {len(data)} examples")

    # If critic model provided, use it to augment/verify tokens
    if critic_model:
        print("Augmenting data with critic model predictions...")
        for example in tqdm(data):
            predictions = critic_model.predict_all_tokens(
                question=example['question'],
                passage=example.get('passage'),
                answer=example.get('answer')
            )
            # Use critic predictions (can override or merge with existing)
            example['reflection_tokens'] = predictions

    # Convert to dataset
    dataset = Dataset.from_list(data)
    return dataset


def format_training_examples(examples: Dataset, tokenizer: AutoTokenizer) -> list:
    """
    Format examples for generator training.

    Format: Question + [optional Passage] + Answer + Reflection Tokens

    Args:
        examples: Dataset with augmented examples
        tokenizer: Tokenizer instance

    Returns:
        List of formatted training examples
    """
    formatted_examples = []

    for example in tqdm(examples, desc="Formatting examples"):
        question = example['question']
        passage = example.get('passage', '')
        answer = example.get('answer', '')
        tokens = example.get('reflection_tokens', {})

        # Build training text
        text_parts = [f"Question: {question}"]

        # Add passage if available
        if passage:
            text_parts.append(f"Passage: {passage}")

        # Add answer
        text_parts.append(f"Answer: {answer}")

        # Add reflection tokens
        token_str = []
        for token_type in ['retrieve', 'isrel', 'issup', 'isuse']:
            if token_type in tokens and tokens[token_type]:
                token_str.append(tokens[token_type])

        if token_str:
            text_parts.append(" ".join(token_str))

        # Combine and add EOS
        full_text = "\n".join(text_parts) + tokenizer.eos_token

        formatted_examples.append({'text': full_text})

    return formatted_examples


def train_generator(
    config_path: str,
    critic_weights_path: str = None,
    resume_from_checkpoint: str = None
):
    """
    Train generator model with QLoRA.

    Args:
        config_path: Path to configuration YAML file
        critic_weights_path: Path to trained critic LoRA weights (optional)
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("GENERATOR MODEL TRAINING")
    print("=" * 80)
    print(f"\nConfiguration loaded from: {config_path}")

    # Extract config sections
    model_config = config.get('model', {})
    quant_config = config.get('quantization', {})
    lora_config = config.get('lora', {})
    data_config = config.get('data', {})
    training_config = config.get('training', {})

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['base_model'],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add reflection tokens
    special_tokens = ReflectionTokenizer.get_all_special_tokens()
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"   Added {len(special_tokens)} reflection tokens to vocabulary")

    # Load critic model if provided
    critic_model = None
    if critic_weights_path:
        print(f"\n2. Loading critic model from {critic_weights_path}...")
        critic_model = CriticModel(model_name=model_config['base_model'])
        critic_model.load_model(lora_weights_path=critic_weights_path)
        print("   Critic model loaded")

    # Load and augment training data
    print(f"\n{'2' if not critic_weights_path else '3'}. Loading training data...")
    training_data = load_and_augment_data(
        data_config['training_data_dir'],
        critic_model=critic_model,
        num_samples=data_config.get('num_samples')
    )

    # Format examples
    formatted_examples = format_training_examples(training_data, tokenizer)
    print(f"   Created {len(formatted_examples)} training examples")

    # Create dataset
    train_dataset = Dataset.from_list(formatted_examples)

    # Split into train/val
    val_split = data_config.get('validation_split', 0.1)
    if val_split > 0:
        split_dataset = train_dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"   Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    else:
        eval_dataset = None
        print(f"   Train: {len(train_dataset)}")

    # Load model with quantization
    step_num = 3 if not critic_weights_path else 4
    print(f"\n{step_num}. Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.get('load_in_4bit', True),
        bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'float16')),
        bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Prepare for k-bit training
    step_num += 1
    print(f"\n{step_num}. Preparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model)

    # Setup LoRA (more target modules for generator)
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=lora_config.get('target_modules'),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        bias=lora_config.get('bias', 'none'),
        task_type=lora_config.get('task_type', 'CAUSAL_LM'),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=data_config.get('max_seq_length', 1024),
            padding='max_length',
        )

    step_num += 1
    print(f"\n{step_num}. Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )

    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing validation"
        )

    # Setup training arguments
    step_num += 1
    print(f"\n{step_num}. Setting up training...")
    output_dir = training_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 2),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
        learning_rate=training_config.get('learning_rate', 2e-4),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_ratio=training_config.get('warmup_ratio', 0.03),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        logging_dir=f"{output_dir}/logs",
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 200),
        eval_steps=training_config.get('eval_steps', 200),
        save_total_limit=training_config.get('save_total_limit', 3),
        evaluation_strategy="steps" if eval_dataset else "no",
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', False),
        optim=training_config.get('optim', 'paged_adamw_32bit'),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        max_grad_norm=training_config.get('max_grad_norm', 0.3),
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    step_num += 1
    print(f"\n{step_num}. Starting training...")
    print("=" * 80)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    step_num += 1
    print(f"\n{step_num}. Saving final model...")
    final_output_dir = os.path.join(output_dir, 'final')
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {final_output_dir}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Train Generator Model with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generator_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--critic-weights",
        type=str,
        default=None,
        help="Path to trained critic LoRA weights"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    train_generator(args.config, args.critic_weights, args.resume)


if __name__ == "__main__":
    main()
