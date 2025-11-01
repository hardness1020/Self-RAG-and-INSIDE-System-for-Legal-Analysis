"""
Critic Model Training Script

Trains the critic model using QLoRA (4-bit quantization + LoRA) to predict reflection tokens.
The critic learns to evaluate text and assign appropriate reflection tokens for training the generator.

Usage:
    uv run python -m src.training.train_critic_qlora --config configs/critic_config.yaml
"""

import os
import sys
from pathlib import Path
import argparse
import json
import yaml
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from self_rag.reflection_tokens import ReflectionTokenizer, GPT4_PROMPTS
from self_rag.critic import CriticModel
from utils.device_utils import get_optimal_device


def load_training_data(data_dir: str, num_samples: int = None) -> Dataset:
    """
    Load training data for critic model.

    Expected format: JSON file with fields:
    - question: str
    - passage: str (optional)
    - answer: str (optional)
    - reflection_tokens: dict with token labels

    Args:
        data_dir: Directory containing training data
        num_samples: Number of samples to load (None for all)

    Returns:
        HuggingFace Dataset
    """
    data_file = os.path.join(data_dir, 'labeled_data.json')

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Training data not found at {data_file}. "
            "Run generate_labels.py first to create training data."
        )

    print(f"Loading training data from {data_file}")
    with open(data_file, 'r') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    print(f"Loaded {len(data)} examples")

    # Convert to dataset
    dataset = Dataset.from_list(data)
    return dataset


def format_training_examples(examples: Dataset, tokenizer: AutoTokenizer) -> list:
    """
    Format examples for training with prompts and labels.

    Args:
        examples: Dataset with training examples
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

        # Create training examples for each token type
        # Retrieve token
        if 'retrieve' in tokens and answer:
            prompt = GPT4_PROMPTS['retrieve'].format(
                question=question,
                answer=answer
            )
            full_text = f"{prompt} {tokens['retrieve']}{tokenizer.eos_token}"
            formatted_examples.append({'text': full_text})

        # ISREL token
        if 'isrel' in tokens and passage:
            prompt = GPT4_PROMPTS['isrel'].format(
                question=question,
                passage=passage
            )
            full_text = f"{prompt} {tokens['isrel']}{tokenizer.eos_token}"
            formatted_examples.append({'text': full_text})

        # ISSUP token
        if 'issup' in tokens and passage and answer:
            prompt = GPT4_PROMPTS['issup'].format(
                question=question,
                passage=passage,
                answer=answer
            )
            full_text = f"{prompt} {tokens['issup']}{tokenizer.eos_token}"
            formatted_examples.append({'text': full_text})

        # ISUSE token
        if 'isuse' in tokens and answer:
            prompt = GPT4_PROMPTS['isuse'].format(
                question=question,
                answer=answer
            )
            full_text = f"{prompt} {tokens['isuse']}{tokenizer.eos_token}"
            formatted_examples.append({'text': full_text})

    return formatted_examples


def train_critic(config_path: str, resume_from_checkpoint: str = None):
    """
    Train critic model with QLoRA.

    Args:
        config_path: Path to configuration YAML file
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("CRITIC MODEL TRAINING")
    print("=" * 80)
    print(f"\nConfiguration loaded from: {config_path}")

    # Get project root directory (configs directory is one level below project root)
    config_dir = Path(config_path).parent.resolve()
    project_root = config_dir.parent  # Go up one level from configs/ to project root

    # Extract config sections
    model_config = config.get('model', {})
    quant_config = config.get('quantization', {})
    lora_config = config.get('lora', {})
    data_config = config.get('data', {})
    training_config = config.get('training', {})

    # Resolve relative paths based on project root
    if 'training_data_dir' in data_config:
        data_dir = Path(data_config['training_data_dir'])
        if not data_dir.is_absolute():
            data_config['training_data_dir'] = str((project_root / data_dir).resolve())
            print(f"Resolved training_data_dir: {data_config['training_data_dir']}")

    if 'output_dir' in training_config:
        output_dir = Path(training_config['output_dir'])
        if not output_dir.is_absolute():
            training_config['output_dir'] = str((project_root / output_dir).resolve())
            print(f"Resolved output_dir: {training_config['output_dir']}")

    # Setup device (supports MPS for Mac GPU)
    device = get_optimal_device(prefer_gpu=True, verbose=True)

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

    # Load model (4-bit quantization disabled for macOS compatibility)
    print("\n2. Loading base model...")
    print("   Note: 4-bit quantization disabled for macOS compatibility")

    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model'],
        trust_remote_code=True,
    )
    model = model.to(device)

    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = False  # Freeze base model

    # Setup LoRA (without k-bit training preparation for macOS)
    print("\n3. Preparing model for LoRA training...")
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

    # Load training data
    print("\n4. Loading and formatting training data...")
    training_data = load_training_data(
        data_config['training_data_dir'],
        num_samples=data_config.get('num_samples_per_token')
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

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=data_config.get('max_seq_length', 512),
            padding='max_length',
        )

    print("\n5. Tokenizing datasets...")
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
    print("\n6. Setting up training...")
    output_dir = training_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 2e-4),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_ratio=training_config.get('warmup_ratio', 0.03),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        logging_dir=f"{output_dir}/logs",
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 100),
        eval_steps=training_config.get('eval_steps', 100),
        save_total_limit=training_config.get('save_total_limit', 3),
        eval_strategy="steps" if eval_dataset else "no",  # Changed from evaluation_strategy
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', False),
        optim=training_config.get('optim', 'paged_adamw_32bit'),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        max_grad_norm=training_config.get('max_grad_norm', 0.3),
        report_to="none",  # Disable wandb/tensorboard
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
    print("\n7. Starting training...")
    print("=" * 80)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print("\n8. Saving final model...")
    final_output_dir = os.path.join(output_dir, 'final')
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {final_output_dir}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Train Critic Model with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/critic_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    train_critic(args.config, args.resume)


if __name__ == "__main__":
    main()
