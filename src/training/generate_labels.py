"""
Label Generation Module

Generates Self-RAG reflection token labels for training data using:
1. Local LLM (Qwen2.5-7B-Instruct) - Free and open-source
2. GPT-4 prompting (if API available)
3. Rule-based heuristics (fallback)

Creates training data for both critic and generator models.
Generates labels for: Retrieve, ISREL, ISSUP, ISUSE tokens.

Note: INTENT is handled separately by INSIDE's IntentDetector.
"""

from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import torch

# Optional: OpenAI for GPT-4 labeling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Transformers for local LLM
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))

from self_rag.reflection_tokens import (
    RetrieveToken,
    ISRELToken,
    ISSUPToken,
    ISUSEToken,
    GPT4_PROMPTS,
)


class LabelGenerator:
    """
    Generates reflection token labels for training examples.
    """

    def __init__(
        self,
        use_local_llm: bool = True,
        use_gpt4: bool = False,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
        local_model: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
    ):
        """
        Initialize label generator.

        Args:
            use_local_llm: Whether to use local LLM (Qwen) for labeling
            use_gpt4: Whether to use GPT-4 for labeling
            openai_api_key: OpenAI API key
            model: OpenAI model to use
            local_model: Local LLM model name
            device: Device for local LLM (None for auto-detect)
        """
        self.use_local_llm = use_local_llm and TRANSFORMERS_AVAILABLE
        self.use_gpt4 = use_gpt4 and OPENAI_AVAILABLE
        self.local_model = None
        self.local_tokenizer = None

        # Priority: Local LLM > GPT-4 > Rule-based
        if self.use_local_llm:
            try:
                print(f"Loading local LLM: {local_model}...")
                self.local_tokenizer = AutoTokenizer.from_pretrained(local_model)

                # Auto-detect device
                if device is None:
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"

                print(f"Using device: {device}")

                self.local_model = AutoModelForCausalLM.from_pretrained(
                    local_model,
                    torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                    device_map=device if device != "mps" else None,
                )

                if device == "mps":
                    self.local_model = self.local_model.to("mps")

                self.local_model.eval()
                self.device = device
                print("Local LLM loaded successfully!")

            except Exception as e:
                print(f"Error loading local LLM: {e}")
                print("Falling back to rule-based approach.")
                self.use_local_llm = False

        elif self.use_gpt4:
            if openai_api_key:
                openai.api_key = openai_api_key
            elif 'OPENAI_API_KEY' in os.environ:
                openai.api_key = os.environ['OPENAI_API_KEY']
            else:
                print("Warning: GPT-4 requested but no API key found. Using rule-based approach.")
                self.use_gpt4 = False

        self.model = model

    def generate_retrieve_label(
        self,
        question: str,
        answer: str,
    ) -> str:
        """
        Generate Retrieve token label.

        Args:
            question: Question text
            answer: Answer text

        Returns:
            Retrieve token value
        """
        if self.use_local_llm:
            return self._local_llm_label(
                prompt=GPT4_PROMPTS['retrieve'].format(
                    question=question,
                    answer=answer,
                ),
                valid_tokens=RetrieveToken.get_all_tokens(),
            )
        elif self.use_gpt4:
            return self._gpt4_label(
                prompt=GPT4_PROMPTS['retrieve'].format(
                    question=question,
                    answer=answer,
                ),
                valid_tokens=RetrieveToken.get_all_tokens(),
            )
        else:
            # Rule-based: Assume legal questions need retrieval
            legal_keywords = [
                'law', 'legal', 'court', 'statute', 'case', 'rule',
                'negligence', 'tort', 'contract', 'liability', 'damages'
            ]
            question_lower = question.lower()

            if any(keyword in question_lower for keyword in legal_keywords):
                return RetrieveToken.YES.value
            else:
                return RetrieveToken.NO.value

    def generate_isrel_label(
        self,
        question: str,
        passage: str,
    ) -> str:
        """
        Generate ISREL token label.

        Args:
            question: Question text
            passage: Retrieved passage

        Returns:
            ISREL token value
        """
        if self.use_local_llm:
            return self._local_llm_label(
                prompt=GPT4_PROMPTS['isrel'].format(
                    question=question,
                    passage=passage,
                ),
                valid_tokens=ISRELToken.get_all_tokens(),
            )
        elif self.use_gpt4:
            return self._gpt4_label(
                prompt=GPT4_PROMPTS['isrel'].format(
                    question=question,
                    passage=passage,
                ),
                valid_tokens=ISRELToken.get_all_tokens(),
            )
        else:
            # Rule-based: Check keyword overlap
            question_words = set(question.lower().split())
            passage_words = set(passage.lower().split())

            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
            question_words = question_words - stop_words
            passage_words = passage_words - stop_words

            # Calculate overlap
            overlap = len(question_words & passage_words)
            overlap_ratio = overlap / len(question_words) if question_words else 0

            if overlap_ratio > 0.3:
                return ISRELToken.RELEVANT.value
            else:
                return ISRELToken.IRRELEVANT.value

    def generate_issup_label(
        self,
        question: str,
        passage: str,
        answer: str,
    ) -> str:
        """
        Generate ISSUP token label.

        Args:
            question: Question text
            passage: Retrieved passage
            answer: Answer text

        Returns:
            ISSUP token value
        """
        if self.use_local_llm:
            return self._local_llm_label(
                prompt=GPT4_PROMPTS['issup'].format(
                    question=question,
                    passage=passage,
                    answer=answer,
                ),
                valid_tokens=ISSUPToken.get_all_tokens(),
            )
        elif self.use_gpt4:
            return self._gpt4_label(
                prompt=GPT4_PROMPTS['issup'].format(
                    question=question,
                    passage=passage,
                    answer=answer,
                ),
                valid_tokens=ISSUPToken.get_all_tokens(),
            )
        else:
            # Rule-based: Check if answer concepts appear in passage
            answer_words = set(answer.lower().split())
            passage_words = set(passage.lower().split())

            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
            answer_words = answer_words - stop_words
            passage_words = passage_words - stop_words

            # Calculate support
            supported = len(answer_words & passage_words)
            support_ratio = supported / len(answer_words) if answer_words else 0

            if support_ratio > 0.7:
                return ISSUPToken.FULLY_SUPPORTED.value
            elif support_ratio > 0.3:
                return ISSUPToken.PARTIALLY_SUPPORTED.value
            else:
                return ISSUPToken.NO_SUPPORT.value

    def generate_isuse_label(
        self,
        question: str,
        answer: str,
    ) -> str:
        """
        Generate ISUSE token label.

        Args:
            question: Question text
            answer: Answer text

        Returns:
            ISUSE token value
        """
        if self.use_local_llm:
            return self._local_llm_label(
                prompt=GPT4_PROMPTS['isuse'].format(
                    question=question,
                    answer=answer,
                ),
                valid_tokens=ISUSEToken.get_all_tokens(),
            )
        elif self.use_gpt4:
            return self._gpt4_label(
                prompt=GPT4_PROMPTS['isuse'].format(
                    question=question,
                    answer=answer,
                ),
                valid_tokens=ISUSEToken.get_all_tokens(),
            )
        else:
            # Rule-based: Based on answer length and completeness
            answer_length = len(answer.split())

            if answer_length > 50:
                return ISUSEToken.UTILITY_5.value
            elif answer_length > 30:
                return ISUSEToken.UTILITY_4.value
            elif answer_length > 15:
                return ISUSEToken.UTILITY_3.value
            elif answer_length > 5:
                return ISUSEToken.UTILITY_2.value
            else:
                return ISUSEToken.UTILITY_1.value

    def generate_all_labels(
        self,
        question: str,
        passage: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate all applicable reflection token labels.

        Args:
            question: Question text
            passage: Retrieved passage (optional)
            answer: Answer text (optional)

        Returns:
            Dictionary with all labels
        """
        labels = {}

        if answer:
            labels['retrieve'] = self.generate_retrieve_label(question, answer)
            labels['isuse'] = self.generate_isuse_label(question, answer)

        if passage:
            labels['isrel'] = self.generate_isrel_label(question, passage)

        if passage and answer:
            labels['issup'] = self.generate_issup_label(question, passage, answer)

        return labels

    def _local_llm_label(
        self,
        prompt: str,
        valid_tokens: List[str],
    ) -> str:
        """
        Get label from local LLM (Qwen).

        Args:
            prompt: Prompt for the model
            valid_tokens: List of valid token values

        Returns:
            Token value
        """
        try:
            # Format prompt for Qwen
            messages = [
                {"role": "system", "content": "You are a helpful assistant that labels data for training Self-RAG models. Respond with ONLY the exact token requested, nothing else."},
                {"role": "user", "content": prompt}
            ]

            text = self.local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.local_tokenizer([text], return_tensors="pt")

            # Move to device
            if hasattr(self, 'device'):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with constraints
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.local_tokenizer.eos_token_id,
                )

            # Decode response
            generated = self.local_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Extract token from response
            for token in valid_tokens:
                if token in generated:
                    return token

            # Default to first token if none found
            print(f"Warning: Could not extract token from: {generated[:100]}")
            return valid_tokens[0]

        except Exception as e:
            print(f"Error calling local LLM: {e}")
            # Fallback to first valid token
            return valid_tokens[0]

    def _gpt4_label(
        self,
        prompt: str,
        valid_tokens: List[str],
    ) -> str:
        """
        Get label from GPT-4.

        Args:
            prompt: Prompt for GPT-4
            valid_tokens: List of valid token values

        Returns:
            Token value
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that labels data for training."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=50,
            )

            generated = response.choices[0].message.content.strip()

            # Extract token from response
            for token in valid_tokens:
                if token in generated:
                    return token

            # Default to first token if none found
            return valid_tokens[0]

        except Exception as e:
            print(f"Error calling GPT-4: {e}")
            # Fallback to first valid token
            return valid_tokens[0]


def process_dataset(
    input_file: str,
    output_dir: str,
    use_local_llm: bool = True,
    use_gpt4: bool = False,
    num_samples: Optional[int] = None,
    local_model: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """
    Process a dataset to generate reflection token labels.

    Args:
        input_file: Path to input JSON file with QA examples
        output_dir: Directory to save labeled data
        use_local_llm: Whether to use local LLM (Qwen) for labeling
        use_gpt4: Whether to use GPT-4 for labeling
        num_samples: Number of samples to process (None for all)
        local_model: Local LLM model name
    """
    # Load dataset
    with open(input_file, 'r') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    # Initialize generator
    generator = LabelGenerator(
        use_local_llm=use_local_llm,
        use_gpt4=use_gpt4,
        local_model=local_model,
    )

    # Process examples
    labeled_data = []

    for example in tqdm(data, desc="Generating labels"):
        question = example['question']
        passage = example.get('passage', '')
        answer = example.get('answer', '')

        # Generate labels
        labels = generator.generate_all_labels(question, passage, answer)

        # Create labeled example
        labeled_example = {
            'question': question,
            'passage': passage,
            'answer': answer,
            'reflection_tokens': labels,
        }

        labeled_data.append(labeled_example)

    # Save labeled data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'labeled_data.json')

    with open(output_file, 'w') as f:
        json.dump(labeled_data, f, indent=2)

    print(f"\nLabeled {len(labeled_data)} examples")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate reflection token labels")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with QA examples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory for labeled data",
    )
    parser.add_argument(
        "--use-local-llm",
        action="store_true",
        default=True,
        help="Use local LLM (Qwen) for labeling (default: True)",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Local LLM model name",
    )
    parser.add_argument(
        "--use-gpt4",
        action="store_true",
        help="Use GPT-4 for labeling (requires API key, overrides --use-local-llm)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (None for all)",
    )

    args = parser.parse_args()

    process_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        use_local_llm=args.use_local_llm and not args.use_gpt4,
        use_gpt4=args.use_gpt4,
        num_samples=args.num_samples,
        local_model=args.local_model,
    )
