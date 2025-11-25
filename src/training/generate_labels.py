"""
Label Generation Module

Generates Self-RAG reflection token labels for training data using:
1. OpenAI GPT-5.1 (Primary) - Best quality with reasoning
2. Local LLM (Qwen2.5-7B-Instruct) - Fallback, free and open-source
3. Rule-based heuristics (Last resort fallback)

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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads OPENAI_API_KEY and other env vars from .env
except ImportError:
    pass  # python-dotenv not installed, env vars must be set manually

# Optional: OpenAI for GPT-5.1 labeling
try:
    from openai import OpenAI
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
        use_openai: bool = True,
        use_local_llm: bool = True,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-5.1",
        reasoning_effort: str = "auto",
        local_model: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
    ):
        """
        Initialize label generator.

        Args:
            use_openai: Whether to use OpenAI API (GPT-5.1) for labeling (primary)
            use_local_llm: Whether to use local LLM (Qwen) as fallback
            openai_api_key: OpenAI API key (reads from OPENAI_API_KEY env if not provided)
            model: OpenAI model to use (default: gpt-5.1)
            reasoning_effort: Reasoning effort level for GPT-5.1 ("auto", "none", "low", "medium", "high")
            local_model: Local LLM model name
            device: Device for local LLM (None for auto-detect)
        """
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.use_local_llm = use_local_llm and TRANSFORMERS_AVAILABLE
        self.openai_client = None
        self.local_model = None
        self.local_tokenizer = None
        self.model = model
        self.reasoning_effort = reasoning_effort

        # Priority: OpenAI GPT-5.1 > Local LLM > Rule-based
        if self.use_openai:
            try:
                if openai_api_key:
                    self.openai_client = OpenAI(api_key=openai_api_key)
                elif 'OPENAI_API_KEY' in os.environ:
                    self.openai_client = OpenAI()  # Uses OPENAI_API_KEY env var
                else:
                    print("Warning: OpenAI requested but no API key found. Will fall back to local LLM.")
                    self.use_openai = False

                if self.use_openai:
                    print(f"✓ Primary: OpenAI {model} (reasoning_effort={reasoning_effort})")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                print("Falling back to local LLM.")
                self.use_openai = False

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
                if self.use_openai:
                    print(f"✓ Fallback: Qwen {local_model.split('/')[-1]} (Local)")
                else:
                    print(f"✓ Primary: Qwen {local_model.split('/')[-1]} (Local)")

            except Exception as e:
                print(f"Error loading local LLM: {e}")
                if not self.use_openai:
                    print("Falling back to rule-based approach.")
                self.use_local_llm = False

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
        if self.use_openai:
            return self._openai_label(
                prompt=GPT4_PROMPTS['retrieve'].format(
                    question=question,
                    answer=answer,
                ),
                valid_tokens=RetrieveToken.get_all_tokens(),
            )
        elif self.use_local_llm:
            return self._local_llm_label(
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
        if self.use_openai:
            return self._openai_label(
                prompt=GPT4_PROMPTS['isrel'].format(
                    question=question,
                    passage=passage,
                ),
                valid_tokens=ISRELToken.get_all_tokens(),
            )
        elif self.use_local_llm:
            return self._local_llm_label(
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
        if self.use_openai:
            return self._openai_label(
                prompt=GPT4_PROMPTS['issup'].format(
                    question=question,
                    passage=passage,
                    answer=answer,
                ),
                valid_tokens=ISSUPToken.get_all_tokens(),
            )
        elif self.use_local_llm:
            return self._local_llm_label(
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
        if self.use_openai:
            return self._openai_label(
                prompt=GPT4_PROMPTS['isuse'].format(
                    question=question,
                    answer=answer,
                ),
                valid_tokens=ISUSEToken.get_all_tokens(),
            )
        elif self.use_local_llm:
            return self._local_llm_label(
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
                    max_new_tokens=1000,  # Increased for complete label generation
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

    def _openai_label(
        self,
        prompt: str,
        valid_tokens: List[str],
    ) -> str:
        """
        Get label from OpenAI (GPT-5.1).

        Args:
            prompt: Prompt for the model
            valid_tokens: List of valid token values

        Returns:
            Token value
        """
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that labels data for training Self-RAG models. Respond with ONLY the exact token requested, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
            }

            # GPT-5 models use max_completion_tokens, older models use max_tokens
            if self.model.startswith("gpt-5"):
                request_params["max_completion_tokens"] = 1000  # GPT-5.1 parameter
                # Add reasoning_effort for GPT-5.1 if not 'auto'
                if self.reasoning_effort != "auto":
                    request_params["reasoning_effort"] = self.reasoning_effort
            else:
                request_params["max_tokens"] = 1000  # GPT-4 and earlier

            response = self.openai_client.chat.completions.create(**request_params)
            generated = response.choices[0].message.content.strip()

            # Extract token from response
            for token in valid_tokens:
                if token in generated:
                    return token

            # Default to first token if none found
            print(f"Warning: Could not extract token from OpenAI response: {generated[:100]}")
            return valid_tokens[0]

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fall back to local LLM if available
            if self.use_local_llm:
                print("Falling back to local LLM...")
                return self._local_llm_label(prompt, valid_tokens)
            # Otherwise fallback to first valid token
            return valid_tokens[0]


def process_dataset(
    input_file: str,
    output_dir: str,
    use_openai: bool = True,
    use_local_llm: bool = True,
    num_samples: Optional[int] = None,
    openai_model: str = "gpt-5.1",
    reasoning_effort: str = "auto",
    local_model: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """
    Process a dataset to generate reflection token labels.

    Args:
        input_file: Path to input JSON file with QA examples
        output_dir: Directory to save labeled data
        use_openai: Whether to use OpenAI API (GPT-5.1) as primary
        use_local_llm: Whether to use local LLM (Qwen) as fallback
        num_samples: Number of samples to process (None for all)
        openai_model: OpenAI model to use
        reasoning_effort: Reasoning effort level for GPT-5.1
        local_model: Local LLM model name
    """
    # Load dataset
    with open(input_file, 'r') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    # Initialize generator
    generator = LabelGenerator(
        use_openai=use_openai,
        use_local_llm=use_local_llm,
        model=openai_model,
        reasoning_effort=reasoning_effort,
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
        "--use-openai",
        action="store_true",
        default=True,
        help="Use OpenAI API (GPT-5.1) as primary (default: True, requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Disable OpenAI and use only local LLM",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5.1",
        help="OpenAI model to use (default: gpt-5.1)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="auto",
        choices=["auto", "none", "low", "medium", "high"],
        help="Reasoning effort for GPT-5.1 (default: auto)",
    )
    parser.add_argument(
        "--use-local-llm",
        action="store_true",
        default=True,
        help="Use local LLM (Qwen) as fallback (default: True)",
    )
    parser.add_argument(
        "--no-local-llm",
        action="store_true",
        help="Disable local LLM fallback",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Local LLM model name",
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
        use_openai=args.use_openai and not args.no_openai,
        use_local_llm=args.use_local_llm and not args.no_local_llm,
        num_samples=args.num_samples,
        openai_model=args.openai_model,
        reasoning_effort=args.reasoning_effort,
        local_model=args.local_model,
    )
