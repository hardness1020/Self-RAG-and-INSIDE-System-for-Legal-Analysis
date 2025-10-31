"""
Label Generation Module

Generates reflection token labels for training data using:
1. GPT-4 prompting (if API available)
2. Rule-based heuristics (fallback)

Creates training data for both critic and generator models.
"""

from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
import yaml
from tqdm import tqdm

# Optional: OpenAI for GPT-4 labeling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
        use_gpt4: bool = False,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
    ):
        """
        Initialize label generator.

        Args:
            use_gpt4: Whether to use GPT-4 for labeling
            openai_api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.use_gpt4 = use_gpt4 and OPENAI_AVAILABLE

        if self.use_gpt4:
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
        if self.use_gpt4:
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
        if self.use_gpt4:
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
        if self.use_gpt4:
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
        if self.use_gpt4:
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
    use_gpt4: bool = False,
    num_samples: Optional[int] = None,
):
    """
    Process a dataset to generate reflection token labels.

    Args:
        input_file: Path to input JSON file with QA examples
        output_dir: Directory to save labeled data
        use_gpt4: Whether to use GPT-4 for labeling
        num_samples: Number of samples to process (None for all)
    """
    # Load dataset
    with open(input_file, 'r') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    # Initialize generator
    generator = LabelGenerator(use_gpt4=use_gpt4)

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
        "--use-gpt4",
        action="store_true",
        help="Use GPT-4 for labeling (requires API key)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process",
    )

    args = parser.parse_args()

    process_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        use_gpt4=args.use_gpt4,
        num_samples=args.num_samples,
    )
