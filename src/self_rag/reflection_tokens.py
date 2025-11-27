"""
Reflection Tokens Module

Defines the five types of reflection tokens used in Self-RAG + INSIDE:
1. Retrieve - Decides when to retrieve
2. ISREL (Is Relevant) - Evaluates relevance of retrieved passages
3. ISSUP (Is Supported) - Evaluates factual support
4. ISUSE (Is Useful) - Evaluates overall utility
5. INTENT - Detects query intent for adaptive retrieval (INSIDE extension)

Based on the Self-RAG paper by Asai et al. (2023) and INSIDE integration.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class RetrieveToken(Enum):
    """
    Retrieve decision tokens.

    Determines whether to retrieve external knowledge.
    """
    YES = "[Retrieval]"  # Model uses [Retrieval] not [Retrieve]
    NO = "[No Retrieval]"
    CONTINUE = "[Continue to Use Evidence]"

    @staticmethod
    def get_all_tokens() -> List[str]:
        return [token.value for token in RetrieveToken]

    @staticmethod
    def get_description() -> str:
        return """
        Retrieve Token:
        - [Retrieve]: Need to retrieve external knowledge
        - [No Retrieval]: Can answer without retrieval
        - [Continue to Use Evidence]: Continue using previously retrieved evidence
        """


class ISRELToken(Enum):
    """
    Relevance evaluation tokens.

    Evaluates whether retrieved passage provides useful information.
    """
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"

    @staticmethod
    def get_all_tokens() -> List[str]:
        return [token.value for token in ISRELToken]

    @staticmethod
    def get_description() -> str:
        return """
        ISREL Token (Is Relevant):
        - [Relevant]: Retrieved passage provides useful information for the query
        - [Irrelevant]: Retrieved passage does not provide useful information
        """


class ISSUPToken(Enum):
    """
    Support evaluation tokens.

    Evaluates whether the response is supported by retrieved evidence.
    """
    FULLY_SUPPORTED = "[Fully supported]"  # lowercase 's' to match model
    PARTIALLY_SUPPORTED = "[Partially supported]"  # lowercase 's' to match model
    NO_SUPPORT = "[No support / Contradictory]"  # match model token

    @staticmethod
    def get_all_tokens() -> List[str]:
        return [token.value for token in ISSUPToken]

    @staticmethod
    def get_description() -> str:
        return """
        ISSUP Token (Is Supported):
        - [Fully Supported]: All claims in response are supported by evidence
        - [Partially Supported]: Some claims are supported, others are not
        - [No Support]: Response contradicts evidence or is not supported
        """


class ISUSEToken(Enum):
    """
    Utility evaluation tokens.

    Evaluates overall usefulness of the response (5-point scale).
    """
    UTILITY_5 = "[Utility:5]"  # Excellent
    UTILITY_4 = "[Utility:4]"  # Good
    UTILITY_3 = "[Utility:3]"  # Acceptable
    UTILITY_2 = "[Utility:2]"  # Poor
    UTILITY_1 = "[Utility:1]"  # Very Poor

    @staticmethod
    def get_all_tokens() -> List[str]:
        return [token.value for token in ISUSEToken]

    @staticmethod
    def get_score(token: 'ISUSEToken') -> int:
        """Extract numeric score from utility token."""
        return int(token.name.split('_')[1])

    @staticmethod
    def get_description() -> str:
        return """
        ISUSE Token (Is Useful):
        - [Utility:5]: Excellent - Fully answers query with complete information
        - [Utility:4]: Good - Mostly answers query with minor gaps
        - [Utility:3]: Acceptable - Provides some useful information
        - [Utility:2]: Poor - Minimal useful information
        - [Utility:1]: Very Poor - Not useful for answering query
        """


class INTENTToken(Enum):
    """
    Intent detection tokens (INSIDE extension).

    NOTE: Intent is detected by IntentDetector BEFORE retrieval, NOT predicted
    by Self-RAG generator/critic. This enum is used by INSIDE pipeline only.

    Identifies query intent for adaptive retrieval strategies.
    """
    FACTUAL = "[Intent:Factual]"  # Seeking specific facts
    EXPLORATORY = "[Intent:Exploratory]"  # Broad exploration
    COMPARATIVE = "[Intent:Comparative]"  # Comparing concepts
    PROCEDURAL = "[Intent:Procedural]"  # How-to queries
    UNKNOWN = "[Intent:Unknown]"  # Cannot determine

    @staticmethod
    def get_all_tokens() -> List[str]:
        return [token.value for token in INTENTToken]

    @staticmethod
    def get_description() -> str:
        return """
        INTENT Token (Query Intent):
        - [Intent:Factual]: Seeking specific facts or definitions
        - [Intent:Exploratory]: Broad exploration of a topic
        - [Intent:Comparative]: Comparing cases, laws, or concepts
        - [Intent:Procedural]: How-to or process-oriented queries
        - [Intent:Unknown]: Cannot determine query intent
        """


class PassageMarker:
    """
    Passage marker tokens for Self-RAG training.

    Per the Self-RAG paper (Section 3.2.2), retrieved passages are surrounded
    by <p> and </p> markers and masked from loss calculation during training.
    """
    START = "<p>"  # Start of retrieved passage
    END = "</p>"   # End of retrieved passage

    @staticmethod
    def get_all_tokens() -> List[str]:
        return [PassageMarker.START, PassageMarker.END]

    @staticmethod
    def get_description() -> str:
        return """
        Passage Markers:
        - <p>: Start of retrieved passage (content masked from loss)
        - </p>: End of retrieved passage
        """


@dataclass
class ReflectionAnnotation:
    """
    Complete reflection annotation for a text segment.
    """
    retrieve: Optional[RetrieveToken] = None
    isrel: Optional[ISRELToken] = None
    issup: Optional[ISSUPToken] = None
    isuse: Optional[ISUSEToken] = None
    intent: Optional[INTENTToken] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        return {
            'retrieve': self.retrieve.value if self.retrieve else None,
            'isrel': self.isrel.value if self.isrel else None,
            'issup': self.issup.value if self.issup else None,
            'isuse': self.isuse.value if self.isuse else None,
            'intent': self.intent.value if self.intent else None,
        }

    def to_string(self) -> str:
        """Convert to string representation for text generation."""
        tokens = []
        if self.intent:
            tokens.append(self.intent.value)
        if self.retrieve:
            tokens.append(self.retrieve.value)
        if self.isrel:
            tokens.append(self.isrel.value)
        if self.issup:
            tokens.append(self.issup.value)
        if self.isuse:
            tokens.append(self.isuse.value)
        return " ".join(tokens)


class ReflectionTokenizer:
    """
    Helper class for working with reflection tokens.
    """

    @staticmethod
    def get_all_special_tokens() -> List[str]:
        """Get list of all reflection token strings including passage markers."""
        all_tokens = []
        all_tokens.extend(INTENTToken.get_all_tokens())
        all_tokens.extend(RetrieveToken.get_all_tokens())
        all_tokens.extend(ISRELToken.get_all_tokens())
        all_tokens.extend(ISSUPToken.get_all_tokens())
        all_tokens.extend(ISUSEToken.get_all_tokens())
        all_tokens.extend(PassageMarker.get_all_tokens())  # Add passage markers
        return all_tokens

    @staticmethod
    def get_token_descriptions() -> str:
        """Get descriptions of all token types."""
        descriptions = []
        descriptions.append(INTENTToken.get_description())
        descriptions.append(RetrieveToken.get_description())
        descriptions.append(ISRELToken.get_description())
        descriptions.append(ISSUPToken.get_description())
        descriptions.append(ISUSEToken.get_description())
        return "\n".join(descriptions)

    @staticmethod
    def parse_retrieve_token(token_str: str) -> Optional[RetrieveToken]:
        """Parse string to RetrieveToken."""
        for token in RetrieveToken:
            if token.value == token_str:
                return token
        return None

    @staticmethod
    def parse_isrel_token(token_str: str) -> Optional[ISRELToken]:
        """Parse string to ISRELToken."""
        for token in ISRELToken:
            if token.value == token_str:
                return token
        return None

    @staticmethod
    def parse_issup_token(token_str: str) -> Optional[ISSUPToken]:
        """Parse string to ISSUPToken."""
        for token in ISSUPToken:
            if token.value == token_str:
                return token
        return None

    @staticmethod
    def parse_isuse_token(token_str: str) -> Optional[ISUSEToken]:
        """Parse string to ISUSEToken."""
        for token in ISUSEToken:
            if token.value == token_str:
                return token
        return None

    @staticmethod
    def parse_intent_token(token_str: str) -> Optional[INTENTToken]:
        """Parse string to INTENTToken."""
        for token in INTENTToken:
            if token.value == token_str:
                return token
        return None

    @staticmethod
    def extract_tokens_from_text(text: str) -> ReflectionAnnotation:
        """
        Extract reflection tokens from generated text.

        Args:
            text: Text containing reflection tokens

        Returns:
            ReflectionAnnotation with extracted tokens
        """
        annotation = ReflectionAnnotation()

        # Extract intent token first
        for token in INTENTToken:
            if token.value in text:
                annotation.intent = token
                break

        # Extract each token type
        for token in RetrieveToken:
            if token.value in text:
                annotation.retrieve = token
                break

        for token in ISRELToken:
            if token.value in text:
                annotation.isrel = token
                break

        for token in ISSUPToken:
            if token.value in text:
                annotation.issup = token
                break

        for token in ISUSEToken:
            if token.value in text:
                annotation.isuse = token
                break

        return annotation


# Prompts for GPT-4 to generate reflection token labels
GPT4_PROMPTS = {
    'intent': """
Given a question, determine the query intent to guide retrieval strategy.

Output one of:
- [Intent:Factual]: Seeking specific facts or definitions
- [Intent:Exploratory]: Broad exploration of a topic
- [Intent:Comparative]: Comparing cases, laws, or concepts
- [Intent:Procedural]: How-to or process-oriented queries

Question: {question}

Decision:""",

    'retrieve': """
Given a question and its answer, determine if external knowledge retrieval would be helpful.

Output one of:
- [Retrieve]: External knowledge would significantly help answer this question
- [No Retrieval]: Can answer confidently without external knowledge

Question: {question}
Answer: {answer}

Decision:""",

    'isrel': """
Given a question and a retrieved passage, determine if the passage is relevant.

Output one of:
- [Relevant]: The passage provides useful information for answering the question
- [Irrelevant]: The passage does not provide useful information

Question: {question}
Passage: {passage}

Decision:""",

    'issup': """
Given a question, a passage, and an answer, determine if the answer is supported by the passage.

Output one of:
- [Fully Supported]: All claims in the answer are directly supported by the passage
- [Partially Supported]: Some claims are supported, but others are not or go beyond the passage
- [No Support]: The answer contradicts the passage or is not supported at all

Question: {question}
Passage: {passage}
Answer: {answer}

Decision:""",

    'isuse': """
Given a question and an answer, rate the utility of the answer on a scale of 1-5.

Output one of:
- [Utility:5]: Excellent - Completely answers the question with all necessary details
- [Utility:4]: Good - Answers the question well with minor details missing
- [Utility:3]: Acceptable - Provides useful information but incomplete
- [Utility:2]: Poor - Limited useful information
- [Utility:1]: Very Poor - Does not help answer the question

Question: {question}
Answer: {answer}

Rating:""",
}


if __name__ == "__main__":
    # Example usage
    print("Reflection Token System\n")
    print("=" * 80)

    # Print all token types
    print("\n1. RETRIEVE TOKENS:")
    print(RetrieveToken.get_description())

    print("\n2. ISREL TOKENS:")
    print(ISRELToken.get_description())

    print("\n3. ISSUP TOKENS:")
    print(ISSUPToken.get_description())

    print("\n4. ISUSE TOKENS:")
    print(ISUSEToken.get_description())

    # Get all special tokens
    print("\n" + "=" * 80)
    print("\nAll special tokens for tokenizer:")
    all_tokens = ReflectionTokenizer.get_all_special_tokens()
    for token in all_tokens:
        print(f"  {token}")

    # Example annotation
    print("\n" + "=" * 80)
    print("\nExample annotation:")
    annotation = ReflectionAnnotation(
        retrieve=RetrieveToken.YES,
        isrel=ISRELToken.RELEVANT,
        issup=ISSUPToken.FULLY_SUPPORTED,
        isuse=ISUSEToken.UTILITY_5,
    )
    print(f"As dict: {annotation.to_dict()}")
    print(f"As string: {annotation.to_string()}")

    # Example extraction
    print("\n" + "=" * 80)
    print("\nExample token extraction:")
    text = "Let me retrieve information. [Retrieve] The passage is [Relevant] and [Fully Supported] with [Utility:5]"
    extracted = ReflectionTokenizer.extract_tokens_from_text(text)
    print(f"Input text: {text}")
    print(f"Extracted: {extracted.to_dict()}")
