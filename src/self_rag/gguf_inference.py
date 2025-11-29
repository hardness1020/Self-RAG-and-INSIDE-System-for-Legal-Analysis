# Suppress Metal kernel init messages BEFORE any imports
# Must be at absolute top to take effect before llama_cpp loads
import os
os.environ["GGML_METAL_LOG_LEVEL"] = "0"

"""
Self-RAG GGUF Inference Module

Provides inference using pre-trained Self-RAG models converted to GGUF format
for efficient execution on Mac with Metal acceleration via llama.cpp.

Includes INSIDE (INternal States for hallucInation DEtection) integration
via multi-generation EigenScore computation using external embeddings.

OPTIMIZATION: Uses a single persistent model with reset() between generations
instead of load-delete-reload pattern. This reduces model loads from ~9,300
to just 1 for full evaluation (776 queries × 4 methods).

The retrieval check uses token sampling (like the official Self-RAG fallback)
instead of logits_all=True which caused memory overflow (-3 errors) on llama-cpp.
"""

import re
import math
import gc
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.self_rag.reflection_tokens import ReflectionTokenizer

# INSIDE constants
EIGENSCORE_THRESHOLD = -2.0  # Calibrated for LegalBench (No-RAG≈-1, RAG≈-3)
DEFAULT_NUM_GENERATIONS = 10  # K generations for EigenScore computation


def compute_eigenscore(
    answers: List[str],
    embedding_model: Any,
    alpha: float = 0.001,
    threshold: float = EIGENSCORE_THRESHOLD,
) -> Tuple[float, bool]:
    """
    Compute EigenScore for any list of answer strings (INSIDE paper, ICLR 2024).

    This is a STANDALONE function that works with ANY generation method.
    EigenScore measures semantic consistency across multiple generations.
    Higher EigenScore = more diverse responses = higher hallucination risk.

    Formula (per paper Section 3.1):
    - Z ∈ R^(d×K): each column is a sentence embedding
    - Σ = Z^T · J_d · Z: K×K covariance matrix (centered)
    - EigenScore = (1/K) * Σ log(λ_i + α)

    Args:
        answers: List of K generated answer strings
        embedding_model: Encoder with encode() method (e.g., sentence-transformers)
        alpha: Regularization term (default 0.001 per INSIDE paper)
        threshold: Threshold for hallucination detection (default -5.0)

    Returns:
        Tuple of (eigenscore, hallucination_detected)
    """
    if len(answers) < 2:
        return 0.0, False

    K = len(answers)

    # Embed all answers using external encoder
    embeddings = []
    for answer in answers:
        if answer:
            emb = embedding_model.encode(answer)
            embeddings.append(np.array(emb).squeeze())

    if len(embeddings) < 2:
        return 0.0, False

    # Z: (d, K) - each column is a sentence embedding
    Z = np.column_stack(embeddings)

    # Center embeddings (per paper Eq. 4: J_d · z = z - mean(z))
    Z_centered = Z - Z.mean(axis=0, keepdims=True)

    # Compute K×K covariance matrix: Σ = Z_centered^T · Z_centered
    Sigma = Z_centered.T @ Z_centered

    # Add regularization to ensure full rank (per paper Eq. 5)
    Sigma_reg = Sigma + alpha * np.eye(K)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(Sigma_reg)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)

    # EigenScore = (1/K) * Σ log(λ_i) (per paper Eq. 6)
    eigenscore = float(np.mean(np.log(eigenvalues)))

    # Detect hallucination
    hallucination_detected = eigenscore > threshold

    return eigenscore, hallucination_detected


@dataclass
class SelfRAGOutput:
    """Structured output from Self-RAG inference."""
    answer: str
    retrieve: Optional[str] = None
    isrel: Optional[str] = None
    issup: Optional[str] = None
    isuse: Optional[str] = None
    raw_output: str = ""
    retrieval_score: Optional[float] = None  # Probability score for retrieval decision


@dataclass
class SelfRAGOutputWithEigenScore(SelfRAGOutput):
    """Extended output with INSIDE EigenScore metrics for hallucination detection.

    INSIDE uses multi-generation EigenScore computation to detect hallucinations
    by measuring semantic consistency across K generations. Higher EigenScore
    indicates higher entropy (less consistency) suggesting potential hallucination.
    """
    eigenscore: Optional[float] = None
    hallucination_detected: Optional[bool] = None
    num_generations: int = 1
    all_generations: Optional[List['SelfRAGOutput']] = field(default=None)


class SelfRAGGGUFInference:
    """
    Self-RAG inference using GGUF model with llama.cpp.

    Integrates with existing LegalRetriever for passage retrieval.
    Designed for Mac M-series chips with Metal acceleration.

    OPTIMIZATION: Uses a single persistent model with reset() between
    generations instead of load-delete-reload pattern. The retrieval check
    uses token sampling (like official Self-RAG fallback) instead of
    logits_all=True which caused memory overflow (-3 errors).

    Example usage:
        >>> inference = SelfRAGGGUFInference("models/selfrag_llama2_7b.Q4_K_M.gguf")
        >>> result = inference.generate("What are the elements of negligence?")
        >>> print(result.answer)
        >>> print(result.isuse)  # e.g., "[Utility:5]"
    """

    # Token patterns for extraction (regex) - matches selfrag_llama2_7b model tokens
    TOKEN_PATTERNS = {
        'retrieve': r'\[(Retrieval|No Retrieval|Continue to Use Evidence)\]',
        'isrel': r'\[(Relevant|Irrelevant)\]',
        'issup': r'\[(Fully supported|Partially supported|No support / Contradictory)\]',
        'isuse': r'\[Utility:([1-5])\]',
    }

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,  # Default 2048 for 16GB Mac (4096 may cause OOM)
        n_gpu_layers: int = -1,  # -1 = use all (Metal)
        verbose: bool = False,
    ):
        """
        Initialize GGUF model and load it persistently.

        OPTIMIZATION: Single model is loaded ONCE at init and reused for ALL
        operations including retrieval check. Uses reset() to clear KV cache
        between generations instead of reloading.

        The retrieval check uses token sampling (generate 1 token, check if
        it's [Retrieval] or [No Retrieval]) instead of logits_all=True which
        caused memory overflow (-3 errors) on long prompts.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (default 2048)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all for Metal)
            verbose: Whether to show llama.cpp verbose output
        """
        try:
            from llama_cpp import Llama
            self._Llama = Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install with: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
            )

        # Single model config (no logits_all=True needed - we use token sampling)
        self._gen_config = {
            'model_path': model_path,
            'n_ctx': n_ctx,
            'n_batch': 512,  # Batch size for prompt processing
            'n_gpu_layers': n_gpu_layers,
            'verbose': verbose,
            'logits_all': False,  # KEY: Always False to prevent -3 overflow errors
            'embedding': False,
        }

        self.model_path = model_path

        # Load single persistent model
        print(f"Loading Self-RAG model: {model_path}")
        self._llm = self._Llama(**self._gen_config)
        print("✓ Model loaded (persistent, single instance for all operations)")

    def _get_gen_model(self):
        """
        Get the persistent generation model with reset KV cache.

        Returns the persistent self._llm instance after calling reset()
        to clear the KV cache from previous generations.

        Returns:
            Llama instance with logits_all=False (persistent)
        """
        self._llm.reset()  # Clear KV cache from previous generation
        return self._llm

    def _format_prompt(
        self,
        question: str,
        passage: Optional[str] = None,
        no_retrieval: bool = False,
    ) -> str:
        """
        Format prompt in Self-RAG expected format.

        Per Self-RAG paper, HuggingFace model card, and official implementation:
        - Format: ### Instruction:\n{question}\n\n### Response:\n
        - With passage: add [Retrieval]<paragraph>{passage}</paragraph>
        - Without passage: add [No Retrieval] to guide model (per official impl)

        Args:
            question: The input question
            passage: Optional retrieved passage to include
            no_retrieval: If True, append [No Retrieval] token to guide model
                         to NOT generate reflection tokens (per official impl)

        Returns:
            Formatted prompt string
        """
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"

        if passage:
            # Truncate passage if too long
            max_passage_len = 1000
            if len(passage) > max_passage_len:
                passage = passage[:max_passage_len] + "..."
            prompt += f"[Retrieval]<paragraph>{passage}</paragraph>"
        elif no_retrieval:
            # Guide model to NOT generate reflection tokens (per official impl)
            # See: github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py
            prompt += "[No Retrieval]"

        return prompt

    def _check_retrieval_needed(
        self,
        question: str,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Check if retrieval is needed using token sampling (llama-cpp compatible).

        Unlike vLLM which can get probability of any token via logprobs=vocab_size,
        llama-cpp only returns top-K logprobs. We use the official Self-RAG fallback:
        generate 1 token and check if it's [Retrieval] or [No Retrieval].

        Per official Self-RAG implementation:
        - If threshold provided: compute P([Retrieval]) / (P([Retrieval]) + P([No Retrieval]))
        - Fallback: do_retrieve = "[Retrieval]" in pred

        Since llama-cpp can't reliably get [Retrieval] probability when it's not in
        top-K, we use the fallback approach: check what token was actually generated.

        Args:
            question: Input question
            threshold: Retrieval threshold (default 0.5, used for logprobs fallback)

        Returns:
            Tuple of (needs_retrieval: bool, retrieval_score: float)
        """
        # Use generation model (no separate logprobs model needed)
        llm = self._get_gen_model()

        prompt = self._format_prompt(question, passage=None)

        # Generate 1 token with greedy decoding to see retrieval decision
        # Note: Can't use logprobs with logits_all=False, so we just check the generated token
        output = llm(
            prompt,
            max_tokens=1,
            temperature=0.0,  # Greedy - deterministic
            echo=False,
        )

        generated_text = output['choices'][0]['text'].strip()

        # Check what token was actually generated
        # This is the official Self-RAG fallback approach: do_retrieve = "[Retrieval]" in pred
        if '[Retrieval]' in generated_text:
            return True, 1.0
        elif '[No Retrieval]' in generated_text:
            return False, 0.0

        # Neither token found - default to RETRIEVE (conservative for legal domain)
        # Legal questions generally benefit from retrieval
        return True, 0.5

    def _generate_with_passage(
        self,
        llm,
        question: str,
        passage: str,
        max_tokens: int,
        temperature: float,
    ) -> SelfRAGOutput:
        """
        Generate response with passage context.

        Args:
            llm: Llama model instance to use
            question: Input question
            passage: Retrieved passage to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            SelfRAGOutput with answer and reflection tokens
        """
        prompt = self._format_prompt(question, passage)
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["###", "\n\n\n"],
            echo=False,
        )
        result = self._parse_output(output['choices'][0]['text'])
        # Explicitly set retrieve since we used retrieval
        # ([Retrieval] is in the prompt, not the generated text)
        result.retrieve = "[Retrieval]"
        return result

    def _parse_output(self, text: str) -> SelfRAGOutput:
        """
        Parse reflection tokens from generated text.

        Uses ReflectionTokenizer.extract_tokens_from_text() for consistency
        with existing codebase.

        Args:
            text: Generated text containing reflection tokens

        Returns:
            SelfRAGOutput with parsed answer and tokens
        """
        # Use existing ReflectionTokenizer
        annotation = ReflectionTokenizer.extract_tokens_from_text(text)

        # Extract answer (text after tokens, before utility)
        # Remove all reflection tokens to get clean answer
        answer = text
        for pattern in self.TOKEN_PATTERNS.values():
            answer = re.sub(pattern, '', answer)
        answer = re.sub(r'<paragraph>.*?</paragraph>', '', answer, flags=re.DOTALL)
        answer = answer.strip()

        return SelfRAGOutput(
            answer=answer,
            retrieve=annotation.retrieve.value if annotation.retrieve else None,
            isrel=annotation.isrel.value if annotation.isrel else None,
            issup=annotation.issup.value if annotation.issup else None,
            isuse=annotation.isuse.value if annotation.isuse else None,
            raw_output=text,
        )

    def generate(
        self,
        question: str,
        passage: Optional[str] = None,
        retriever: Any = None,
        max_tokens: int = 512,
        temperature: float = 0.0,  # Deterministic by default
        retrieval_threshold: float = 0.5,  # Configurable threshold (like official impl)
    ) -> SelfRAGOutput:
        """
        Generate answer with adaptive retrieval per Self-RAG Algorithm 1.

        Implements the official Self-RAG approach:
        1. Generate first token to check if model wants [Retrieval] or [No Retrieval]
        2. If [Retrieval] and retriever provided: retrieve and generate with passage
        3. Otherwise: generate without retrieval

        Uses single persistent model with reset() between calls (llama-cpp compatible).
        The retrieval check uses token sampling (like official fallback) instead of
        logits_all=True which caused memory overflow on long prompts.

        Args:
            question: Input question
            passage: Optional retrieved passage (skips adaptive detection if provided)
            retriever: Optional retriever for adaptive retrieval (must have .retrieve() method)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            retrieval_threshold: Threshold for retrieval decision (default 0.5)

        Returns:
            SelfRAGOutput with answer and reflection tokens
        """
        # If passage provided, generate directly (no retrieval check needed)
        if passage is not None:
            llm = self._get_gen_model()  # Persistent, reset internally
            result = self._generate_with_passage(llm, question, passage, max_tokens, temperature)
            result.retrieval_score = 1.0  # Passage was explicitly provided
            return result

        # If no retriever provided, skip retrieval check entirely (No-RAG mode)
        if retriever is None:
            llm = self._get_gen_model()  # Persistent, reset internally
            # Per official impl: append [No Retrieval] to guide model
            prompt = self._format_prompt(question, passage=None, no_retrieval=True)
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["###", "\n\n\n"],
                echo=False,
            )
            result = self._parse_output(output['choices'][0]['text'])
            result.retrieve = "[No Retrieval]"
            result.retrieval_score = 0.0  # No retrieval possible
            # Per Self-RAG Algorithm 1: ISREL/ISSUP meaningless without passage
            result.isrel = None
            result.issup = None
            return result

        # Retriever provided - check retrieval decision via token probabilities
        # _check_retrieval_needed uses persistent logprobs model
        needs_retrieval, retrieval_score = self._check_retrieval_needed(
            question, threshold=retrieval_threshold
        )

        # Get generation model (persistent, reset internally)
        llm = self._get_gen_model()

        if needs_retrieval:
            # Retrieve and generate with passage
            results = retriever.retrieve(question, top_k=1)
            if results:
                retrieved_passage = results[0]['text']
                result = self._generate_with_passage(
                    llm, question, retrieved_passage, max_tokens, temperature
                )
                result.retrieval_score = retrieval_score
                return result

        # Generate without retrieval
        # Per official impl: append [No Retrieval] to guide model
        prompt = self._format_prompt(question, passage=None, no_retrieval=True)
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["###", "\n\n\n"],
            echo=False,
        )

        result = self._parse_output(output['choices'][0]['text'])
        result.retrieve = "[No Retrieval]"
        result.retrieval_score = retrieval_score
        # Per Self-RAG Algorithm 1: ISREL/ISSUP meaningless without passage
        result.isrel = None
        result.issup = None
        return result

    def generate_with_retrieval(
        self,
        question: str,
        retriever: Any,  # LegalRetriever
        top_k: int = 3,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        RAG pipeline that ALWAYS retrieves (like Basic RAG).

        NOTE: This method always retrieves, bypassing the adaptive retrieval check.
        For proper Self-RAG per Algorithm 1 (adaptive retrieval decision), use:
            result = inference.generate(question, retriever=retriever)

        Steps:
        1. Retrieve passages using LegalRetriever
        2. Generate with best passage (forced retrieval)
        3. Return structured output with retrieval info

        Args:
            question: Input question
            retriever: LegalRetriever instance with loaded index
            top_k: Number of passages to retrieve
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary containing:
                - output: SelfRAGOutput with answer and tokens
                - passages: List of retrieved passages
                - used_passage: The passage used for generation
                - passage_score: Similarity score of used passage
        """
        # Step 1: Retrieve passages
        results = retriever.retrieve(question, top_k=top_k)

        if not results:
            # No passages found - generate without retrieval
            output = self.generate(question, passage=None, max_tokens=max_tokens)
            return {
                'output': output,
                'passages': [],
                'used_passage': None,
            }

        # Step 2: Use top passage (truncate to prevent context overflow)
        top_passage = results[0]['text']
        max_passage_len = 2000  # Safe limit with n_ctx=4096
        if len(top_passage) > max_passage_len:
            top_passage = top_passage[:max_passage_len] + "..."

        # Step 3: Generate with passage
        output = self.generate(question, passage=top_passage, max_tokens=max_tokens)

        return {
            'output': output,
            'passages': results,
            'used_passage': top_passage,
            'passage_score': results[0]['score'],
        }

    def batch_generate(
        self,
        questions: List[str],
        passages: Optional[List[Optional[str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        show_progress: bool = True,
    ) -> List[SelfRAGOutput]:
        """
        Generate answers for multiple questions.

        Args:
            questions: List of input questions
            passages: Optional list of passages (one per question, or None)
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            show_progress: Whether to show progress bar

        Returns:
            List of SelfRAGOutput objects
        """
        if passages is None:
            passages = [None] * len(questions)

        results = []
        iterator = zip(questions, passages)

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Generating")
            except ImportError:
                pass

        for question, passage in iterator:
            result = self.generate(
                question,
                passage=passage,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)

        return results

    def generate_multiple(
        self,
        question: str,
        passage: Optional[str] = None,
        retriever: Any = None,
        num_generations: int = DEFAULT_NUM_GENERATIONS,
        temperature: float = 0.7,
        max_tokens: int = 512,
        retrieval_threshold: float = 0.5,
    ) -> List[SelfRAGOutput]:
        """
        Generate K responses for EigenScore computation.

        Works with ANY mode:
        - No-RAG:    passage=None, retriever=None
        - Basic RAG: passage=<text>, retriever=None
        - Self-RAG:  passage=None, retriever=<instance>

        This method is designed to be used with the standalone compute_eigenscore()
        function for hallucination detection across any generation method.

        Args:
            question: Input question
            passage: Pre-retrieved passage (for Basic RAG mode)
            retriever: Retriever instance (for Self-RAG adaptive mode)
            num_generations: K generations for diversity (default 10)
            temperature: Must be > 0 for diversity (default 0.7)
            max_tokens: Max tokens per generation
            retrieval_threshold: Threshold for Self-RAG adaptive retrieval

        Returns:
            List of K SelfRAGOutput objects
        """
        if temperature <= 0:
            temperature = 0.7  # Need diversity for EigenScore

        generations = []

        # Determine mode and get passage if needed
        used_passage = passage
        if used_passage is None and retriever is not None:
            # Self-RAG mode: check if retrieval needed
            needs_retrieval, _ = self._check_retrieval_needed(question, retrieval_threshold)
            if needs_retrieval:
                results = retriever.retrieve(question, top_k=1)
                if results:
                    used_passage = results[0]['text']

        # Generate K responses
        for i in range(num_generations):
            if i > 0:
                self._llm.reset()

            llm = self._get_gen_model()

            if used_passage:
                result = self._generate_with_passage(llm, question, used_passage, max_tokens, temperature)
            else:
                prompt = self._format_prompt(question, passage=None, no_retrieval=True)
                output = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["###", "\n\n\n"],
                    echo=False,
                )
                result = self._parse_output(output['choices'][0]['text'])
                result.retrieve = "[No Retrieval]"
                result.isrel = None
                result.issup = None

            generations.append(result)

        return generations

    # =========================================================================
    # INSIDE Methods (EigenScore-based hallucination detection)
    # =========================================================================

    def _compute_eigenscore(self, embeddings: List[np.ndarray]) -> float:
        """
        Compute EigenScore per INSIDE paper (Chen et al., ICLR 2024).

        Formula (Equations 4-6, Section 3.1):
        - Z ∈ R^(d×K): each column is a sentence embedding
        - J_d = I_d - (1/d) * 1_d * 1_d^T: centering matrix
        - Σ = Z^T · J_d · Z: K×K covariance matrix
        - EigenScore = (1/K) * Σ log(λ_i + α) where α = 0.001

        Higher EigenScore indicates higher semantic divergence (less consistency),
        suggesting potential hallucination.

        Args:
            embeddings: List of 1D embedding vectors, each shape (d,)

        Returns:
            EigenScore value (float)
        """
        if len(embeddings) < 2:
            return 0.0  # Can't compute with < 2 samples

        K = len(embeddings)
        alpha = 0.001  # Regularization term (per paper Section 4.1)

        # Z: (d, K) - each column is a sentence embedding
        Z = np.column_stack(embeddings)  # Shape: (d, K)

        # Center each embedding by subtracting its mean (per paper Eq. 4)
        # J_d · z = z - mean(z) for each column
        Z_centered = Z - Z.mean(axis=0, keepdims=True)

        # Compute K×K covariance matrix: Σ = Z^T · J_d · Z
        # Since Z is already centered: Σ = Z_centered^T @ Z_centered
        Sigma = Z_centered.T @ Z_centered  # Shape: (K, K)

        # Add regularization to ensure full rank (per paper Eq. 5)
        Sigma_reg = Sigma + alpha * np.eye(K)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(Sigma_reg)

        # Clip to avoid log(0)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)

        # EigenScore = (1/K) * Σ log(λ_i) (per paper Eq. 6)
        eigenscore = np.mean(np.log(eigenvalues))

        return float(eigenscore)

    def _compute_critique_score(
        self,
        output: SelfRAGOutput,
        w_isrel: float = 1.0,
        w_issup: float = 1.0,
        w_isuse: float = 0.5,
    ) -> float:
        """
        Compute critique score per Self-RAG paper Equation 4.

        S(Critique) = w_ISREL × s_ISREL + w_ISSUP × s_ISSUP + w_ISUSE × s_ISUSE

        Since we can't get token probabilities from GGUF, we use discrete values:
        - ISREL: Relevant=1.0, Irrelevant=0.0
        - ISSUP: Fully=1.0, Partially=0.5, No=0.0
        - ISUSE: Normalized (1-5) → (0.0-1.0)

        Args:
            output: SelfRAGOutput with reflection tokens
            w_isrel: Weight for ISREL score (default 1.0)
            w_issup: Weight for ISSUP score (default 1.0)
            w_isuse: Weight for ISUSE score (default 0.5)

        Returns:
            Weighted critique score (float)
        """
        score = 0.0

        # ISREL score
        if output.isrel:
            s_isrel = 1.0 if 'Relevant' in output.isrel and 'Irrelevant' not in output.isrel else 0.0
            score += w_isrel * s_isrel

        # ISSUP score
        if output.issup:
            if 'Fully' in output.issup:
                s_issup = 1.0
            elif 'Partially' in output.issup:
                s_issup = 0.5
            else:
                s_issup = 0.0
            score += w_issup * s_issup

        # ISUSE score (normalize 1-5 to 0-1)
        if output.isuse:
            match = re.search(r'(\d)', output.isuse)
            if match:
                s_isuse = (int(match.group(1)) - 1) / 4.0  # 1→0.0, 5→1.0
                score += w_isuse * s_isuse

        return score

    def _select_best_generation(
        self,
        generations: List[SelfRAGOutput],
        w_isrel: float = 1.0,
        w_issup: float = 1.0,
        w_isuse: float = 0.5,
    ) -> SelfRAGOutput:
        """
        Select best generation using weighted critique score (paper Eq. 4).

        Args:
            generations: List of SelfRAGOutput from multi-generation
            w_isrel: Weight for ISREL score (default 1.0)
            w_issup: Weight for ISSUP score (default 1.0)
            w_isuse: Weight for ISUSE score (default 0.5)

        Returns:
            Best SelfRAGOutput
        """
        if not generations:
            raise ValueError("No generations to select from")

        if len(generations) == 1:
            return generations[0]

        # Score each generation using weighted critique score
        scored = [(self._compute_critique_score(g, w_isrel, w_issup, w_isuse), i, g)
                  for i, g in enumerate(generations)]
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[0][2]

    def generate_with_multi_passage_ranking(
        self,
        question: str,
        retriever: Any,
        top_k: int = 5,  # Paper default
        max_tokens: int = 512,
        temperature: float = 0.0,
        w_isrel: float = 1.0,
        w_issup: float = 1.0,
        w_isuse: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate with multi-passage ranking per Self-RAG paper Section 3.3.

        Simplified version (no beam search):
        1. Retrieve K passages
        2. Generate one output per passage
        3. Score each using weighted critique score (Eq. 4)
        4. Return best output

        Args:
            question: Input question
            retriever: Retriever with retrieve(query, top_k) method
            top_k: Number of passages to retrieve and rank
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            w_isrel: Weight for ISREL score (default 1.0)
            w_issup: Weight for ISSUP score (default 1.0)
            w_isuse: Weight for ISUSE score (default 0.5)

        Returns:
            Dict with 'output', 'used_passage', 'passage_score',
            'critique_score', and 'all_candidates' for analysis
        """
        # Step 1: Retrieve K passages
        results = retriever.retrieve(question, top_k=top_k)

        if not results:
            # No passages found, generate without retrieval
            output = self.generate(question, retriever=None, max_tokens=max_tokens)
            return {
                'output': output,
                'used_passage': None,
                'passage_score': None,
                'critique_score': self._compute_critique_score(output, w_isrel, w_issup, w_isuse),
                'all_candidates': [],
            }

        # Step 2: Generate for each passage
        candidates = []
        for result in results:
            passage = result['text'][:1000]  # Truncate long passages

            # Reset model state before each generation
            self._llm.reset()
            llm = self._get_gen_model()

            # Generate with this passage
            output = self._generate_with_passage(llm, question, passage, max_tokens, temperature)

            # Step 3: Compute critique score
            score = self._compute_critique_score(output, w_isrel, w_issup, w_isuse)
            candidates.append({
                'output': output,
                'passage': passage,
                'passage_score': result['score'],
                'critique_score': score,
            })

        # Step 4: Select best by critique score
        candidates.sort(key=lambda x: x['critique_score'], reverse=True)
        best = candidates[0]

        return {
            'output': best['output'],
            'used_passage': best['passage'],
            'passage_score': best['passage_score'],
            'critique_score': best['critique_score'],
            'all_candidates': candidates,  # For analysis
        }

    def generate_with_eigenscore(
        self,
        question: str,
        passage: Optional[str] = None,
        retriever: Any = None,
        embedding_model: Any = None,  # External encoder (e.g., EmbeddingModel from sentence-transformers)
        num_generations: int = DEFAULT_NUM_GENERATIONS,
        temperature: float = 0.7,  # Need temperature > 0 for diversity
        max_tokens: int = 512,
        eigenscore_threshold: float = EIGENSCORE_THRESHOLD,
        retrieval_threshold: float = 0.5,  # Threshold for adaptive retrieval
    ) -> SelfRAGOutputWithEigenScore:
        """
        Generate with INSIDE hallucination detection via EigenScore.

        Uses external encoder (sentence-transformers) for embeddings instead of
        LLM internal states. Per INSIDE paper research, external embeddings achieve
        ~79% AUROC (vs 80% for middle layer, 77% for final layer).

        Implements multi-generation EigenScore computation:
        1. Check if retrieval is needed using token sampling (like Self-RAG)
        2. Generate K responses with temperature sampling
        3. Embed each response using external encoder
        4. Compute EigenScore from covariance matrix eigenvalues
        5. Select best generation based on reflection tokens
        6. Flag potential hallucination if EigenScore > threshold

        Uses single persistent model with reset() between generations.

        Args:
            question: Input question
            passage: Optional retrieved passage
            retriever: Optional retriever for passage retrieval
            embedding_model: External encoder for embeddings (required).
                            Must have encode(text) method (e.g., EmbeddingModel).
            num_generations: Number of generations for EigenScore (K, default 7)
            temperature: Sampling temperature (must be > 0 for diversity)
            max_tokens: Maximum tokens per generation
            eigenscore_threshold: Threshold for hallucination detection
            retrieval_threshold: Threshold for adaptive retrieval decision

        Returns:
            SelfRAGOutputWithEigenScore with answer, tokens, and EigenScore

        Raises:
            ValueError: If embedding_model not provided
        """
        if embedding_model is None:
            raise ValueError(
                "embedding_model required for EigenScore computation. "
                "Pass the EmbeddingModel instance used for retrieval."
            )

        if temperature <= 0:
            temperature = 0.7  # Need diversity for EigenScore

        generations: List[SelfRAGOutput] = []
        embeddings: List[np.ndarray] = []

        # Step 1: Determine if retrieval is needed using adaptive check (like Self-RAG)
        used_passage = passage
        retrieval_score = None

        if used_passage is None and retriever is not None:
            # Use adaptive retrieval check like regular Self-RAG
            needs_retrieval, retrieval_score = self._check_retrieval_needed(
                question, threshold=retrieval_threshold
            )
            if needs_retrieval:
                results = retriever.retrieve(question, top_k=1)
                if results:
                    used_passage = results[0]['text']

        # Step 2: Generate K responses using persistent model with reset()
        llm = self._get_gen_model()  # Persistent model, reset internally

        for i in range(num_generations):
            # Reset KV cache between generations (except first, already reset)
            if i > 0:
                self._llm.reset()

            if used_passage:
                # Generate with passage
                result = self._generate_with_passage(
                    llm, question, used_passage, max_tokens, temperature
                )
                result.retrieve = "[Retrieval]"
                result.retrieval_score = retrieval_score if retrieval_score else 1.0
            else:
                # Generate without passage
                # Per official impl: append [No Retrieval] to guide model
                prompt = self._format_prompt(question, passage=None, no_retrieval=True)
                output = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["###", "\n\n\n"],
                    echo=False,
                )
                result = self._parse_output(output['choices'][0]['text'])
                result.retrieve = "[No Retrieval]"
                result.retrieval_score = retrieval_score if retrieval_score else 0.0
                # Per Self-RAG Algorithm 1: ISREL/ISSUP meaningless without passage
                result.isrel = None
                result.issup = None

            generations.append(result)

            # Get embedding using EXTERNAL encoder (sentence-transformers)
            if result.answer:
                emb = embedding_model.encode(result.answer)
                # Squeeze to 1D - encode returns (1, dim) for single text
                embeddings.append(np.array(emb).squeeze())

        # Compute EigenScore
        eigenscore = self._compute_eigenscore(embeddings) if embeddings else 0.0

        # Detect hallucination
        hallucination_detected = eigenscore > eigenscore_threshold

        # Select best generation
        best_gen = self._select_best_generation(generations)

        # Create extended output
        return SelfRAGOutputWithEigenScore(
            answer=best_gen.answer,
            retrieve=best_gen.retrieve,
            isrel=best_gen.isrel,
            issup=best_gen.issup,
            isuse=best_gen.isuse,
            raw_output=best_gen.raw_output,
            retrieval_score=best_gen.retrieval_score,
            eigenscore=eigenscore,
            hallucination_detected=hallucination_detected,
            num_generations=num_generations,
            all_generations=generations,
        )


if __name__ == "__main__":
    # Example usage (requires GGUF model file)
    print("SelfRAGGGUFInference Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from src.self_rag.gguf_inference import SelfRAGGGUFInference")
    print("  inference = SelfRAGGGUFInference('path/to/model.gguf')")
    print("  result = inference.generate('What is negligence?')")
    print("  print(result.answer)")
    print("  print(result.isuse)")
