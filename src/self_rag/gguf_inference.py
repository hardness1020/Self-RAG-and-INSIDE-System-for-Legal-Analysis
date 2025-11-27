"""
Self-RAG GGUF Inference Module

Provides inference using pre-trained Self-RAG models converted to GGUF format
for efficient execution on Mac with Metal acceleration via llama.cpp.

Includes INSIDE (INternal States for hallucInation DEtection) integration
via multi-generation EigenScore computation using final layer embeddings.
"""

import re
import math
import gc
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import os

# Suppress Metal kernel init messages (BF16 "not supported" warnings)
os.environ["GGML_METAL_LOG_LEVEL"] = "0"

from src.self_rag.reflection_tokens import ReflectionTokenizer

# INSIDE constants
EIGENSCORE_THRESHOLD = -5.0  # Higher values indicate potential hallucination
DEFAULT_NUM_GENERATIONS = 7  # K generations for EigenScore computation (reduced from 10)


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

    Uses fresh model instance per generation to avoid KV cache corruption
    issues with llama-cpp-python on Metal backend.

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
        Initialize GGUF model configuration for inference.

        Note: Model is NOT loaded here. A fresh instance is created for each
        generation call to avoid KV cache corruption on Metal backend.

        Uses TWO configs:
        - _gen_config: logits_all=False for text generation (prevents memory overflow)
        - _logprobs_config: logits_all=True only for retrieval check (needs token probs)

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (default 4096)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all for Metal)
            verbose: Whether to show llama.cpp verbose output
        """
        try:
            from llama_cpp import Llama
            self._Llama = Llama  # Store class reference for creating fresh instances
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install with: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
            )

        # Config for generation (logits_all=False - don't store intermediate logits)
        # This prevents 'llama_decode returned -3' memory overflow errors
        self._gen_config = {
            'model_path': model_path,
            'n_ctx': n_ctx,
            'n_batch': 512,  # Batch size for prompt processing (prevents KV cache overflow)
            'n_gpu_layers': n_gpu_layers,
            'verbose': verbose,
            'logits_all': False,  # KEY: False for generation to prevent overflow
            'embedding': False,
        }

        # Config for retrieval check (needs logprobs for token probabilities)
        # logits_all=True stores logits for ALL tokens, but we only need prompt + 1 token
        self._logprobs_config = {
            'model_path': model_path,
            'n_ctx': n_ctx,  # Use same context as generation config
            'n_batch': 512,  # Batch size for prompt processing (prevents KV cache overflow)
            'n_gpu_layers': n_gpu_layers,
            'verbose': verbose,
            'logits_all': True,  # Only True when we need logprobs
            'embedding': False,
        }

        self.model_path = model_path
        self._llm = None  # Not pre-loaded - load on-demand to avoid memory issues

        print(f"Model configured: {model_path}")
        print("Note: Model loaded on-demand per query (load-delete-reload pattern)")

    def _get_fresh_model(self):
        """
        Load a fresh Llama instance for text generation.

        Creates new model with logits_all=False configuration.
        Caller is responsible for deleting after use with:
            del llm; gc.collect()

        Returns:
            Llama instance with logits_all=False
        """
        return self._Llama(**self._gen_config)

    def _get_logprobs_model(self):
        """
        Load a fresh Llama instance for retrieval check with logprobs.

        Creates new model with logits_all=True configuration.
        This is required for getting token probabilities.
        Caller is responsible for deleting after use with:
            del llm; gc.collect()

        Returns:
            Llama instance with logits_all=True
        """
        return self._Llama(**self._logprobs_config)

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
        Check if retrieval is needed using token probability scoring (like official Self-RAG).

        This implements the same approach as the official vLLM implementation:
        score = P([Retrieval]) / (P([Retrieval]) + P([No Retrieval]))

        Uses load-delete-reload pattern: creates temporary model with logits_all=True,
        gets token probabilities, then deletes model to free memory.

        Args:
            question: Input question
            threshold: Retrieval threshold (default 0.5, like official impl)

        Returns:
            Tuple of (needs_retrieval: bool, retrieval_score: float)
        """
        # Load temporary model with logits_all=True for logprobs
        llm = self._get_logprobs_model()

        prompt = self._format_prompt(question, passage=None)

        # Generate with logprobs enabled to get token probabilities
        output = llm(
            prompt,
            max_tokens=1,  # Just need first token probabilities
            temperature=0.0,
            logprobs=20,  # Get top 20 token probabilities
            echo=False,
        )

        # Get logprobs from response
        logprobs_data = output['choices'][0].get('logprobs', {})
        top_logprobs = logprobs_data.get('top_logprobs', [{}])[0] if logprobs_data else {}

        # Get log probabilities for retrieval tokens
        # llama-cpp returns log probs, need to convert to probabilities
        ret_logprob = top_logprobs.get('[Retrieval]', -100)
        no_ret_logprob = top_logprobs.get('[No Retrieval]', -100)

        # Convert log probs to probabilities
        ret_prob = math.exp(ret_logprob) if ret_logprob > -50 else 0.0
        no_ret_prob = math.exp(no_ret_logprob) if no_ret_logprob > -50 else 0.0

        # DELETE temporary model to free memory BEFORE loading generation model
        del llm
        gc.collect()

        # Compute normalized score (like official Self-RAG impl)
        total = ret_prob + no_ret_prob
        if total > 0:
            retrieval_score = ret_prob / total
        else:
            # Neither token found in top logprobs - default to retrieval
            retrieval_score = 0.5

        needs_retrieval = retrieval_score > threshold
        return needs_retrieval, retrieval_score

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
        Generate answer with adaptive retrieval using token probability scoring.

        Implements the same approach as official Self-RAG (vLLM implementation):
        1. Compute P([Retrieval]) / (P([Retrieval]) + P([No Retrieval]))
        2. If score > threshold and retriever provided, retrieve and generate with passage
        3. Otherwise, generate without retrieval

        Uses fresh model instances and TWO configs:
        - logits_all=True only for retrieval check (needs token probs)
        - logits_all=False for actual generation (prevents memory overflow)

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
            llm = self._get_fresh_model()  # logits_all=False
            result = self._generate_with_passage(llm, question, passage, max_tokens, temperature)
            result.retrieval_score = 1.0  # Passage was explicitly provided
            del llm
            gc.collect()
            return result

        # If no retriever provided, skip retrieval check entirely (No-RAG mode)
        # This avoids loading logits_all=True model which causes memory overflow
        if retriever is None:
            llm = self._get_fresh_model()  # logits_all=False
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
            del llm
            gc.collect()
            return result

        # Retriever provided - check retrieval decision via token probabilities
        # _check_retrieval_needed loads/deletes its own model internally
        needs_retrieval, retrieval_score = self._check_retrieval_needed(
            question, threshold=retrieval_threshold
        )

        # Load generation model (logits_all=False to prevent overflow)
        llm = self._get_fresh_model()

        if needs_retrieval:
            # Retrieve and generate with passage
            results = retriever.retrieve(question, top_k=1)
            if results:
                retrieved_passage = results[0]['text']
                result = self._generate_with_passage(
                    llm, question, retrieved_passage, max_tokens, temperature
                )
                result.retrieval_score = retrieval_score
                del llm
                gc.collect()
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
        del llm
        gc.collect()
        return result

    def generate_with_retrieval(
        self,
        question: str,
        retriever: Any,  # LegalRetriever
        top_k: int = 3,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Full Self-RAG pipeline with retrieval.

        1. Retrieve passages using LegalRetriever
        2. Generate with best passage
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

    def _select_best_generation(
        self,
        generations: List[SelfRAGOutput],
    ) -> SelfRAGOutput:
        """
        Select best generation based on reflection tokens.

        Selection criteria (in order):
        1. Highest ISUSE utility score
        2. ISSUP = "Fully supported" preferred
        3. First generation as fallback

        Args:
            generations: List of SelfRAGOutput from multi-generation

        Returns:
            Best SelfRAGOutput
        """
        if not generations:
            raise ValueError("No generations to select from")

        if len(generations) == 1:
            return generations[0]

        # Score each generation
        def score_generation(gen: SelfRAGOutput) -> Tuple[int, int]:
            # Extract utility score (1-5), default 0
            utility = 0
            if gen.isuse:
                match = re.search(r'(\d)', gen.isuse)
                if match:
                    utility = int(match.group(1))

            # ISSUP score: Fully=2, Partially=1, No=0
            support = 0
            if gen.issup:
                if 'Fully' in gen.issup:
                    support = 2
                elif 'Partially' in gen.issup:
                    support = 1

            return (utility, support)

        # Sort by score (descending)
        scored = [(score_generation(g), i, g) for i, g in enumerate(generations)]
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[0][2]

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
    ) -> SelfRAGOutputWithEigenScore:
        """
        Generate with INSIDE hallucination detection via EigenScore.

        Uses external encoder (sentence-transformers) for embeddings instead of
        LLM internal states. Per INSIDE paper research, external embeddings achieve
        ~79% AUROC (vs 80% for middle layer, 77% for final layer).

        Implements multi-generation EigenScore computation:
        1. Generate K responses with temperature sampling
        2. Embed each response using external encoder
        3. Compute EigenScore from covariance matrix eigenvalues
        4. Select best generation based on reflection tokens
        5. Flag potential hallucination if EigenScore > threshold

        Uses fresh model instance for EACH generation to avoid KV cache corruption.

        Args:
            question: Input question
            passage: Optional retrieved passage
            retriever: Optional retriever for passage retrieval
            embedding_model: External encoder for embeddings (required).
                            Must have encode(text) method (e.g., EmbeddingModel).
            num_generations: Number of generations for EigenScore (K, default 10)
            temperature: Sampling temperature (must be > 0 for diversity)
            max_tokens: Maximum tokens per generation
            eigenscore_threshold: Threshold for hallucination detection

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

        # Step 1: Retrieve passage ONCE upfront
        used_passage = passage
        if used_passage is None and retriever is not None:
            results = retriever.retrieve(question, top_k=1)
            if results:
                used_passage = results[0]['text']

        # Step 2: Generate K responses with fresh instance for EACH generation
        for i in range(num_generations):
            # Create fresh model instance to avoid KV cache corruption
            llm = self._get_fresh_model()

            if used_passage:
                # Generate with passage
                result = self._generate_with_passage(
                    llm, question, used_passage, max_tokens, temperature
                )
                result.retrieve = "[Retrieval]"
                result.retrieval_score = 1.0
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
                result.retrieval_score = 0.0
                # Per Self-RAG Algorithm 1: ISREL/ISSUP meaningless without passage
                result.isrel = None
                result.issup = None

            # Delete model to free memory before next iteration
            del llm
            gc.collect()

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
