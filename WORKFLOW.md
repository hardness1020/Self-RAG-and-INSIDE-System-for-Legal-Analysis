# System Workflow: Self-RAG + INSIDE for Legal Document Analysis

## Overview

This system combines **Self-RAG** (Self-Reflective Retrieval-Augmented Generation) with **INSIDE** (INternal States for hallucInation DEtection) to provide reliable, hallucination-resistant question-answering for legal documents.

**Key Components:**
- **Self-RAG**: LLM with 5 reflection tokens for self-verification during generation
- **INSIDE**: Internal state analysis using EigenScore for post-generation hallucination detection
- **Intent-Aware Retrieval**: Adaptive retrieval strategies based on query classification

---

## Complete Workflow

```mermaid
graph TD
    A[Legal Documents] --> B[Phase 1: Data Preparation]
    B --> C[Chunks + Embeddings + FAISS Index]

    D[Q&A Pairs] --> E[Phase 2: Label Generation]
    E --> F[Labeled Training Data]

    F --> G[Phase 3A: Train Critic]
    F --> H[Phase 3B: Train Generator]
    G --> I[Critic Model]
    H --> J[Generator Model]

    K[User Query] --> L[Phase 4: Self-RAG Inference]
    C --> L
    I --> L
    J --> L
    L --> M[Answer + Reflection Tokens]

    K --> N[Phase 5: INSIDE Enhancement]
    C --> N
    J --> N
    N --> O[Intent Detection]
    O --> P[Intent-Aware Retrieval]
    P --> Q[Generation + Internal States]
    Q --> R[EigenScore Computation]
    R --> S[Dual Hallucination Detection]
    S --> T[Enhanced Answer + Confidence]

    T --> U[Phase 6: Evaluation]
    M --> U
    U --> V[Metrics + Benchmark Results]
```

---

## Phase 1: Data Preparation & Indexing

**Purpose**: Convert legal documents into searchable vector embeddings

**Workflow:**
1. **Chunking** - Split documents preserving structure
   - File: [`src/retrieval/chunking.py`](src/retrieval/chunking.py)
   - Method: RCTS (Recursive Character Text Splitter)
   - Settings: 512 chars, 50 overlap

2. **Embedding** - Convert chunks to dense vectors
   - File: [`src/retrieval/embedding.py`](src/retrieval/embedding.py)
   - Model: `sentence-transformers/all-mpnet-base-v2` (768 dims)
   - Device: MPS (Mac GPU optimized)

3. **Indexing** - Build FAISS vector index
   - File: [`src/retrieval/indexing.py`](src/retrieval/indexing.py)
   - Index Type: IndexFlatIP (exact search)
   - Output: `data/embeddings/` or `data/legalbench_embeddings/`

**Configuration**: [`configs/retrieval_config.yaml`](configs/retrieval_config.yaml)

**Entry Point**: [`notebooks/01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb)

**Key Class**: `LegalRetriever` in [`src/retrieval/retriever.py`](src/retrieval/retriever.py)

---

## Phase 2: Training Label Generation

**Purpose**: Generate reflection token labels for training data

**Workflow:**
1. Load Q&A pairs from `data/samples/sample_qa_data.json`
2. Generate 5 types of reflection tokens:
   - **Retrieve**: `[Retrieve]`, `[No Retrieval]`
   - **ISREL**: `[Relevant]`, `[Irrelevant]`
   - **ISSUP**: `[Fully Supported]`, `[Partially Supported]`, `[No Support]`
   - **ISUSE**: `[Utility:1]` through `[Utility:5]`
   - **INTENT**: `[Intent:Factual/Exploratory/Comparative/Procedural]`
3. Save labeled data to `data/training/labeled_data.json`

**Methods:**
- **Rule-Based**: Fast, keyword-based heuristics (no API required)
- **GPT-4 Based**: Higher quality using GPT-4 prompts

**Files:**
- [`src/training/generate_labels.py`](src/training/generate_labels.py) - LabelGenerator class
- [`src/self_rag/reflection_tokens.py`](src/self_rag/reflection_tokens.py) - Token definitions + GPT-4 prompts

---

## Phase 3: Model Training (QLoRA)

**Purpose**: Train Critic and Generator models to predict/generate reflection tokens

### Phase 3A: Critic Training

**Workflow:**
1. Load labeled data
2. Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA
3. Train to predict all 5 reflection token types
4. Save LoRA adapters (~50MB)

**File**: [`src/training/train_critic_qlora.py`](src/training/train_critic_qlora.py)

**Config**: [`configs/critic_config.yaml`](configs/critic_config.yaml)

**Model Class**: `CriticModel` in [`src/self_rag/critic.py`](src/self_rag/critic.py)

**Output**: `models/critic_lora/final/`

### Phase 3B: Generator Training

**Workflow:**
1. Load labeled data
2. Use Critic to augment data with predicted tokens
3. Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA
4. Train to generate answers WITH reflection tokens
5. Save LoRA adapters (~50MB)

**File**: [`src/training/train_generator_qlora.py`](src/training/train_generator_qlora.py)

**Config**: [`configs/generator_config.yaml`](configs/generator_config.yaml)

**Model Class**: `SelfRAGGenerator` in [`src/self_rag/generator.py`](src/self_rag/generator.py)

**Output**: `models/generator_lora/final/`

**Entry Point**: [`notebooks/03_self_rag_training.ipynb`](notebooks/03_self_rag_training.ipynb)

**⚠️ Note**: The basic training in notebook 03 uses a toy dataset (10 Q&A pairs) for quick demonstration. Models will be undertrained. For production-quality results on LegalBench, use the **pre-trained Self-RAG GGUF** (notebook 10a) instead of custom training (notebook 10b).

**Training Limitation**: Custom training with Qwen2.5-0.5B (494M params) failed to learn the Self-RAG interleaved token pattern. For custom training, use 7B+ models or fine-tune GPT-4o-mini.

---

## Phase 4: Self-RAG Inference

**Purpose**: RAG pipeline with self-reflective generation using reflection tokens

### Approach A: GGUF Inference (Pre-trained Model) ✅ **Recommended**

For efficient inference without training, use the official pre-trained Self-RAG model:

**File**: [`src/self_rag/gguf_inference.py`](src/self_rag/gguf_inference.py)

**Key Class**: `SelfRAGGGUFInference`
- Uses llama.cpp with Metal acceleration (Mac M-series)
- Implements vLLM-style token probability scoring for retrieval decisions
- Supports adaptive retrieval via `retriever` parameter
- Q4_K_M quantization (~4GB) fits on 16GB machines
- Pre-trained on 150k examples - reflection tokens work out of the box

**Workflow:**
1. Download `selfrag/selfrag_llama2_7b` from HuggingFace (~14GB)
2. Convert to GGUF using llama.cpp `convert_hf_to_gguf.py`
3. Quantize to Q4_K_M using `llama-quantize` (~4GB)
4. Load with `SelfRAGGGUFInference` and integrate with `LegalRetriever`

**Example Usage:**
```python
from src.self_rag.gguf_inference import SelfRAGGGUFInference

inference = SelfRAGGGUFInference(
    model_path="models/selfrag_llama2_7b.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # Metal acceleration
)

# Generate with retrieval
result = inference.generate_with_retrieval(
    question="What are the elements of negligence?",
    retriever=legal_retriever
)
# Returns: SelfRAGOutput with answer, retrieve, isrel, issup, isuse tokens
```

**Entry Point**: [`notebooks/10_legalbench_pretrained_selfrag.ipynb`](notebooks/10_legalbench_pretrained_selfrag.ipynb)

### Approach B: Custom Trained Model ⚠️ Experimental

For custom training on domain-specific data (requires 7B+ model):

**Workflow:**
1. **Query Input** → User question
2. **Retrieval** → FAISS search for top-k relevant passages
3. **Adaptive Retrieval** → Generator decides if more context needed
4. **Generation** → Produce answer with reflection tokens
5. **Token Parsing** → Extract and score reflection tokens
6. **Output** → Answer + reflection metadata + score

**Example Flow:**
```
Query: "What are the elements of negligence?"
  ↓
Retrieval: Top-5 relevant legal passages
  ↓
Generation: "Negligence consists of duty, breach, causation, and damages."
  + [Relevant] + [Fully Supported] + [Utility:5]
  ↓
Output: {
  'answer': "Negligence consists of...",
  'reflection': {'isrel': 'RELEVANT', 'issup': 'FULLY_SUPPORTED', 'isuse': 5},
  'score': 0.95
}
```

**Files:**
- [`src/self_rag/inference.py`](src/self_rag/inference.py) - `SelfRAGPipeline` orchestration
- [`src/self_rag/generator.py`](src/self_rag/generator.py) - Generation + token scoring
- [`src/self_rag/critic.py`](src/self_rag/critic.py) - Token prediction

**Entry Points**:
- [`notebooks/05_demo.ipynb`](notebooks/05_demo.ipynb) - Demo with toy dataset
- [`notebooks/10_legalbench_training.ipynb`](notebooks/10_legalbench_training.ipynb) - LegalBench training (experimental)

**⚠️ Training Limitation**: Qwen2.5-0.5B (494M params) is too small to learn the Self-RAG interleaved token pattern. Use 7B+ models for custom training.

---

## Phase 5: INSIDE Enhancement (Dual Hallucination Detection)

**Purpose**: Add intent-aware retrieval and internal-state-based hallucination detection

**Workflow:**

### Step 1: Intent Detection
- Classify query into 4 types: FACTUAL, EXPLORATORY, COMPARATIVE, PROCEDURAL
- File: [`src/inside/intent_detector.py`](src/inside/intent_detector.py)

### Step 2: Intent-Aware Retrieval
- Adapt retrieval strategy based on intent:
  - FACTUAL: top_k=3, diversity=0.0 (precision)
  - EXPLORATORY: top_k=10, diversity=0.7 (breadth)
  - COMPARATIVE: top_k=6, diversity=0.5 (contrast)
  - PROCEDURAL: top_k=5, diversity=0.3 (sequential)
- File: [`src/retrieval/inside_retriever.py`](src/retrieval/inside_retriever.py)

### Step 3: Generation with Internal State Extraction
- Generate answer while extracting hidden layer embeddings
- Target layer: 14 (middle of 28 layers)
- File: [`src/inside/internal_states.py`](src/inside/internal_states.py)

### Step 4: EigenScore Computation
- Compute covariance matrix of sentence embeddings
- Calculate eigenvalue spectrum
- Differential entropy: `H = 0.5 × (n×log(2πe) + Σ(log(λᵢ)))`
- Higher score = higher entropy = less semantic consistency = hallucination
- File: [`src/inside/eigenscore.py`](src/inside/eigenscore.py)

### Step 5: Dual Hallucination Detection
- **Method 1**: EigenScore > threshold → hallucination
- **Method 2**: ISSUP token = `[No Support]` → hallucination
- **Combined**: Both methods for robust detection
- File: [`src/inside/hallucination_detector.py`](src/inside/hallucination_detector.py)

### Step 6: Combined Scoring
- Final score = 0.7 × reflection_score + 0.3 × eigenscore
- Unified quality metric

**Main File**: [`src/self_rag/inside_generator.py`](src/self_rag/inside_generator.py) - `INSIDEGenerator` class

**Configuration**: [`configs/inside_config.yaml`](configs/inside_config.yaml)

**Entry Points:**
- [`notebooks/06_inside_eigenscore.ipynb`](notebooks/06_inside_eigenscore.ipynb) - INSIDE hallucination detection
- [`notebooks/07_intent_aware_retrieval.ipynb`](notebooks/07_intent_aware_retrieval.ipynb) - Intent-aware strategies
- [`notebooks/08_combined_system.ipynb`](notebooks/08_combined_system.ipynb) - Full integration

---

## Phase 6: Evaluation

**Purpose**: Comprehensive assessment of retrieval, generation, and hallucination detection

### Retrieval Evaluation
- **Metrics**: Precision@k, Recall@k, MRR, MAP
- **Per-Intent**: Breakdown by query intent type
- File: [`src/evaluation/retrieval_eval.py`](src/evaluation/retrieval_eval.py)

### Generation Evaluation
- **Metrics**: Hallucination rate, FactScore, ROUGE-L, Utility, Relevance
- File: [`src/evaluation/generation_eval.py`](src/evaluation/generation_eval.py)

### INSIDE Evaluation
- **Metrics**: EigenScore calibration, Intent detection accuracy, Dual detection F1
- File: [`src/evaluation/inside_eval.py`](src/evaluation/inside_eval.py)

### LegalBench-RAG Benchmark
- **Purpose**: Industry-standard legal RAG benchmark
- **Dataset**: 6,858 queries over 714 legal documents
- **Metrics**: Document-level + snippet-level precision/recall
- File: [`src/evaluation/legalbench_eval.py`](src/evaluation/legalbench_eval.py)
- Config: [`configs/legalbench_config.yaml`](configs/legalbench_config.yaml)

**Entry Points:**
- [`notebooks/04_evaluation.ipynb`](notebooks/04_evaluation.ipynb) - General evaluation (toy dataset)
- [`notebooks/09_legalbench_retrieval.ipynb`](notebooks/09_legalbench_retrieval.ipynb) - Retrieval evaluation ✅
- [`notebooks/10_legalbench_pretrained_selfrag.ipynb`](notebooks/10_legalbench_pretrained_selfrag.ipynb) - Pre-trained Self-RAG GGUF ✅ **Recommended**
- [`notebooks/10_legalbench_training.ipynb`](notebooks/10_legalbench_training.ipynb) - Custom training ⚠️ Experimental
- [`notebooks/11_legalbench_generation.ipynb`](notebooks/11_legalbench_generation.ipynb) - Generation evaluation ✅

**LegalBench Workflow** (Notebooks 09-11):
1. **Retrieval Evaluation** (09): Evaluate retrieval system on 776 queries
2. **Pre-trained Self-RAG** (10a): Use official Self-RAG 7B model with GGUF conversion ✅ **Recommended**
3. **Custom Training** (10b): Train custom model (experimental, requires 7B+ model) ⚠️
4. **Generation Evaluation** (11): Compare 4 methods (No-RAG, Basic RAG, Self-RAG, Self-RAG+INSIDE) ✅

**Generation Evaluation Results** (10-query subset):
| Method | F1 | ROUGE-L | Halluc% (ISSUP) | EigenScore |
|--------|-----|---------|-----------------|------------|
| No-RAG | 0.200 | 0.141 | 70.0% | N/A |
| Basic RAG | 0.248 | 0.176 | 0.0% | N/A |
| Self-RAG | 0.200 | 0.141 | 70.0% | N/A |
| Self-RAG+INSIDE | 0.247 | 0.170 | 0.0% | -5.20 |

**Key Finding**: RAG-based methods (Basic RAG, Self-RAG+INSIDE) show significantly lower hallucination rates and higher F1 scores compared to No-RAG.

**Output Directory**: `results/`

---

## References

- **Self-RAG Paper**: [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)
- **INSIDE Paper**: [https://arxiv.org/abs/2402.03744](https://arxiv.org/abs/2402.03744)
- **LegalBench-RAG**: [https://arxiv.org/abs/2406.10667](https://arxiv.org/abs/2406.10667)
- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Embeddings**: sentence-transformers/all-mpnet-base-v2