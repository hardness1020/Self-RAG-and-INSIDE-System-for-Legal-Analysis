# Self-RAG + INSIDE System for Legal Analysis

A complete Retrieval-Augmented Generation (RAG) system with dual hallucination detection for legal document analysis, combining the Self-RAG framework with INSIDE (INternal States for hallucInation DEtection) for enhanced reliability.

**Status**: 100% Complete with INSIDE Integration | **Files**: 30+ Python modules | **Lines of Code**: 9,000+

## What This Is

This project implements a **Self-RAG + INSIDE** system for legal document analysis, combining two complementary approaches to hallucination mitigation:

1. **Legal Retrieval Pipeline**: Efficient retrieval using RCTS chunking and FAISS vector search
2. **Self-RAG System**: LLM with 5 reflection tokens for adaptive retrieval and self-verification
3. **INSIDE Framework**: Intent detection and EigenScore-based hallucination detection via internal states
4. **Training Infrastructure**: QLoRA-based training scripts for both critic and generator models
5. **Evaluation Framework**: Comprehensive metrics for retrieval, generation quality, and hallucination detection

### Key Features

- **Dual Hallucination Detection**: ISSUP token (Self-RAG) + EigenScore (INSIDE) for robust verification
- **Intent-Aware Retrieval**: Auto-detect query intent (Factual, Exploratory, Comparative, Procedural) with adaptive strategies
- **Self-Verification**: Five reflection tokens (Retrieve, ISREL, ISSUP, ISUSE, INTENT) + internal state analysis
- **CPU-Compatible**: 4-bit quantization (QLoRA) enables training on consumer hardware
- **Modern Tooling**: Uses `uv` for ultra-fast dependency management (10-100x faster than pip)

## Quick Start (5 Minutes)

### Installation

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies (auto-creates virtual environment!)
uv sync
```

### Run the System

```bash
# Test components
uv run python src/retrieval/chunking.py
uv run python src/self_rag/reflection_tokens.py

# Start tutorial (recommended)
uv run jupyter notebook notebooks/00_getting_started.ipynb
```

**See notebooks/00_getting_started.ipynb for complete indexing and retrieval examples.**

## Using INSIDE Features

```python
# INSIDE-Enhanced Generator (Dual Hallucination Detection)
from src.self_rag.inside_generator import INSIDEGenerator
generator = INSIDEGenerator.from_config(
    generator_config_path='configs/generator_config.yaml',
    inside_config_path='configs/inside_config.yaml'
)
result = generator.generate_with_inside(
    query="What are the elements of negligence?",
    detect_hallucination=True
)

# Intent-Aware Retrieval
from src.retrieval.inside_retriever import INSIDERetriever
inside_retriever = INSIDERetriever(base_retriever=retriever)
result = inside_retriever.retrieve("Compare negligence vs strict liability")

# Standalone Hallucination Detection
from src.inside import create_detector
detector = create_detector(model, tokenizer, device='cpu')
result = detector.detect_from_generations(query, generations)
```

**See notebooks 06-08 for complete tutorials.**

## Complete Workflow

```bash
# 1. Generate training labels
uv run python -m src.training.generate_labels \
    --input data/samples/sample_qa_data.json --output-dir data/training

# 2. Train critic model (CPU-compatible, reduce epochs for testing)
uv run python -m src.training.train_critic_qlora --config configs/critic_config.yaml

# 3. Train generator model
uv run python -m src.training.train_generator_qlora \
    --config configs/generator_config.yaml --critic-weights models/critic_lora/final

# 4. Evaluate retrieval
uv run python -m src.evaluation.retrieval_eval \
    --config configs/retrieval_config.yaml --index-dir data/embeddings \
    --test-data data/samples/sample_test_queries.json

# 5. Evaluate generation
uv run python -m src.evaluation.generation_eval \
    --retrieval-config configs/retrieval_config.yaml \
    --generator-config configs/generator_config.yaml \
    --generator-weights models/generator_lora/final \
    --test-data data/samples/sample_qa_data.json
```

**For inference examples, see notebooks 05_demo.ipynb and 08_combined_system.ipynb.**

## Project Structure

```
DSC261_Responsible_DS/
├── data/
│   ├── samples/              # Sample legal documents and Q&A data
│   ├── embeddings/           # FAISS vector indices (created after indexing)
│   └── training/             # Generated training labels
├── models/
│   ├── critic_lora/          # Trained critic LoRA adapters
│   └── generator_lora/       # Trained generator LoRA adapters
├── src/
│   ├── retrieval/            # Retrieval pipeline (chunking, embedding, indexing)
│   │   └── inside_retriever.py  # Intent-aware retrieval wrapper
│   ├── self_rag/             # Self-RAG system (reflection tokens, critic, generator)
│   │   └── inside_generator.py  # INSIDE-enhanced generator
│   ├── inside/               # INSIDE framework (NEW)
│   │   ├── internal_states.py   # Internal state extraction
│   │   ├── eigenscore.py        # EigenScore computation
│   │   ├── feature_clipping.py  # Overconfident hallucination detection
│   │   ├── intent_detector.py   # Query intent classification
│   │   └── hallucination_detector.py  # Unified detection interface
│   ├── training/             # Training scripts (label generation, QLoRA training)
│   └── evaluation/           # Evaluation metrics (retrieval and generation)
│       └── inside_eval.py       # INSIDE evaluation framework
├── notebooks/                # Jupyter notebooks for tutorials and experiments
│   ├── 00_getting_started.ipynb
│   ├── 01_data_preparation.ipynb
│   ├── 02_retrieval_pipeline.ipynb
│   ├── 03_self_rag_training.ipynb       # ⚠️ Uses toy dataset (10 Q&A pairs)
│   ├── 04_evaluation.ipynb
│   ├── 05_demo.ipynb
│   ├── 06_inside_eigenscore.ipynb       # INSIDE EigenScore tutorial (NEW)
│   ├── 07_intent_aware_retrieval.ipynb  # Intent-aware retrieval (NEW)
│   ├── 08_combined_system.ipynb         # Complete system demo (NEW)
│   ├── 09_legalbench_retrieval.ipynb    # LegalBench retrieval evaluation
│   ├── 10_legalbench_training.ipynb     # LegalBench production training 🚧
│   └── 11_legalbench_generation.ipynb   # LegalBench generation evaluation 🚧
├── configs/                  # YAML configuration files
│   ├── retrieval_config.yaml
│   ├── critic_config.yaml
│   ├── generator_config.yaml
│   └── inside_config.yaml    # INSIDE configuration (NEW)
├── results/                  # Evaluation results (created after evaluation)
├── pyproject.toml           # uv project configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

### Configuration Files (`configs/`)

All settings customizable via YAML files:

- **retrieval_config.yaml**: Chunk size, embedding model, top-k settings, intent-aware strategies
- **critic_config.yaml**: Base model, QLoRA params, training hyperparameters, INTENT token
- **generator_config.yaml**: Reflection token weights, adaptive retrieval, INSIDE integration
- **inside_config.yaml**: EigenScore threshold, internal state extraction, intent detection
- **legalbench_config.yaml**: LegalBench-RAG evaluation settings

## Evaluation Metrics

### Retrieval Quality
- **Precision@k**: Fraction of top-k results that are relevant
- **Recall@k**: Fraction of all relevant documents in top-k
- **Mean Reciprocal Rank (MRR)**: Reciprocal rank of first relevant result
- **Mean Average Precision (MAP)**: Average precision across queries
- **Per-Intent Metrics** (NEW): Retrieval quality breakdown by query intent

### Generation Quality
- **Hallucination Rate (Dual)** (NEW): Detection via both ISSUP token and EigenScore
- **EigenScore** (NEW): Semantic consistency from internal state embeddings (lower = potential hallucination)
- **Intent Detection Accuracy** (NEW): Correctness of query intent classification
- **Combined Score** (NEW): Weighted fusion of reflection tokens + EigenScore
- **FactScore**: Factual accuracy of generated responses
- **ROUGE Scores**: Lexical overlap with reference answers
- **Utility Score**: Overall response quality (1-5 scale via ISUSE token)

### INSIDE-Specific Metrics (NEW)
- **EigenScore Calibration**: ROC curves and optimal thresholds per intent
- **Feature Clipping Impact**: Comparison of clipped vs unclipped generations
- **Intent Classification F1**: Precision and recall per intent type

## Notebooks

Twelve tutorial notebooks provide hands-on learning:

### Core Self-RAG (Notebooks 00-05)
1. **00_getting_started.ipynb** - Quick start (10 min)
2. **01_data_preparation.ipynb** - Load and prepare corpus
3. **02_retrieval_pipeline.ipynb** - Build retrieval system
4. **03_self_rag_training.ipynb** - Train critic and generator
5. **04_evaluation.ipynb** - Performance evaluation
6. **05_demo.ipynb** - Interactive demo

**⚠️ Note**: Notebooks 03-05 use a toy dataset (10 Q&A pairs) for quick demonstration. Models will be undertrained. For production training, see notebooks 10-11.

### INSIDE Integration (Notebooks 06-08)
7. **06_inside_eigenscore.ipynb** - EigenScore hallucination detection (30-45 min)
8. **07_intent_aware_retrieval.ipynb** - Intent detection and adaptive retrieval (30-45 min)
9. **08_combined_system.ipynb** - Complete Self-RAG + INSIDE pipeline (45-60 min)

### LegalBench Dataset (Notebooks 09-11)
10. **09_legalbench_retrieval.ipynb** - Retrieval evaluation on LegalBench-RAG (60-90 min) ✅
11. **10_legalbench_training.ipynb** - Production training on 776 queries (2-3 hours) 🚧
12. **11_legalbench_generation.ipynb** - Generation method comparison (1-2 hours) 🚧

**Workflow**: Start with 00, run 01-05 for core Self-RAG, then 06-08 for INSIDE features, and finally 09-11 for LegalBench evaluation and production training.

## LegalBench-RAG Benchmark

This project now includes support for the **LegalBench-RAG** benchmark dataset, the first dedicated benchmark for evaluating retrieval systems in the legal domain.

### Dataset Overview

- **6,858 queries** over 714 legal documents (79M+ characters)
- **Human-annotated** by legal experts with character-level precision
- **4 subdatasets**: ContractNLI, CUAD, MAUD, PrivacyQA
- **Mini version**: 776 queries (194 per dataset) for rapid iteration
- **Public dataset**: https://github.com/zeroentropy-cc/legalbenchrag

### Quick Start with LegalBench-RAG

```bash
# 1. Download the dataset
git clone https://github.com/zeroentropy-cc/legalbenchrag data/legalbench-rag

# 2. Index the LegalBench-RAG corpus
uv run python -m src.retrieval.indexing \
    --corpus-dir data/legalbench-rag/corpus \
    --output-dir data/legalbench-rag/embeddings \
    --config configs/retrieval_config.yaml

# 3. Evaluate your retrieval system
uv run python -m src.evaluation.legalbench_eval \
    --config configs/legalbench_config.yaml \
    --retrieval-config configs/retrieval_config.yaml \
    --index-dir data/legalbench-rag/embeddings \
    --output results/legalbench_results.json
```

### Evaluation Metrics

LegalBench-RAG provides both **document-level** and **snippet-level** metrics:

**Document-level** (standard IR):
- Precision@k: Fraction of retrieved docs that are relevant
- Recall@k: Fraction of relevant docs that are retrieved

**Snippet-level** (character-precise):
- Snippet Precision@k: Fraction of retrieved chunks matching ground truth spans
- Snippet Recall@k: Fraction of ground truth spans found in top-k
- Uses IoU (Intersection over Union) for span overlap

**Per-dataset breakdown**: Metrics aggregated by ContractNLI, CUAD, MAUD, PrivacyQA

## How INSIDE Enhances Self-RAG

| Aspect | Self-RAG | INSIDE | Combined Benefit |
|--------|----------|--------|------------------|
| **Detection Stage** | During generation | Post-generation analysis | Multi-layer verification |
| **Signal Source** | Output tokens | Internal embeddings | Diverse evidence |
| **Hallucination Detection** | ISSUP token | EigenScore | Robust dual detection |
| **Retrieval** | Adaptive (when needed) | Intent-aware (how to retrieve) | Smart + targeted |

### Key Improvements

1. **Dual verification**: Combines reflection tokens with internal state analysis
2. **Intent-aware retrieval**: Adaptive strategies based on query type
3. **Automatic intent classification**: Factual, Exploratory, Comparative, Procedural
4. **Unified quality score**: Weighted fusion of multiple signals

## References

1. Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
2. INSIDE: Hallucination Detection via Internal States - Framework for semantic consistency analysis
3. Pipitone & Houir Alami (2024). "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in Legal Domain" - First benchmark for legal RAG retrieval
4. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
