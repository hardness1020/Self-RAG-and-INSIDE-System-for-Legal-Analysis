# Notebooks Tutorial Series

This directory contains a comprehensive tutorial series for the **Self-RAG + INSIDE System for Legal Analysis** - a complete Retrieval-Augmented Generation (RAG) system that combines the Self-RAG framework with INSIDE (INternal States for hallucInation DEtection) for enhanced reliability in legal document analysis.

## Overview

These interactive Jupyter notebooks guide you through building, training, evaluating, and deploying a production-ready legal RAG system. The tutorials progress from basic retrieval concepts to advanced hallucination detection and intent-aware retrieval, culminating in evaluation against the LegalBench-RAG benchmark.

## Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)
- GPU recommended but not required (CPU-compatible with 4-bit quantization)
- Sample legal documents provided in `data/samples/`
- LegalBench-RAG dataset (for benchmark evaluation)

## Notebooks

### Core Tutorial Series (Beginner → Intermediate)

#### 00. Getting Started (10-15 minutes)
**Quick start tutorial introducing basic retrieval pipeline components**

- Document chunking with RCTS (Recursive Character Text Splitter)
- Embedding generation using sentence transformers
- Similarity search and semantic matching
- FAISS indexing for efficient retrieval
- Saving and loading indexes

**Output**: Creates FAISS index with sample documents, demonstrates top-k retrieval

---

#### 01. Data Preparation (5-10 minutes)
**Prepare legal corpus and Q&A data for training**

- Loading sample legal documents
- Exploring document statistics and characteristics
- Testing chunking strategies (chunk size, overlap)
- Preparing Q&A pairs for training
- Creating train/test splits (80/20)

**Output**: Prepared training data in `data/training/` (10 documents → 21 chunks, 10 Q&A pairs)

---

#### 02. Retrieval Pipeline (10-15 minutes)
**Build production-ready retrieval system with FAISS**

- Loading configuration from YAML files
- Indexing legal documents with embeddings
- Testing retrieval quality on legal queries
- Persisting vector database for reuse
- Mac GPU (MPS) optimization

**Output**: Production FAISS index in `data/embeddings/`, relevance scores for test queries

---

#### 03. Self-RAG Training (30-60 minutes)
**Train critic and generator models using QLoRA**

- Generating reflection token labels (ISREL, ISSUP, ISUSE, INTENT)
- Training critic model to predict reflection tokens
- Training generator with critic-augmented data
- Quick validation of trained pipeline
- 4-bit quantization for CPU compatibility

**Output**: LoRA adapters in `models/critic_lora/` and `models/generator_lora/`
**Training Stats**: Critic loss ~2.13, Generator loss ~1.73 (3 epochs each, ~11 min on Mac GPU)

---

#### 04. Evaluation (20-30 minutes)
**Comprehensive performance evaluation with multiple metrics**

- Retrieval evaluation (P@k, R@k, MRR, MAP)
- Generation quality metrics (hallucination rate, FactScore, utility)
- Visualization of retrieval and generation metrics
- Comparison with baselines (vanilla RAG, no-RAG)
- Reflection token correlation analysis

**Output**: Results in `results/` with visualizations
**Sample Metrics**: 0% hallucination, 0.91 completeness, MRR: 0.09, MAP: 0.11

---

#### 05. Demo (15-20 minutes)
**Interactive demonstration of complete Self-RAG system**

- Loading full Self-RAG pipeline (retriever + generator + critic)
- Testing with legal questions (negligence, res ipsa loquitur, defenses)
- Interactive Q&A with custom questions
- Analyzing reflection tokens and self-verification
- Batch processing multiple questions
- Exporting demo results

**Output**: Interactive demo with reflection token analysis in `results/demo_results.json`

---

### Advanced INSIDE Integration

#### 06. INSIDE EigenScore (30-45 minutes)
**Hallucination detection via internal state analysis**

- Understanding LLM internal states and hidden layer embeddings
- Computing EigenScore from covariance eigenvalue spectrum
- Comparing factual vs hallucinated content
- Using the unified HallucinationDetector interface
- Multi-generation detection for robustness
- Calibrating thresholds on labeled data

**Key Insight**: Lower EigenScore → Higher hallucination risk
**Output**: EigenScore computation, eigenvalue visualizations, calibrated thresholds

---

#### 07. Intent-Aware Retrieval (30-45 minutes)
**Adaptive retrieval strategies based on query intent**

- Automatic intent detection (Factual, Exploratory, Comparative, Procedural)
- Intent-specific retrieval strategies:
  - **Factual**: 3 docs, diversity=0.0 (high precision)
  - **Exploratory**: 10 docs, diversity=0.7 (broad coverage)
  - **Comparative**: 6 docs, diversity=0.5 (contrasting views)
  - **Procedural**: 5 docs, diversity=0.3 (sequential steps)
- Query characteristics analysis
- Confidence scores for intent classification
- Integration with production retriever

**Output**: Intent classifications, adaptive retrieval parameters, confidence visualizations

---

#### 08. Combined System (45-60 minutes)
**Complete Self-RAG + INSIDE integrated system**

- Unified configuration management (3 config files)
- INSIDE-enhanced generator initialization
- End-to-end pipeline demonstration
- Comparative analysis: Self-RAG vs Self-RAG+INSIDE
- Feature comparison and integration benefits
- Production deployment code examples
- Performance monitoring framework

**Complete Pipeline**:
```
Query → Intent Detection → Adaptive Retrieval → Self-RAG Generation
           ↓                       ↓                      ↓
    INTENT Token          Diversity/Precision    Reflection Tokens
                                                          ↓
                                              Internal States Extraction
                                                          ↓
                                                  EigenScore Computation
                                                          ↓
                                              Hallucination Detection
                                                          ↓
                                        Combined Scoring (70% reflection + 30% EigenScore)
```

**Output**: Complete system analysis with visualizations
**Benefits**: +15-25% hallucination detection, +10-20% retrieval quality

---

### Benchmarking

#### 09. LegalBench-RAG Benchmark (60-90 minutes)
**Evaluation using the first dedicated benchmark for legal RAG**

- Automatic dataset preparation (merging 4 subdatasets)
- Exploring 6,858 queries across 714 legal documents
- Indexing large corpus (~327k chunks, 79M+ characters)
- Running full benchmark evaluation (776 queries in mini version)
- Document-level and snippet-level metrics
- Per-dataset performance breakdown (ContractNLI, CUAD, MAUD, PrivacyQA)
- Comparison to published baselines
- Comprehensive visualizations (P@k curves, heatmaps)

**Benchmark Results (Mini Version)**:
- Overall: P@1: 55.93%, R@64: 84.41%
- Best performance: PrivacyQA (88.66% P@1, 100% R@64)
- Most challenging: MAUD (13.92% P@1, 61.34% R@64)
- **vs Baseline**: +49.52% P@1, +22.19% R@64 (significant improvement!)

**Output**: Comprehensive results in `results/legalbench_evaluation_results.json` with plots

---

## Recommended Workflow

### Path 1: Core System (Beginner)
**Total Time: ~2 hours**

1. `00_getting_started.ipynb` (10 min) - Learn basics
2. `01_data_preparation.ipynb` (5 min) - Prepare data
3. `02_retrieval_pipeline.ipynb` (10 min) - Build retrieval
4. `03_self_rag_training.ipynb` (30-60 min) - Train models
5. `04_evaluation.ipynb` (20 min) - Measure performance
6. `05_demo.ipynb` (15 min) - Interactive testing

### Path 2: Advanced Features (Intermediate)
**Total Time: ~3-4 hours (includes Path 1)**

Complete Path 1, then:

7. `06_inside_eigenscore.ipynb` (30-45 min) - Hallucination detection
8. `07_intent_aware_retrieval.ipynb` (30-45 min) - Adaptive retrieval
9. `08_combined_system.ipynb` (45-60 min) - Full integration

### Path 3: Benchmarking (Advanced)
**Total Time: ~4-5 hours (includes Paths 1 & 2)**

Complete Paths 1 & 2, then:

10. `09_legalbench_benchmark.ipynb` (60-90 min) - Industry benchmark

---

## Quick Start

To get started quickly:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook

# 3. Open and run notebooks in order (00 → 01 → 02 → ...)
```

Start with `00_getting_started.ipynb` for a 10-minute introduction to the core concepts.

---

## Key Technologies & Concepts

- **Self-RAG**: Retrieval-Augmented Generation with reflection tokens (Retrieve, ISREL, ISSUP, ISUSE, INTENT) for self-verification
- **INSIDE**: INternal States for hallucInation DEtection using EigenScore analysis
- **QLoRA**: 4-bit quantized LoRA for memory-efficient training (CPU-compatible)
- **RCTS**: Recursive Character Text Splitter optimized for legal document structure
- **FAISS**: Facebook AI Similarity Search for efficient vector database operations
- **Intent Detection**: Classifies queries into 4 types (Factual, Exploratory, Comparative, Procedural)
- **Reflection Tokens**: Self-verification signals that guide generation quality
- **EigenScore**: Eigenvalue-based metric for detecting hallucinations via internal states
- **LegalBench-RAG**: First comprehensive benchmark for legal RAG systems
- **Mac GPU (MPS)**: Optimized for Apple Silicon acceleration

---

## Configuration Files

The notebooks use these configuration files:

- `configs/retrieval_config.yaml` - Retrieval system settings
- `configs/generator_config.yaml` - Generator model configuration
- `configs/inside_config.yaml` - INSIDE system parameters

---

## Data Directories

- `data/samples/` - Sample legal documents and Q&A pairs
- `data/training/` - Prepared training and test splits
- `data/embeddings/` - FAISS indexes and vector databases
- `data/legalbench-rag/` - LegalBench-RAG benchmark dataset
- `results/` - Evaluation results and visualizations
- `models/` - Trained LoRA adapters

---

## Support

For questions or issues with the notebooks, please refer to the main project README or open an issue on the project repository.

---

## Citation

If you use these notebooks or the LegalBench-RAG benchmark in your research, please cite:

- **Self-RAG**: Asai et al. (2023) - Self-Reflective Retrieval-Augmented Generation
- **INSIDE**: [Relevant paper for INSIDE framework]
- **LegalBench-RAG**: The first dedicated benchmark for legal RAG systems

---

**Last Updated**: 2025-11-20
