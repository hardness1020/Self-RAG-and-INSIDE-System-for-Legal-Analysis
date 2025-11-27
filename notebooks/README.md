# Notebooks Tutorial Series

This directory contains a comprehensive tutorial series for the **Self-RAG + INSIDE System for Legal Analysis** - a complete Retrieval-Augmented Generation (RAG) system that combines the Self-RAG framework with INSIDE (INternal States for hallucInation DEtection) for enhanced reliability in legal document analysis.

## Overview

These interactive Jupyter notebooks (13 total) guide you through building, training, evaluating, and deploying a production-ready legal RAG system. The tutorials progress from basic retrieval concepts to advanced hallucination detection and intent-aware retrieval, culminating in evaluation against the LegalBench-RAG benchmark.

**Note**: For LegalBench evaluation, use the **pre-trained Self-RAG GGUF** approach (notebook 10a) which works out of the box. Custom training (notebook 10b) is experimental and requires a larger model.


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

- Generating reflection token labels (Retrieve, ISREL, ISSUP, ISUSE)
- Training critic model to predict reflection tokens
- Training generator with critic-augmented data
- Quick validation of trained pipeline
- 4-bit quantization for CPU compatibility

**Output**: LoRA adapters in `models/critic_lora/` and `models/generator_lora/`
**Training Stats**: Critic loss ~2.13, Generator loss ~1.73 (3 epochs each, ~11 min on Mac GPU)

**Note**: This notebook trains Self-RAG reflection tokens only. INTENT detection (part of INSIDE) is covered in notebook 07.

**⚠️ Dataset Limitation**: This notebook uses only **10 Q&A pairs** for demonstration purposes. Models trained on this toy dataset will be undertrained and show limited performance. For production-quality training, see notebooks 10-11 which use the full LegalBench dataset (776 queries).

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

**⚠️ Performance Note**: Metrics shown are from models trained on only 10 Q&A pairs (notebook 03). These results demonstrate the evaluation pipeline but do not represent production-quality performance. The low retrieval scores (MRR: 0.09, MAP: 0.11) reflect underfitting due to insufficient training data.

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

**⚠️ Model Quality Note**: This demo uses models trained on a small toy dataset (10 Q&A pairs). Responses may be suboptimal due to underfitting. The notebook demonstrates the Self-RAG pipeline architecture rather than production-quality generation.

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

**Key Insight**: Higher EigenScore → Higher hallucination risk
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

### LegalBench Dataset Integration

#### 09. LegalBench Retrieval Evaluation (60-90 minutes)
**Retrieval evaluation using the LegalBench-RAG benchmark**

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

#### 10a. Pre-trained Self-RAG with GGUF Conversion (30-60 minutes) ✅ **Recommended**
**Use official pre-trained Self-RAG 7B model for efficient inference**

- Downloading pre-trained selfrag_llama2_7b from HuggingFace
- Converting to GGUF format using llama.cpp
- Quantizing to Q4_K_M (~4GB, fits 16GB Mac)
- Validating reflection tokens work after conversion
- Using `SelfRAGGGUFInference` class with Metal acceleration
- Integrating with existing `LegalRetriever` for full RAG pipeline
- Evaluating on LegalBench queries (50 queries evaluated)

**Model**: selfrag/selfrag_llama2_7b → Q4_K_M GGUF (~4GB)
**Output**: Evaluation results in `results/selfrag_7b_gguf_evaluation.json`
**Key Insight**: Pre-trained model already learned reflection tokens on 150k examples

**Status**: ✅ Complete - Model converted, tokens validated, evaluation run

---

#### 10b. Custom Self-RAG Training (30-60 minutes) ⚠️ Experimental
**Attempt to train Self-RAG from scratch on LegalBench data**

- Generating training labels with GPT-5.1 (776 queries)
- Training Qwen2.5-0.5B with QLoRA on Self-RAG interleaved format
- Interleaved token format: `[Retrieve]<p>passage</p>[ISREL]answer[ISSUP][ISUSE]`
- Passage masking in loss calculation
- Vocabulary expansion with 20 special tokens

**Result**: ⚠️ **Model failed to learn the Self-RAG pattern**
- Qwen2.5-0.5B is too small (494M params) to learn the interleaved token format
- Generated outputs lack reflection tokens
- Loss remained high (~2.2) after training

**Recommendations**:
- Use pre-trained Self-RAG GGUF (notebook 10a) instead
- For custom training: Use 7B+ model or fine-tune GPT-4o-mini (~$9 for 776 examples)

**Status**: ⚠️ Experimental - Training code works but model too small

---

#### 11. LegalBench Generation Evaluation (1-2 hours) ✅
**Compare generation methods on LegalBench benchmark using pre-trained Self-RAG GGUF**

- Method comparison: No-RAG, Basic RAG, Self-RAG, Self-RAG+INSIDE
- Generation quality metrics (F1, ROUGE-L, hallucination rate, utility)
- Per-subdataset performance breakdown
- Multi-metric radar charts and visualizations
- Example outputs with side-by-side comparison
- EigenScore hallucination detection using external encoder (~79% AUROC)

**Evaluation Results (10-query subset)**:
| Method | F1 | ROUGE-L | Halluc% (ISSUP) |
|--------|-----|---------|-----------------|
| No-RAG | 0.200 | 0.141 | 70.0% |
| Basic RAG | 0.248 | 0.176 | 0.0% |
| Self-RAG | 0.200 | 0.141 | 70.0% |
| Self-RAG+INSIDE | 0.247 | 0.170 | 0.0% |

**Output**: Results in `results/generation_results_subset.json` with visualizations

**Status**: ✅ Complete - Subset evaluation done, full evaluation ready

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

**⚠️ Note**: Notebooks 03-05 use a toy dataset (10 Q&A pairs) for quick demonstration. Models will be undertrained. For production training, complete Path 1 then proceed to notebooks 10-11.

### Path 2: Advanced Features (Intermediate)
**Total Time: ~3-4 hours (includes Path 1)**

Complete Path 1, then:

7. `06_inside_eigenscore.ipynb` (30-45 min) - Hallucination detection
8. `07_intent_aware_retrieval.ipynb` (30-45 min) - Adaptive retrieval
9. `08_combined_system.ipynb` (45-60 min) - Full integration

### Path 3: LegalBench Dataset (Advanced)
**Total Time: ~5-7 hours (includes Paths 1 & 2)**

Complete Paths 1 & 2, then work with the production LegalBench-RAG dataset:

10. `09_legalbench_retrieval.ipynb` (60-90 min) - Retrieval evaluation ✅
11. `10_legalbench_pretrained_selfrag.ipynb` (30-60 min) - Pre-trained Self-RAG GGUF ✅ **Recommended**
12. `10_legalbench_training.ipynb` (30-60 min) - Custom training ⚠️ Experimental (optional)
13. `11_legalbench_generation.ipynb` (1-2 hours) - Generation evaluation ✅

**Progress**:
- ✅ Retrieval evaluation complete with strong results (P@1: 55.93%, R@64: 84.41%)
- ✅ Pre-trained Self-RAG 7B converted to GGUF, tokens validated, evaluation run
- ⚠️ Custom training attempted but Qwen2.5-0.5B too small (use 7B+ for production)
- ✅ Generation evaluation complete with 4 methods compared (No-RAG, Basic RAG, Self-RAG, Self-RAG+INSIDE)

**Key Insight**: Use the pre-trained GGUF model (notebook 10a) for best results. Custom training (notebook 10b) requires a larger base model to learn the Self-RAG reflection token pattern.
