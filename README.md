# Self-RAG System for Legal Analysis

A Retrieval-Augmented Generation (RAG) system with dual hallucination detection for legal document analysis, combining the Self-RAG framework with EigenScore-based semantic consistency analysis.

## Key Features

- **Self-RAG 7B**: Pre-trained model with reflection tokens for adaptive retrieval and self-verification
- **Multi-Passage Ranking**: Critique-based scoring
- **Quantized GGUF**: Q4_K_M quantization (~4GB) fits in 16GB memory, runs on Mac via llama.cpp + Metal
- **EigenScore**: Hallucination detection via multi-generation semantic consistency (external encoder)
- **LegalBench-RAG**: Evaluation on 6,858 legal queries benchmark

## Quick Start

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Project Structure

```
DSC261_Responsible_DS/
├── data/
│   ├── samples/              # Sample legal documents and Q&A data
│   ├── embeddings/           # FAISS vector indices
│   └── legalbench-rag/       # LegalBench-RAG dataset (6,858 queries)
├── models/
│   ├── critic_lora/          # Trained critic LoRA adapters
│   ├── generator_lora/       # Trained generator LoRA adapters
│   └── selfrag_llama2_7b.Q4_K_M.gguf  # Pre-trained Self-RAG 7B
├── src/
│   ├── retrieval/            # Retrieval pipeline (chunking, embedding, indexing)
│   ├── self_rag/             # Self-RAG system (reflection tokens, critic, generator)
│   ├── inside/               # EigenScore hallucination detection, intent detection
│   ├── training/             # Training scripts (label generation, QLoRA)
│   └── evaluation/           # Evaluation metrics
├── notebooks/                # Jupyter tutorials (see table below)
├── configs/                  # YAML configuration files
├── results/                  # Evaluation results
├── pyproject.toml           # uv project configuration
└── README.md                # This file
```

## Configuration Files

All settings customizable via YAML files in `configs/`:

- **retrieval_config.yaml**: Chunk size, embedding model, top-k settings
- **critic_config.yaml**: Base model, QLoRA params, training hyperparameters
- **generator_config.yaml**: Reflection token weights, adaptive retrieval
- **inside_config.yaml**: EigenScore threshold, intent detection
- **legalbench_config.yaml**: LegalBench-RAG evaluation settings

## Notebooks

| # | Notebook | Time | Description |
|---|----------|------|-------------|
| 00 | getting_started | 10 min | Quick start: chunking, embedding, FAISS |
| 01 | data_preparation | 5 min | Load and prepare legal corpus |
| 02 | retrieval_pipeline | 10 min | Build production FAISS index |
| 03 | self_rag_training | 30-60 min | Train critic + generator with QLoRA |
| 04 | evaluation | 20 min | Evaluate retrieval and generation |
| 05 | demo | 15 min | Interactive Self-RAG demo |
| 06 | inside_eigenscore | 30-45 min | EigenScore hallucination detection |
| 07 | intent_aware_retrieval | 30-45 min | Intent-based adaptive retrieval |
| 08 | combined_system | 45-60 min | Complete Self-RAG + EigenScore pipeline |
| 09 | legalbench_retrieval | 60-90 min | LegalBench retrieval evaluation |
| 10 | legalbench_pretrained_selfrag | 30-60 min | Pre-trained Self-RAG 7B GGUF |
| 11 | legalbench_generation | 6-8 hrs | Generation comparison with multi-passage ranking |

**Recommended path**: 00-05 (Tutorial) → 06-08 (EigenScore) → **09-11 (LegalBench)**

## Evaluation Results

### Generation Method Comparison (776 queries)

| Method | F1 | ROUGE-L | EigenScore | Halluc% |
|--------|-----|---------|------------|---------|
| No-RAG | 0.181 | 0.125 | -0.75 | 99% |
| Basic RAG | 0.223 | 0.156 | -1.84 | 60% |
| Self-RAG | 0.219 | 0.157 | -2.00 | 55% |

*EigenScore threshold: -2.0. Lower = more consistent = lower hallucination risk.*

### LegalBench-RAG Retrieval (776 queries)

| Metric | Score |
|--------|-------|
| Precision@1 | 55.93% |
| Recall@64 | 84.41% |

## Metrics Reference

### Retrieval
- **Precision@k**: Fraction of top-k results that are relevant
- **Recall@k**: Fraction of relevant documents in top-k
- **MRR**: Reciprocal rank of first relevant result

### Generation
- **F1 Score**: Token-level overlap with ground truth
- **ROUGE-L**: Longest common subsequence similarity
- **EigenScore**: Semantic consistency across K generations (external encoder)

## References

1. Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
2. Chen et al. (2024). "INSIDE: LLM's Internal States Retain the Power of Hallucination Detection" (ICLR 2024)
3. Pipitone & Houir Alami (2024). "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in Legal Domain"
4. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
