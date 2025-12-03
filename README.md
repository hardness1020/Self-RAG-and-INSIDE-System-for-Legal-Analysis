# Self-RAG System for Legal Analysis

A Retrieval-Augmented Generation (RAG) system with hallucination detection for legal document analysis, using the Self-RAG framework with EigenScore-based semantic consistency analysis.

## Key Features

- **Self-RAG 7B**: Pre-trained model with reflection tokens for adaptive retrieval and self-verification
- **Multi-Passage Ranking**: Critique-based scoring using reflection tokens
- **Quantized GGUF**: Q4_K_M quantization (~4GB) fits in 16GB memory, runs on Mac via llama.cpp + Metal
- **EigenScore**: Hallucination detection via multi-generation semantic consistency
- **LegalBench-RAG**: Evaluation run on a 776-query subset (mini benchmark). Running the full 6,858-query set is time-consuming (~1 minute per query).

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
│   ├── legalbench_embeddings/        # FAISS vector indices
│   └── legalbench-rag/               # LegalBench-RAG dataset (776 queries subset)
├── models/
│   └── selfrag_llama2_7b.Q4_K_M.gguf # Pre-trained Self-RAG 7B (quantized)
├── src/
│   ├── data/                 # LegalBench data loader
│   ├── retrieval/            # Retrieval pipeline (chunking, embedding, indexing)
│   ├── self_rag/             # Self-RAG system (reflection tokens, GGUF inference)
│   ├── evaluation/           # LegalBench evaluation metrics
│   └── utils/                # Device utilities
├── notebooks/                # Main notebooks (1-3)
├── configs/                  # YAML configuration files
├── results/                  # Evaluation results
├── references/               # Research papers
└── README.md                 # This file
```

## Configuration Files

- **retrieval_config.yaml**: Chunk size, embedding model, top-k settings
- **legalbench_config.yaml**: LegalBench-RAG evaluation settings

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | legalbench_retrieval | LegalBench retrieval evaluation with FAISS |
| 2 | legalbench_pretrained_selfrag | Pre-trained Self-RAG 7B GGUF setup and inference |
| 3 | legalbench_generation | Generation comparison: No-RAG vs Basic RAG vs Self-RAG |

**Run order**: 1 (builds index) → 2 (sets up model) → 3 (evaluation)

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
