# Notebooks

## Main Notebooks

| Notebook | Description |
|----------|-------------|
| `1_legalbench_retrieval.ipynb` | LegalBench retrieval evaluation with FAISS indexing |
| `2_legalbench_pretrained_selfrag.ipynb` | Pre-trained Self-RAG 7B GGUF setup and inference |
| `3_legalbench_generation.ipynb` | Generation comparison: No-RAG vs Basic RAG vs Self-RAG |

## Execution Order

1. **1_legalbench_retrieval** - Creates FAISS index from LegalBench corpus
2. **2_legalbench_pretrained_selfrag** - Downloads and converts Self-RAG model to GGUF
3. **3_legalbench_generation** - Evaluates generation methods on 776 queries

See the main [README.md](../README.md) for full project documentation.
