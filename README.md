# RAG Evaluation System

A domain-specific Retrieval-Augmented Generation (RAG) pipeline built to systematically compare chunking strategies and measure retrieval quality using standard evaluation metrics.

> **Status:** In Progress — Phase 1 (Ingestion) complete, Phase 2 (Retrieval) in development.

---

## What This Project Does

Most RAG tutorials stop at "it works." This project goes further — it asks *how well* does it work, and *why*.

The system ingests a corpus of domain-specific documents, chunks them using three different strategies, embeds them into a vector store, and retrieves answers to natural language questions. Every design decision — chunk size, chunking strategy, embedding model — is treated as an experimental variable with measurable outcomes.

---

## Pipeline Overview

```
Raw Documents (PDF, TXT, MD, HTML)
        ↓
   Document Loader          ← supports multiple formats, attaches metadata
        ↓
   Chunking Strategies      ← fixed / semantic / hierarchical (RAPTOR)
        ↓
   Embedding Model          ← sentence-transformers (all-MiniLM-L6-v2 / bge-large)
        ↓
   FAISS Vector Store       ← persisted locally with full metadata
        ↓
   Hybrid Retrieval         ← BM25 sparse + dense vector + cross-encoder re-ranking
        ↓
   LLM Answer Generation    ← Mistral / LLaMA via Ollama (local, free)
        ↓
   RAGAs Evaluation         ← faithfulness, answer relevancy, context recall
```

---

## Chunking Strategies Compared

| Strategy | How it works | Best for |
|---|---|---|
| Fixed-size | Splits every N tokens with overlap | Baseline, fast, predictable |
| Semantic | Splits on meaning shift using embeddings | Preserving ideas across boundaries |
| Hierarchical (RAPTOR) | Leaf chunks + summarized parent layers | Both broad and specific queries |

---

## Evaluation Metrics (Phase 3)

Using the [RAGAs](https://github.com/explodinggradients/ragas) framework:

- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevancy** — does the answer actually address the question?
- **Context Recall** — did retrieval find the right chunks?

Results will be logged to Weights & Biases for full experiment tracking.

---

## Project Structure

```
rag-eval-system/
├── ingestion/
│   ├── loader.py       # multi-format document loading with metadata
│   ├── chunker.py      # fixed, semantic, hierarchical chunking strategies
│   └── embedder.py     # sentence-transformer embeddings + FAISS indexing
├── retrieval/          # BM25 + dense hybrid retrieval, re-ranking
├── evaluation/         # RAGAs metrics, ablation study, results logging
├── app/                # Streamlit demo with source attribution
├── data/raw/           # document corpus (not tracked by git)
├── vectorstore/        # persisted FAISS index (not tracked by git)
└── requirements.txt
```

---

## Tech Stack

- **LangChain** — pipeline orchestration
- **Sentence-Transformers** — embedding models
- **FAISS** — vector similarity search
- **RAGAs** — retrieval evaluation framework
- **Ollama** — local LLM inference (Mistral / LLaMA)
- **Streamlit** — interactive demo UI
- **Weights & Biases** — experiment tracking

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/vabhishekprakash/rag-eval-system.git
cd rag-eval-system

# Install dependencies
pip install -r requirements.txt

# Add your documents
# Drop PDF / TXT / MD files into data/raw/

# Run the ingestion pipeline
python -c "
from ingestion.loader import load_documents
from ingestion.chunker import get_chunks
docs = load_documents('data/raw')
chunks = get_chunks(docs, strategy='fixed')
print(f'Loaded {len(docs)} pages → {len(chunks)} chunks')
"
```

---

## Roadmap

- [x] Phase 1 — Document ingestion pipeline (loader, chunker, embedder)
- [ ] Phase 2 — Hybrid retrieval + LLM generation
- [ ] Phase 3 — RAGAs evaluation + ablation study across chunking strategies
- [ ] Phase 4 — Streamlit demo + Hugging Face Spaces deployment
---

## Why This Project

Retrieval quality is the most underexplored part of RAG systems. Most implementations treat chunking as a one-line decision. This project treats it as a research question — systematically measuring how chunking strategy, chunk size, and embedding model affect downstream answer quality on a domain-specific corpus.

---

## Author

**Vallamalla Abhishek Prakash**
B.Tech CSE (AI & ML) — Final Year
[GitHub](https://github.com/vabhishekprakash) · [LinkedIn](https://linkedin.com/in/vabhishekprakash)