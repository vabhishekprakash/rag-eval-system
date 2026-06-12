# RAG Evaluation System

A domain-specific Retrieval-Augmented Generation (RAG) pipeline built to systematically compare chunking strategies and measure retrieval quality using standard evaluation metrics.

---

## What This Project Does

Most RAG tutorials stop at "it works." This project goes further — it asks *how well* does it work, and *why*.

The system ingests a corpus of domain-specific documents, chunks them using multiple strategies, embeds them into a vector store, and retrieves answers to natural language questions using a local LLM. Every design decision — chunk size, chunking strategy, embedding model — is treated as an experimental variable with measurable outcomes.

---

## Evaluation Results

| Strategy | Chunk Size | Faithfulness | Answer Relevancy |
|---|---|---|---|
| Fixed | 256 | 1.0000 | 0.8968 |
| Fixed | 512 | 1.0000 | N/A |
| Fixed | 1024 | 1.0000 | 0.9756 |
| Semantic | — | 1.0000 | 0.9756 |

**Key finding:** All strategies maintain perfect faithfulness (no hallucination).
Semantic chunking and fixed-size 1024 achieve highest answer relevancy (0.9756).

---

## Pipeline Overview

```
Raw Documents (PDF, TXT, MD, HTML)
        ↓
   Document Loader          ← multi-format support with metadata tracking
        ↓
   Chunking Strategies      ← fixed / semantic (with configurable parameters)
        ↓
   Embedding Model          ← sentence-transformers (all-MiniLM-L6-v2)
        ↓
   FAISS Vector Store       ← persisted locally with full metadata
        ↓
   Similarity Retrieval     ← top-k most relevant chunks
        ↓
   LLM Answer Generation    ← Mistral 7B via Ollama (local, offline)
        ↓
   Source Attribution       ← answers cite which documents were used
```

---

## Current Features

1) **Multi-format document loading** — PDF, TXT, MD, HTML with automatic format detection  
2) **Configurable chunking** — fixed-size with overlap + semantic boundary detection  
3) **Local embeddings** — sentence-transformers, no API costs  
4) **FAISS vector search** — efficient similarity retrieval with persistence  
5) **Local LLM inference** — Mistral 7B via Ollama, fully offline  
6) **Source attribution** — every answer tracks which files it came from  

---

## Chunking Strategies Implemented

| Strategy | How it works | Configuration |
|---|---|---|
| **Fixed-size** | Splits every N tokens with K overlap | `chunk_size=512, chunk_overlap=64` |
| **Semantic** | Splits on meaning shift using embeddings | `breakpoint_threshold=95` (percentile) |

---

## Project Structure

```
rag-eval-system/
├── ingestion/
│   ├── loader.py           # multi-format document loading
│   ├── chunker.py          # fixed + semantic chunking strategies
│   └── embedder.py         # sentence-transformer embeddings + FAISS
├── retrieval/
│   ├── retriever.py        # vector similarity search
│   └── generator.py        # LLM answer generation with Ollama
├── evaluation/             # [Phase 3 - in progress]
│   ├── metrics.py          # RAGAs evaluation framework
│   └── experiments.py      # ablation studies across strategies
├── data/raw/               # document corpus (not tracked)
├── vectorstore/            # persisted FAISS index (not tracked)
└── requirements.txt
```

---

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **Sentence-Transformers** — all-MiniLM-L6-v2 embeddings
- **FAISS** — vector similarity search (CPU)
- **Ollama + Mistral 7B** — local LLM inference (no API costs)
- **RAGAs** — retrieval evaluation metrics *(Phase 3)*
- **Python 3.10+** — core runtime

---

## Installation & Usage

### Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed with Mistral model

### Setup

```bash
# Clone the repo
git clone https://github.com/vabhishekprakash/rag-eval-system.git
cd rag-eval-system

# Install dependencies
pip install -r requirements.txt

# Download Mistral model (one-time, ~4GB)
ollama pull mistral
```

### Add Your Documents

Drop PDF, TXT, MD, or HTML files into `data/raw/`

### Run the Pipeline

```python
from ingestion.loader import load_documents
from ingestion.chunker import get_chunks
from ingestion.embedder import build_vectorstore
from retrieval.retriever import retrieve_context
from retrieval.generator import generate_answer

# Load and process documents
docs = load_documents('data/raw')
chunks = get_chunks(docs, strategy='fixed', chunk_size=512)
vectorstore = build_vectorstore(chunks)

# Ask a question
query = "What is the main topic of these documents?"
context = retrieve_context(query, vectorstore, k=3)
result = generate_answer(query, context)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

---

## Example Output

```
Question: What is the student name?
Answer: The student name is VALLAMALLA ABHISHEK PRAKASH (237Y1A66C5).
Sources: ['Vallamalla-Abhishek-Prakash Resume.pdf']
```

---

## Why This Project Matters

Retrieval quality determines RAG performance, yet most implementations treat chunking as a one-line decision. This project treats it as a research question — systematically measuring how chunking strategy, chunk size, and embedding model affect answer quality on domain-specific corpora.

The evaluation framework (Phase 3) will quantify:
- **Faithfulness** — does the answer stay grounded in retrieved context?
- **Answer Relevancy** — does it actually address the question asked?
- **Context Recall** — did retrieval surface the right information?

---

## Technical Decisions

**local LLM over API:**  
No rate limits, no costs, full privacy, works offline — and more importantly, demonstrates system design skills beyond "call GPT-4 API."

**FAISS over vector databases:**  
Simplicity and portability. The entire vector store is a few files that can be versioned and moved. No server dependencies.

**sentence-transformers over OpenAI embeddings:**  
Same reason — local, free, reproducible. `all-MiniLM-L6-v2` is small (80MB) but performs well for most domains.

---

## Author

**Vallamalla Abhishek Prakash**  
B.Tech CSE (AI & ML) — Final Year  
[GitHub](https://github.com/vabhishekprakash) · [LinkedIn](https://linkedin.com/in/vabhishekprakash)

---

## License

MIT License — free to use, modify, and distribute with attribution.
