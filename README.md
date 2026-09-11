# RAG Evaluation System

A local Retrieval-Augmented Generation (RAG) pipeline built to compare chunking strategies and measure answer quality with RAGAs.

---

## What This Project Does

Most RAG tutorials stop at "it works." This project asks how well it works, and why.

The system ingests a folder of documents, chunks them with different strategies, embeds the chunks into a FAISS vector store, and answers questions with a local LLM. Chunk size and chunking strategy are treated as experimental variables, and an evaluation harness scores each configuration on the same questions.

---

## Evaluation Status

`evaluation/run_eval.py` runs four configurations (fixed-size chunks of 256, 512 and 1024 characters, and semantic chunking) and scores each one with RAGAs faithfulness and answer relevancy, using the local Mistral model as the judge.

An earlier run used only two test questions over personal documents. That is too few to compare strategies, so those numbers have been removed. Results will be published here once a public evaluation set and corpus are committed, together with the command that produces them.

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

1) **Multi-format document loading:** PDF, TXT, MD and HTML, with automatic format detection
2) **Configurable chunking:** fixed-size with overlap, plus semantic boundary detection
3) **Local embeddings:** sentence-transformers, no API costs
4) **FAISS vector search:** similarity retrieval with a persisted index
5) **Local LLM inference:** Mistral 7B via Ollama, fully offline
6) **Source attribution:** every answer lists the files it came from
7) **Streamlit app:** ask questions and inspect the retrieved chunks in a browser

---

## Chunking Strategies Implemented

| Strategy | How it works | Configuration |
|---|---|---|
| **Fixed-size** | Splits text into chunks of up to N characters, with K characters of overlap | `chunk_size=512, chunk_overlap=64` |
| **Semantic** | Splits where the meaning shifts, using embeddings | `breakpoint_threshold=95` (percentile) |

---

## Project Structure

```
rag-eval-system/
├── app/
│   └── streamlit_app.py    # browser UI for questions and retrieved chunks
├── ingestion/
│   ├── loader.py           # multi-format document loading
│   ├── chunker.py          # fixed + semantic chunking strategies
│   └── embedder.py         # sentence-transformer embeddings + FAISS
├── retrieval/
│   ├── retriever.py        # vector similarity search
│   └── generator.py        # LLM answer generation with Ollama
├── evaluation/
│   ├── metrics.py          # RAGAs evaluation (faithfulness, answer relevancy)
│   └── run_eval.py         # runs and compares the four chunking configurations
├── data/raw/               # document corpus (not tracked)
├── vectorstore/            # persisted FAISS index (not tracked)
└── requirements.txt
```

---

## Tech Stack

- **LangChain:** RAG pipeline orchestration
- **Sentence-Transformers:** all-MiniLM-L6-v2 embeddings
- **FAISS:** vector similarity search (CPU)
- **Ollama + Mistral 7B:** local LLM inference and evaluation judge
- **RAGAs:** faithfulness and answer relevancy metrics
- **Streamlit:** browser UI
- **Python 3.10+**

---

## Installation & Usage

### Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed with the Mistral model

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

Create a `data/raw/` folder in the project root and put your PDF, TXT, MD or HTML files in it. This folder is excluded from the repository so private documents never get committed.

```bash
mkdir -p data/raw
```

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

### Run the Evaluation

From the project root:

```bash
python evaluation/run_eval.py
```

This builds one vector store per configuration, answers the test questions in `run_eval.py`, scores them with RAGAs and writes `evaluation_results.json`.

### Run the App

```bash
streamlit run app/streamlit_app.py
```

---

## Why This Project Matters

Retrieval quality decides how good a RAG system's answers are, yet most implementations treat chunking as a one-line decision. This project treats it as an experiment: it measures how chunking strategy and chunk size affect answer quality on the same set of questions.

---

## Technical Decisions

**Local LLM over an API:**
No rate limits, no costs, full privacy, and it works offline. The same local model also acts as the RAGAs judge, so the evaluation needs no API key either.

**FAISS over a vector database:**
Simplicity and portability. The whole vector store is a few files that can be versioned and moved, with no server to run.

**sentence-transformers over OpenAI embeddings:**
Local, free and reproducible. `all-MiniLM-L6-v2` is small (about 80 MB) and performs well for most domains.

---

## Author

**Vallamalla Abhishek Prakash**
B.Tech CSE (AI & ML), final year
[GitHub](https://github.com/vabhishekprakash) · [LinkedIn](https://www.linkedin.com/in/vallamalla-abhishek-prakash)

---

## License

MIT License. See [LICENSE](LICENSE).
