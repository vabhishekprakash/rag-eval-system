import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from ingestion.loader import load_documents
# ... rest of the imports
from ingestion.chunker import get_chunks
from ingestion.embedder import build_vectorstore, load_vectorstore
from retrieval.retriever import retrieve_context, load_retriever
from retrieval.generator import generate_answer
from evaluation.metrics import evaluate_rag_system, format_results
from langchain_community.llms import Ollama

# Test questions with ground truth answers
EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
with open(EVAL_SET_PATH, encoding="utf-8") as f:
    TEST_QUESTIONS = json.load(f)["questions"]

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus")

def run_experiment(strategy="fixed", chunk_size=512, chunk_overlap=64):
    """
    Run a complete evaluation experiment with specific parameters.
    
    Returns:
        Dictionary with results and configuration
    """
    if strategy == "fixed":
        print(f"\nRunning experiment: strategy={strategy}, chunk_size={chunk_size}")
    else:
        print(f"\nRunning experiment: strategy={strategy}")
    
    # Load and chunk documents
    docs = load_documents(CORPUS_DIR)
    # data/corpus also holds NOTICE.md; only the PDFs are the corpus
    docs = [d for d in docs if d.metadata["source_file"].endswith(".pdf")]
    if strategy == "fixed":
        chunks = get_chunks(docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        chunks = get_chunks(docs, strategy=strategy)
    
    # Build vector store
    store_name = f"vectorstore_{strategy}_{chunk_size}" if strategy == "fixed" else f"vectorstore_{strategy}"
    vectorstore = build_vectorstore(chunks, save_path=f"{store_name}/")
    
    # Collect results for all test questions
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for test in TEST_QUESTIONS:
        query = test["question"]
        
        # Retrieve and generate
        retrieved_docs = retrieve_context(query, vectorstore, k=3)
        result = generate_answer(query, retrieved_docs)
        
        # Store for evaluation
        questions.append(query)
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(test["ground_truth"])
    
    # Evaluate
    eval_result = evaluate_rag_system(questions, answers, contexts, ground_truths)
    formatted = format_results(eval_result)
    
    # Return config + results
    return {
        "config": {
            "strategy": strategy,
            "chunk_size": chunk_size if strategy == "fixed" else None,
            "chunk_overlap": chunk_overlap if strategy == "fixed" else None,
            "num_chunks": len(chunks)
        },
        "metrics": formatted
    }

def run_baseline(model_name="mistral"):
    """
    Answer every question from the model alone: no retrieval, no context, and
    no instruction to stay within a context. Same model and temperature as
    generate_answer, so the chunking strategies have a floor to beat.
    """
    print("\nRunning baseline: no retrieval")

    llm = Ollama(model=model_name, temperature=0.1)

    questions = []
    answers = []
    ground_truths = []

    for test in TEST_QUESTIONS:
        query = test["question"]
        questions.append(query)
        answers.append(llm.invoke(query).strip())
        ground_truths.append(test["ground_truth"])

    # No retrieval means no context; faithfulness has nothing to check against
    contexts = [[""] for _ in questions]

    eval_result = evaluate_rag_system(questions, answers, contexts, ground_truths)
    formatted = format_results(eval_result)

    return {
        "config": {
            "strategy": "none",
            "chunk_size": None,
            "chunk_overlap": None,
            "num_chunks": 0
        },
        "metrics": formatted
    }

def compare_strategies():
    """
    Run ablation study comparing different chunking strategies.
    """
    experiments = [
        {"strategy": "fixed", "chunk_size": 256, "chunk_overlap": 32},
        {"strategy": "fixed", "chunk_size": 512, "chunk_overlap": 64},
        {"strategy": "fixed", "chunk_size": 1024, "chunk_overlap": 128},
        {"strategy": "semantic"},
    ]
    
    results = []
    for exp in experiments:
        result = run_experiment(**exp)
        results.append(result)
    results.append(run_baseline())

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    for r in results:
        cfg = r["config"]
        metrics = r["metrics"]
        print(f"\nStrategy: {cfg['strategy']}, Chunk Size: {cfg.get('chunk_size', 'N/A')}")
        print(f"  Faithfulness:      {metrics.get('faithfulness', 0):.4f}")
        print(f"  Answer Relevancy:  {metrics.get('answer_relevancy', 0):.4f}")
        
    print("="*70)
    
    return results

if __name__ == "__main__":
    compare_strategies()