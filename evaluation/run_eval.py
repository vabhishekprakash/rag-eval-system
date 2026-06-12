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

# Test questions with ground truth answers
TEST_QUESTIONS = [
    {
        "question": "What is the student name?",
        "ground_truth": "Vallamalla Abhishek Prakash"
    },
    {
        "question": "What is the examination fee amount?",
        "ground_truth": "Fee amount mentioned in the receipt"
    },
    # Add more test questions based on your documents
]

def run_experiment(strategy="fixed", chunk_size=512, chunk_overlap=64):
    """
    Run a complete evaluation experiment with specific parameters.
    
    Returns:
        Dictionary with results and configuration
    """
    print(f"\nRunning experiment: strategy={strategy}, chunk_size={chunk_size}")
    
    # Load and chunk documents
    docs = load_documents('data/raw')
    if strategy == "fixed":
        chunks = get_chunks(docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        chunks = get_chunks(docs, strategy=strategy)
    
    # Build vector store
    vectorstore = build_vectorstore(chunks, save_path=f"vectorstore_{strategy}_{chunk_size}/")
    
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
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(chunks)
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
        print(f"  Context Recall:    {metrics.get('context_recall', 0):.4f}")
    print("="*70)
    
    return results

if __name__ == "__main__":
    compare_strategies()