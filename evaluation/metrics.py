from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

def evaluate_rag_system(questions, answers, contexts, ground_truths):
    """
    Evaluate RAG system using RAGAs metrics with local LLM.
    """
    # Use local Mistral instead of OpenAI
    local_llm = LangchainLLMWrapper(Ollama(model="mistral", temperature=0))
    local_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # RAGAs dataset format
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=local_llm,
        embeddings=local_embeddings
    )

    return result

def format_results(result):
    """Pretty print evaluation results."""
    print("\n" + "="*50)
    print("RAG EVALUATION RESULTS")
    print("="*50)
    # Convert to pandas dataframe first, then extract scores
    result_df = result.to_pandas()
    scores = {
        "faithfulness": result_df["faithfulness"].mean(),
        "answer_relevancy": result_df["answer_relevancy"].mean()
    }
    for metric, score in scores.items():
        print(f"{metric:20s}: {score:.4f}")
    print("="*50 + "\n")
    return scores