from langchain_community.llms import Ollama

def generate_answer(query, context_docs, model_name="mistral"):
    """
    Generate an answer using retrieved context and a local LLM.
    
    Args:
        query: user's question
        context_docs: list of retrieved Document objects
        model_name: Ollama model to use (mistral, llama2, etc.)
    
    Returns:
        Generated answer as string
    """
    # Combine all retrieved chunks into one context block
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Get sources for attribution
    sources = [doc.metadata.get("source_file", "unknown") for doc in context_docs]
    unique_sources = list(set(sources))
    
    # Build the prompt directly
    prompt = f"""You are a helpful assistant answering questions based on provided context.

Context:
{context}

Question: {query}

Answer the question using ONLY the information in the context above. If the context doesn't contain enough information to answer, say so. Be concise and specific.

Answer:"""
    
    # Initialize Ollama LLM
    llm = Ollama(model=model_name, temperature=0.1)
    
    # Generate answer
    answer = llm.invoke(prompt)
    
    return {
        "answer": answer.strip(),
        "sources": unique_sources,
        "num_chunks_used": len(context_docs)
    }