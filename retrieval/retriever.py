from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def retrieve_context(query, vectorstore, k=5):
    """
    Retrieve top-k most relevant chunks for a query.
    
    Args:
        query: user's question as string
        vectorstore: loaded FAISS index
        k: number of chunks to retrieve
    
    Returns:
        list of Document objects with page_content and metadata
    """
    results = vectorstore.similarity_search(query, k=k)
    return results

def load_retriever(model_name="all-MiniLM-L6-v2", vectorstore_path="vectorstore/"):
    """Load the vector store and return it ready for retrieval."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(
        vectorstore_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vectorstore