import faiss
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vectorstore(chunks, model_name="all-MiniLM-L6-v2", save_path="vectorstore/"):
    """Embed chunks and persist a FAISS index locally."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"Saved {len(chunks)} chunks to {save_path}")
    return vectorstore

def load_vectorstore(model_name="all-MiniLM-L6-v2", save_path="vectorstore/"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)