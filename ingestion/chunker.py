from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def fixed_chunker(docs, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["chunk_strategy"] = "fixed"
        c.metadata["chunk_size"] = chunk_size
    return chunks

def semantic_chunker(docs, model_name="all-MiniLM-L6-v2", breakpoint_threshold=95):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["chunk_strategy"] = "semantic"
    return chunks

def get_chunks(docs, strategy="fixed", **kwargs):
    if strategy == "fixed":
        return fixed_chunker(docs, **kwargs)
    elif strategy == "semantic":
        return semantic_chunker(docs, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")