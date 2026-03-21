from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader
)

def load_documents(data_dir: str) -> list:
    docs = []
    for path in Path(data_dir).rglob("*"):
        if path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix in [".txt", ".md"]:
            loader = TextLoader(str(path))
        elif path.suffix == ".html":
            loader = UnstructuredHTMLLoader(str(path))
        else:
            continue
        loaded = loader.load()
        # attach source metadata to every page
        for doc in loaded:
            doc.metadata["source_file"] = path.name
        docs.extend(loaded)
    print(f"Loaded {len(docs)} document pages from {data_dir}")
    return docs