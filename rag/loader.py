import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCS_DIR


def load_all_pdfs():
    """
    Load and chunk all PDFs from the docs/ folder.
    """
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DOCS_DIR, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"Loaded {filename} → {len(chunks)} chunks")

    return all_chunks