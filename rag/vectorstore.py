import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import FAISS_INDEX_PATH, DOCS_DIR
from rag.loader import load_all_pdfs


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vectorstore():
    """
    Build FAISS index from all PDFs in docs/ and save to disk.
    Returns None if no PDFs are found.
    """
    chunks = load_all_pdfs()

    if not chunks:
        print("No PDF chunks found. Skipping FAISS index build.")
        return None

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    return vector_store


def load_vectorstore():
    """
    Load existing FAISS index from disk.
    Build it first if it doesn't exist.
    Returns None if no documents are available.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print(f"FAISS index loaded from {FAISS_INDEX_PATH}")
        return vector_store

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")] if os.path.exists(DOCS_DIR) else []

    if not pdf_files:
        print("No PDFs in docs/ and no saved index. RAG is disabled.")
        return None

    print("No FAISS index found, building from docs/...")
    return build_vectorstore()


def get_retriever():
    """
    Returns a retriever, or None if no documents have been loaded yet.
    """
    vector_store = load_vectorstore()
    if vector_store is None:
        return None
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def rebuild_vectorstore():
    """
    Force rebuild the FAISS index from scratch. Called after new PDFs are added.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        import shutil
        shutil.rmtree(FAISS_INDEX_PATH)
    return build_vectorstore()