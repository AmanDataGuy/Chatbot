from langchain_core.tools import tool
from rag.vectorstore import get_retriever


@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the PDF documents stored in docs/.
    Use this tool when the user asks factual or conceptual questions
    that might be answered from the stored documents.
    """
    retriever = get_retriever()

    if retriever is None:
        return {
            "query": query,
            "context": [],
            "metadata": [],
            "message": "No documents have been uploaded yet. Please add PDFs to the docs/ folder.",
        }

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
    }