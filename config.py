import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5442/chatbot")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "C9PE94QUEW9VWGFM")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

CHAT_MODEL = "llama-3.1-8b-instant"
MEMORY_MODEL = "llama-3.3-70b-versatile"

SUMMARIZE_THRESHOLD = 6
DOCS_DIR = "docs"
FAISS_INDEX_PATH = "faiss_index"