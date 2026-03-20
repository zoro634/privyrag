import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "vector_db" / "faiss_index"

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Document Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retrieval Settings
TOP_K = 3

# Model Settings
EMBEDDING_MODEL = "all-minilm"
LLM_MODEL = "gemma2:2b"
