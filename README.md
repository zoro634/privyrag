# PrivyRAG 🔒

PrivyRAG is a privacy-preserving offline document intelligence system. All document parsing, embeddings generation, vector storage, and QA generation runs entirely on your local machine using Ollama and local storage.

## Features
- **Local Everything:** No data ever leaves your computer.
- **Support for multiple formats:** PDF, DOCX, TXT, and MD.
- **Intelligent Chunking & Source Citations:** Answers include exact page and file sources.
- **Hybrid Search:** Combines semantic search (FAISS) with keyword search (BM25) using Reciprocal Rank Fusion.
- **Document Insights:** Automatically extracts summary, key entities, and risk indicators.
- **Semantic Caching:** Instantly answers previously asked identical questions.
- **Comparison Tool:** Compare two distinct documents directly.

## Prerequisites
1. **Python 3.10+**
2. **Ollama:** Installed and running locally. (https://ollama.com/)

## Installation

1. Create a virtual environment and activate it:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Pull Ollama Models:**
Before running the app, ensure you have pulled the required LLM and Embedding models in Ollama:
```bash
python models/pull_models.py
```
*(You can also use bge-small or mistral by modifying `config/settings.py`)*

## Running the Application

To start both the FastAPI backend and the Streamlit frontend concurrently, simply run:

```bash
python run.py
```

- **Frontend Interface:** Available at `http://localhost:8501`
- **Backend API Docs:** Available at `http://localhost:8000/docs`

## Folder Structure
- `/backend`: FastAPI endpoints for all functions.
- `/config`: Configuration and environments settings.
- `/document_processing`: Parsers (PDF, DOCX, TXT), Chunking, Insight Extraction.
- `/embeddings`: Ollama embedding configurations.
- `/frontend`: Streamlit unified interface.
- `/models`: Scripts/info for model management.
- `/rag_pipeline`: Hybrid Retriever, LLM orchestration, Context injection.
- `/vector_db`: FAISS vector store and Semantic Cache.
