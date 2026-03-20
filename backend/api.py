import os
import shutil
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel

from config.settings import settings
from document_processing.processor import process_document
from document_processing.insight_engine import generate_insights
from vector_db.store import VectorStore
from vector_db.cache import SemanticCache
from rag_pipeline.retriever import HybridRetriever
from rag_pipeline.generator import RAGPipeline

router = APIRouter()

# Initialize global components
vector_store = VectorStore()
cache = SemanticCache()
retriever = HybridRetriever(vector_store)
rag_pipeline = RAGPipeline(retriever, cache)

# In-memory storage for simple chat history and insights
chat_history = []
document_insights = {}

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    cached: bool

class CompareRequest(BaseModel):
    doc1: str
    doc2: str
    query: str

@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads a document to the local disk and triggers processing asynchronously."""
    file_path = settings.UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    def process_and_index():
        try:
            # Chunking and extraction
            chunks = process_document(str(file_path), settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            # Index into hybrid retriever (FAISS + BM25)
            retriever.index_chunks(chunks)
            # Generate insights
            full_text = " ".join([c["content"] for c in chunks])
            insights = generate_insights(full_text)
            document_insights[file.filename] = insights
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")

    background_tasks.add_task(process_and_index)
    
    return {"message": "File uploaded and processing started.", "filename": file.filename}

@router.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """Processes a natural language query over uploaded documents."""
    result = rag_pipeline.generate_answer(request.query)
    
    chat_history.append({"role": "user", "content": request.query})
    chat_history.append({
        "role": "assistant", 
        "content": result["answer"], 
        "sources": result.get("sources", []),
        "cached": result.get("cached", False)
    })
    
    return result

@router.get("/history")
async def get_history():
    """Returns chat conversation history."""
    return {"history": chat_history}

@router.get("/insights")
async def get_insights():
    """Returns processing insights for all uploaded documents."""
    return {"insights": document_insights}

@router.post("/compare")
async def compare_documents(request: CompareRequest):
    """Answers a comparison question directly targeted at two specific files."""
    # Custom pipeline for comparison logic requiring fetching specific document chunks
    # Not fully hybrid-retrieval based, as we want to constrain the semantic search
    query = f"Compare {request.doc1} and {request.doc2} regarding: {request.query}"
    # Let standard RAG handle it but inject the doc names
    result = rag_pipeline.generate_answer(query)
    return {"answer": result["answer"]}
