from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os
import shutil
import json
from config.settings import DOCS_DIR
from document_processing.parser import parse_documents
from document_processing.chunker import chunk_documents
from document_processing.insight_engine import extract_insights
from rag_pipeline.orchestrator import answer_query, preload_caches, clear_global_caches
from rag_pipeline.comparator import compare_documents

METADATA_FILE = os.path.join(DOCS_DIR, "metadata.json")

app = FastAPI(title="PrivyRAG API")

class ChatRequest(BaseModel):
    query: str
    selected_docs: list[str] = []

class CompareRequest(BaseModel):
    doc1: str
    doc2: str

class InsightRequest(BaseModel):
    filename: str

@app.on_event("startup")
def startup_event():
    try:
        preload_caches()
        print("Loaded existing index on startup.")
    except Exception as e:
        print("No index found or failed to load on startup:", e)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), doc_type: str = Form("General")):
    file_path = os.path.join(DOCS_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Update metadata.json
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
            
        metadata = {}
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        metadata[file.filename] = {"type": doc_type}
        
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)
            
        from document_processing.parser import parse_document, parse_documents
        new_docs = parse_document(file_path)
        new_chunks = chunk_documents(new_docs)
        
        all_docs = parse_documents(str(DOCS_DIR))
        all_chunks = chunk_documents(all_docs)
        
        from rag_pipeline.retriever import get_hybrid_retriever
        from rag_pipeline.llm import get_embeddings
        get_hybrid_retriever(new_documents=new_chunks, all_documents=all_chunks, embeddings=get_embeddings())
        
        clear_global_caches()
        
        return {"message": f"Successfully uploaded {file.filename} and updated knowledge base."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = answer_query(request.query, request.selected_docs)
        
        context_sources = []
        if "context" in result and result["context"]:
            for doc in result["context"]:
                source = doc.metadata.get("source", "Unknown")
                if isinstance(source, str):
                    source = os.path.basename(source)
                context_sources.append({
                    "source": source,
                    "content": doc.page_content
                })
                
        return {
            "answer": result["answer"],
            "cached": result.get("cached", False),
            "sources": context_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare(request: CompareRequest):
    try:
        result = compare_documents(request.doc1, request.doc2)
        return {"comparison": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insights")
async def insights(request: InsightRequest):
    try:
        docs = parse_documents(str(DOCS_DIR))
        target_docs = [d for d in docs if str(d.metadata.get('source', '')).endswith(request.filename)]
        if not target_docs:
            raise HTTPException(status_code=404, detail=f"File {request.filename} not found.")
            
        full_text = "\n".join([d.page_content for d in target_docs])
        insights_result = extract_insights(full_text)
        return {"insights": insights_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    if not os.path.exists(DOCS_DIR):
        return {"files": []}
        
    # Read metadata if it exists
    metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                pass
                
    files = []
    for f in os.listdir(DOCS_DIR):
        if f == "metadata.json" or f.startswith("."):
            continue
        doc_type = metadata.get(f, {}).get("type", "General")
        files.append({"name": f, "type": doc_type})
        
    return {"files": files}
