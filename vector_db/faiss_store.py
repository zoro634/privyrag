import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import DB_DIR

def get_vector_store(documents: list[Document] = None, embeddings: Embeddings = None) -> FAISS:
    """
    Returns a FAISS vector store. If documents are provided, it builds a brand new index 
    from scratch and saves it. If no documents are provided, it loads the existing index.
    """
    index_path = os.path.join(DB_DIR, "index.faiss")
    
    if documents is not None and len(documents) > 0:
        print("Documents provided! Rebuilding FAISS index from scratch...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(DB_DIR)
        return vectorstore
        
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    raise ValueError("No documents provided and no existing FAISS index found.")
