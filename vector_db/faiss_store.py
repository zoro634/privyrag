import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import DB_DIR

def get_vector_store(new_documents: list[Document] = None, embeddings: Embeddings = None) -> FAISS:
    """
    Returns a FAISS vector store. If new_documents are provided, it adds them to the existing index
    or builds a brand new index if none exists.
    """
    index_path = os.path.join(DB_DIR, "index.faiss")
    
    if new_documents is not None and len(new_documents) > 0:
        if os.path.exists(index_path):
            print("Documents provided and index exists! Adding to existing FAISS index...")
            vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(new_documents)
            vectorstore.save_local(DB_DIR)
            return vectorstore
        else:
            print("Documents provided! Building FAISS index from scratch...")
            vectorstore = FAISS.from_documents(new_documents, embeddings)
            vectorstore.save_local(DB_DIR)
            return vectorstore
            
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    raise ValueError("No documents provided and no existing FAISS index found.")
