import os
import pickle
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from vector_db.faiss_store import get_vector_store
from config.settings import TOP_K, DB_DIR

BM25_PATH = os.path.join(DB_DIR, "bm25.pkl")

def get_hybrid_retriever(new_documents: list[Document] = None, all_documents: list[Document] = None, embeddings: Embeddings = None):
    """
    Creates an EnsembleRetriever combining FAISS (semantic) and BM25 (keyword) search.
    If new_documents are provided, builds or appends to FAISS index. 
    If all_documents are provided, rebuilds BM25 index.
    """
    # Initialize FAISS Retriever
    faiss_vectorstore = get_vector_store(new_documents, embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Initialize BM25 Retriever
    if os.path.exists(BM25_PATH) and all_documents is None:
        with open(BM25_PATH, 'rb') as f:
            bm25_retriever = pickle.load(f)
    elif all_documents:
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        with open(BM25_PATH, 'wb') as f:
            pickle.dump(bm25_retriever, f)
    else:
        raise ValueError("No existing index found and no documents provided.")
        
    bm25_retriever.k = TOP_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever
