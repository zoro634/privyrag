import os
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embeddings.embedder import get_embedding_model
from config.settings import settings

class VectorStore:
    """FAISS-based vector storage for PrivyRAG."""
    def __init__(self):
        self.persist_dir = str(settings.VECTOR_DB_DIR)
        self.embeddings = get_embedding_model()
        self.index_name = "privyrag_faiss"
        self.db = None
        self._load_or_create()

    def _load_or_create(self):
        index_path = os.path.join(self.persist_dir, self.index_name + ".faiss")
        if os.path.exists(index_path):
            self.db = FAISS.load_local(
                folder_path=self.persist_dir, 
                embeddings=self.embeddings, 
                index_name=self.index_name,
                allow_dangerous_deserialization=True # required for local faiss loading
            )
        else:
            self.db = None
            
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Adds text chunks with metadata to the FAISS store and saves to disk."""
        docs = [Document(page_content=c["content"], metadata=c["metadata"]) for c in chunks]
        if not docs:
            return
            
        if self.db is None:
            self.db = FAISS.from_documents(docs, self.embeddings)
        else:
            self.db.add_documents(docs)
            
        # Persist
        self.db.save_local(self.persist_dir, self.index_name)
        
    def similarity_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Searches the vector db for the most similar chunks.
        Returns a list of tuples containing the Document and the similarity score.
        """
        if self.db is None:
            return []
        
        # Lower score usually means more similar in FAISS L2 distance
        return self.db.similarity_search_with_score(query, k=k)
