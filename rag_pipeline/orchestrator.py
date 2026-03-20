from functools import lru_cache
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from rag_pipeline.llm import get_llm
from vector_db.cache import get_cached_response, set_cached_response

# Global singletons for memory
CACHE = {}

def preload_caches():
    from vector_db.faiss_store import get_vector_store
    from rag_pipeline.llm import get_embeddings
    import pickle, os
    from config.settings import DB_DIR
    CACHE["vs"] = get_vector_store(embeddings=get_embeddings())
    with open(os.path.join(DB_DIR, "bm25.pkl"), 'rb') as f:
        CACHE["bm25"] = pickle.load(f)

def clear_global_caches():
    CACHE.clear()

class PostFilterRetriever(BaseRetriever):
    retriever: BaseRetriever
    allowed_sources: list[str] = Field(default_factory=list)
    
    def _get_relevant_documents(self, query: str, *, run_manager=None):
        docs = self.retriever.invoke(query)
        if not self.allowed_sources: 
            return docs[:get_top_k()]
        filtered = [d for d in docs if any(sd in str(d.metadata.get("source", "")) for sd in self.allowed_sources)]
        return filtered[:get_top_k()]

def get_top_k():
    from config.settings import TOP_K
    return TOP_K

def build_dynamic_chain(selected_docs: list[str]):
    from config.settings import TOP_K
    from langchain.retrievers import EnsembleRetriever
    
    if "vs" not in CACHE:
        preload_caches()
        
    vs = CACHE["vs"]
    bm25 = CACHE["bm25"]
    
    # Optional FAISS filtering (native)
    search_kwargs = {"k": TOP_K * 2}
    if selected_docs:
        search_kwargs["filter"] = lambda metadata: any(sd in str(metadata.get("source", "")) for sd in selected_docs)
        
        # Document length awareness - dynamic k for short documents
        doc_chunk_count = sum(1 for d in vs.docstore._dict.values() if any(sd in str(d.metadata.get("source", "")) for sd in selected_docs))
        if doc_chunk_count > 0:
            search_kwargs["k"] = doc_chunk_count if doc_chunk_count <= 10 else max(5, TOP_K * 2)
        
    faiss_retriever = vs.as_retriever(search_kwargs=search_kwargs)
    ensemble = EnsembleRetriever(retrievers=[faiss_retriever, bm25], weights=[0.5, 0.5])
    
    # Wrap in our robust Post-Filter to sanitize BM25 leakage
    filtered_retriever = PostFilterRetriever(retriever=ensemble, allowed_sources=selected_docs)
    
    llm = get_llm()
    system_prompt = (
        "You are a private, secure document assistant. "
        "Answer based on the document context provided. "
        "If the exact answer is not stated, you may make reasonable inferences directly supported by the document content, but clearly state you are inferring. "
        "Only say 'I could not find this in the provided documents.' if the topic is completely absent from the document.\n"
        "Do not use any outside knowledge.\n"
        "IMPORTANT: You must include inline citations in your answer! Whenever you state a fact from the context, append the source filename in brackets (e.g., [document.pdf]).\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    from langchain_core.prompts import PromptTemplate
    doc_prompt = PromptTemplate.from_template("Source: {source}\n{page_content}")
    
    return create_retrieval_chain(filtered_retriever, create_stuff_documents_chain(llm, prompt, document_prompt=doc_prompt))

def answer_query(query: str, selected_docs: list[str]) -> dict:
    cached = get_cached_response(query)
    if cached:
        return {"answer": cached, "context": [], "cached": True}
        
    chain = build_dynamic_chain(selected_docs)
    result = chain.invoke({"input": query})
    
    set_cached_response(query, result["answer"])
    result["cached"] = False
    return result
