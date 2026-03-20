import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from rag_pipeline.llm import get_llm
from vector_db.cache import SemanticCache
from rag_pipeline.retriever import HybridRetriever

class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, cache: SemanticCache):
        self.retriever = retriever
        self.cache = cache
        self.llm = get_llm()
        
    def decompose_query(self, query: str) -> List[str]:
        """Splits complex queries into multiple simpler sub-queries if needed."""
        prompt = f"""You are an AI assistant. Evaluate the following user query. If it asks multiple distinct questions or contains multiple complex parts, split them into independent sub-queries, one per line. If it is a single simple question, just return the exact query.
Query: {query}
Result:"""
        try:
            result = self.llm.invoke(prompt)
            lines = [line.strip() for line in result.split('\\n') if line.strip()]
            # Clean up potential numbering
            lines = [re.sub(r'^[0-9\\-\\.\\*]+\\s*', '', line).strip() for line in lines]
            if not lines:
                return [query]
            return lines
        except Exception as e:
            print(f"Query decomposition failed: {e}")
            return [query]
            
    def format_context(self, docs: List[Document]) -> str:
        """Formats docs for LLM injection."""
        context = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context += f"\\n--- Chunk {i+1} ---\\n[Source: {source}, Page: {page}]\\n{doc.page_content}\\n"
        return context
        
    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Main pipeline: Cache -> Decompose -> Retrieve -> Generate -> Cache."""
        # 1. Check Cache
        cached_answer = self.cache.get(query)
        if cached_answer:
            return {"answer": cached_answer, "sources": [], "cached": True}
            
        # 2. Decompose query
        sub_queries = self.decompose_query(query)
        
        # 3. Retrieve for all sub-queries
        all_retrieved_docs = []
        for sq in sub_queries:
            # Top 10 chunks per query based on Prompt instructions
            docs = self.retriever.retrieve(sq, top_k=10)
            all_retrieved_docs.extend(docs)
            
        # Deduplicate docs based on content
        unique_docs = {hash(doc.page_content): doc for doc in all_retrieved_docs}.values()
        final_docs = list(unique_docs)[:10]  # Return top 10 unique chunks overall
        
        # 4. Generate Answer
        context = self.format_context(final_docs)
        
        prompt = f"""You are PrivyRAG, a privacy-respecting offline AI assistant. 
Use the following context to answer the user's question accurately. 
If you use a specific fact, you MUST include a citation indicating the source and page number in your answer. Do not say 'According to the context', just state the facts and append the citation like [Source: filename.pdf, Page: 2].
If the answer is not contained in the context, say "I don't have enough information to answer that based on the uploaded documents."

Context:
{context}

Question: {query}
Answer:"""

        try:
            answer = self.llm.invoke(prompt)
        except Exception as e:
            answer = f"Error generating answer: {e}"
        
        # Extract sources from docs
        sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page"), "text": d.page_content} for d in final_docs]
        
        # 5. Set Cache
        self.cache.set(query, answer)
        
        return {"answer": answer, "sources": sources, "cached": False}
