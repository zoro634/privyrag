from langchain_core.prompts import ChatPromptTemplate
from rag_pipeline.llm import get_llm
from config.settings import DOCS_DIR
import os

def compare_documents(doc1_name: str, doc2_name: str) -> str:
    from document_processing.parser import parse_documents
    docs = parse_documents(str(DOCS_DIR))
    
    doc1_content = "\n".join([d.page_content for d in docs if str(d.metadata.get('source', '')).endswith(doc1_name)])
    doc2_content = "\n".join([d.page_content for d in docs if str(d.metadata.get('source', '')).endswith(doc2_name)])
    
    if not doc1_content: return f"Could not find or read {doc1_name}"
    if not doc2_content: return f"Could not find or read {doc2_name}"
        
    doc1_content = doc1_content[:5000]
    doc2_content = doc2_content[:5000]
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare the following two documents. Highlight the main differences, similarities, and provide a clear synthesis."),
        ("human", "Document 1 ({doc1_name}):\n{doc1_content}\n\nDocument 2 ({doc2_name}):\n{doc2_content}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({
        "doc1_name": doc1_name, 
        "doc1_content": doc1_content, 
        "doc2_name": doc2_name, 
        "doc2_content": doc2_content
    })
    return res.content
