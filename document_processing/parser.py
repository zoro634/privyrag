import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

def parse_document(filepath: str) -> list[Document]:
    """Parses a single supported document file."""
    documents = []
    if not os.path.isfile(filepath):
        return documents
        
    filename = os.path.basename(filepath)
    ext = filename.lower().split('.')[-1]
    try:
        if ext == 'pdf':
            loader = PyMuPDFLoader(filepath)
            documents.extend(loader.load())
        elif ext == 'docx':
            loader = Docx2txtLoader(filepath)
            documents.extend(loader.load())
        elif ext in ['txt', 'md']:
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        
    return documents

def parse_documents(docs_dir: str) -> list[Document]:
    """
    Parses all supported documents (PDF, DOCX, TXT, MD) in the given directory.
    """
    documents = []
    
    if not os.path.exists(docs_dir):
        return documents
        
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        documents.extend(parse_document(filepath))
            
    return documents
