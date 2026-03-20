import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

def parse_documents(docs_dir: str) -> list[Document]:
    """
    Parses all supported documents (PDF, DOCX, TXT, MD) in the given directory.
    """
    documents = []
    
    if not os.path.exists(docs_dir):
        return documents
        
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if not os.path.isfile(filepath):
            continue
            
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
            else:
                pass # ignore other extensions
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return documents
