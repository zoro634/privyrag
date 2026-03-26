from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_document(documents: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    return chunk_documents(documents, chunk_size, chunk_overlap)

def chunk_documents(documents: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """
    Splits documents into smaller chunks for vector storage.
    """
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    final_chunks = []
    for doc in documents:
        if len(doc.page_content) <= chunk_size:
            final_chunks.append(doc)
        else:
            final_chunks.extend(text_splitter.split_documents([doc]))
            
    return final_chunks
