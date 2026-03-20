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
        
    full_text = "\n".join([doc.page_content for doc in documents])
    # Detect short documents and don't chunk them at all
    if len(full_text) < 3000:
        # Don't chunk — treat entire document as one chunk
        metadata = documents[0].metadata if documents else {}
        return [Document(page_content=full_text, metadata=metadata)]
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)
