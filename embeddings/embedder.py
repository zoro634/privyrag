from langchain_community.embeddings import OllamaEmbeddings
from config.settings import settings

def get_embedding_model() -> OllamaEmbeddings:
    """
    Returns an instance of the Ollama-based embedding model.
    Throws an error if Ollama is not accessible on the provided BASE_URL.
    """
    return OllamaEmbeddings(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.EMBEDDING_MODEL
    )
