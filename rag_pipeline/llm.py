from langchain_ollama import OllamaEmbeddings, ChatOllama
from config.settings import EMBEDDING_MODEL, LLM_MODEL

def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.1,
        num_ctx=2048,
        num_predict=512,
        num_thread=4,
        repeat_penalty=1.1
    )
