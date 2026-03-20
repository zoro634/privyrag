import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_qa_chain():
    """
    Initializes documents, creates embeddings, and returns a initialized QA chain.
    """
    docs_dir = "./docs"
    db_dir = "./chroma_db"
    
    # Retrieve all PDFs in the docs folder
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{docs_dir}' directory.")
    
    # Load all PDFs
    documents = []
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(pdf_file)
        documents.extend(loader.load())
        
    # Split text into chunks of size 500 with overlap 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    # Generate embeddings using OllamaEmbeddings with model nomic-embed-text
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Store vectors in ChromaDB with local persistence
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name="privy_rag_store"
    )
    
    # Build a retriever that fetches the top 4 chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Configure the system Prompt exactly as requested
    system_prompt = (
        "You are a private, secure document assistant. "
        "Answer only based on the provided document context. "
        "If the answer is not in the document, say 'I could not find this in the provided documents.' "
        "Do not use any outside knowledge.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Initialize the LLM: phi3:mini
    llm = ChatOllama(model="phi3:mini")
    
    # Build the RetrievalQA chain
    # This chain returns both the answer and the source documents
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return qa_chain
