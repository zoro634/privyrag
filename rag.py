import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

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
    
    # Load all PDFs and process chunks
    splits = []
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(pdf_file)
        raw_docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in raw_docs])
        
        # Detect short documents and don't chunk them at all
        if len(full_text) < 3000:
            metadata = raw_docs[0].metadata if raw_docs else {}
            splits.append(Document(page_content=full_text, metadata=metadata))
        else:
            # Normal chunking for larger documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits.extend(text_splitter.split_documents(raw_docs))
    
    # Generate embeddings using OllamaEmbeddings with model nomic-embed-text
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Store vectors in ChromaDB with local persistence
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name="privy_rag_store"
    )
    
    # Build a retriever that fetches dynamically based on chunk count
    total_chunks = len(splits)
    k = total_chunks if total_chunks <= 10 else 5
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # Configure the system Prompt exactly as requested
    system_prompt = (
        "You are a private, secure document assistant. "
        "Answer based on the document context provided. "
        "If the exact answer is not stated, you may make reasonable inferences directly supported by the document content, but clearly state you are inferring. "
        "Only say 'I could not find this in the provided documents.' if the topic is completely absent from the document.\n\n"
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
