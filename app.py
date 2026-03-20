import os
import chainlit as cl
from rag import get_qa_chain

@cl.on_chat_start
async def on_chat_start():
    try:
        # Inform the user that initialization is happening
        msg = cl.Message(content="Initializing PrivyRAG... Loading documents and creating embeddings. This may take a moment...", disable_feedback=True)
        await msg.send()
        
        # Call get_qa_chain() to set up the RAG pipeline
        qa_chain = get_qa_chain()
        
        # Store the chain in the user session
        cl.user_session.set("qa_chain", qa_chain)
        
        msg.content = "PrivyRAG is ready! Ask me anything about your documents."
        await msg.update()
        
    except FileNotFoundError as e:
        await cl.Message(content=f"⚠️ Initialization Error: {str(e)}\nPlease place some PDF files in the `./docs` folder and restart.").send()
    except Exception as e:
        await cl.Message(content=f"⚠️ An unexpected error occurred during initialization:\n```\n{str(e)}\n```\nMake sure Ollama is running and required models are pulled.").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the qa_chain from the user session
    qa_chain = cl.user_session.get("qa_chain")
    
    if not qa_chain:
        await cl.Message(content="The QA chain is not initialized. Please ensure documents are placed in `./docs` and restart the app.").send()
        return

    # Send a waiting message
    res_msg = cl.Message(content="Thinking...")
    await res_msg.send()

    try:
        # Run the question through the QA chain
        res = await qa_chain.ainvoke({"input": message.content})
        
        answer = res.get("answer", "No answer generated.")
        source_documents = res.get("context", [])
        
        # Prepare the citations text
        citations = []
        if source_documents:
            citations.append("\n\n---\n**Citations:**\n")
            for doc in source_documents:
                # Extract filename from metadata if available, otherwise default to "Unknown"
                source = doc.metadata.get("source", "Unknown")
                if "/" in source or "\\" in source:
                    source = os.path.basename(source)
                content = doc.page_content.strip()
                # Format clearly as requested
                citations.append(f"📄 Source: {source}\n> \"{content}\"\n")
                
        # Update message with the answer and citations
        res_msg.content = answer + "\n".join(citations)
        await res_msg.update()
        
    except Exception as e:
        # Handle errors gracefully from the LLM/chain
        err_msg = f"⚠️ An error occurred while processing your request:\n```\n{str(e)}\n```\nPlease make sure Ollama is running properly."
        res_msg.content = err_msg
        await res_msg.update()
