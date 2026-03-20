from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config.settings import LLM_MODEL

def extract_insights(text: str) -> str:
    """
    Extracts a quick summary and key entities from the provided text.
    """
    try:
        llm = ChatOllama(model=LLM_MODEL)
        
        system_prompt = (
            "You are an expert document analyst. Extract concise insights from the provided text. "
            "Return the result exactly as a markdown summary containing: "
            "\n1. Summary (2-3 sentences)\n2. Key Entities (list format)"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")
        ])
        
        chain = prompt | llm
        
        # Limit text length to prevent context bounds errors
        truncated_text = text[:4000]
        
        res = chain.invoke({"text": truncated_text})
        return res.content
    except Exception as e:
        return f"Failed to extract insights: {str(e)}"
