import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    # Load environment variables from .env file

    # LLM API Keys
    GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
    GROK_MODEL="llama-3.3-70b-versatile"
    OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")


    GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
    
    #Vector Database Settings
    QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_COLLECTION = "enterprise_rag"  
    
    # Observability Settings
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

settings = Settings()