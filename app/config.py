import os
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith needs these set as actual env vars at startup ──
os.environ["LANGCHAIN_API_KEY"]      = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"]   = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"]      = os.getenv("LANGCHAIN_PROJECT", "default")

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_WHATSAPP_NUMBER: str = os.getenv("TWILIO_WHATSAPP_NUMBER")

    # LangSmith
    LANGCHAIN_API_KEY: str     = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2: str  = os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_PROJECT: str     = os.getenv("LANGCHAIN_PROJECT")

settings = Settings()