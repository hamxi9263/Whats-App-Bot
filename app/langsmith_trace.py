# test_langsmith.py  — run this file directly: python test_langsmith.py
from app.config import settings        # sets env vars first
from langsmith import traceable
from langsmith import Client

# Test 1: Check connection
client = Client()
print("LangSmith connected:", client.list_projects())

# Test 2: Send a dummy trace
@traceable(run_type="chain", name="Test-Trace")
def dummy_trace(msg: str) -> str:
    return f"received: {msg}"

result = dummy_trace("hello langsmith")
print("Trace sent! Check smith.langchain.com → WhatsApp-Agent project")
print("Result:", result)