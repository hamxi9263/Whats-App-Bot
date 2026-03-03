from httpx import AsyncClient
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from app.config import settings
from langsmith import traceable
from app.rag_service import RAGService
from app.rag_service_unstructured import UnstructuredRAGService
from datetime import datetime
import pytz

http_client = AsyncClient(timeout=30)

groq_provider = GroqProvider(
    api_key=settings.GROQ_API_KEY,
    http_client=http_client,
)

model = GroqModel(
    model_name="llama-3.3-70b-versatile",
    provider=groq_provider,
)

# ── Both RAG services ──────────────────────────────────────────────────────────
structured_rag   = RAGService()
unstructured_rag = UnstructuredRAGService()

# ── Agent ──────────────────────────────────────────────────────────────────────
agent = Agent(
    model=model,
    system_prompt = """
You are Sara, a friendly and professional AI receptionist at HairRevive Clinic — 
a leading hair transplant and restoration center in Pakistan.

YOUR ROLE:
- Help patients with information about clinic timings, doctors, procedures, 
  pricing, aftercare, and appointments.
- You are NOT a doctor. Do not diagnose or create personalized treatment plans.

OPTION RULE:
If a patient asks for "options" or "who should I see" — always
present ALL relevant doctors even if they don't perfectly match
every requirement. Mention what each offers and what the tradeoff is.

LANGUAGE RULES:
- Detect the language of the user's message and reply in the SAME language.
- If the user writes in Urdu or Hinglish → reply in Urdu or Hinglish.
- If the user writes in English → reply in English.

BEHAVIOR RULES:
1. If the question is about HairRevive Clinic-specific details 
   (doctors, timings, pricing, packages, policies) → 
   answer ONLY from retrieved context.
2. If the question is about GENERAL hair transplant knowledge 
   (FUE, FUT, recovery time, pain level, permanence, aftercare basics) → 
   you may answer using general professional knowledge 
   in a safe and neutral way.
3. Do NOT provide diagnosis, medical evaluation, or personalized treatment advice.
4. Do NOT guarantee results or outcomes.
5. NEVER mention words like "context", "document", or "retrieved data".
6. Speak naturally like a real receptionist — warm, clear, empathetic, and helpful.
7. Keep answers concise unless the user asks for more detail.
8. If clinic-specific information is missing → say exactly:
   "I don't have that information right now. Please contact us on 
   WhatsApp +92-300-4247349 and our team will assist you."
9. If the user wants to book → say:
   "You can book via WhatsApp +92-300-4247349, call +92-42-3578-9900, 
   or visit hairreviveclinic.com/book"
10. If cost is asked → check retrieved packages first. If none found,
    explain that pricing depends on graft count and consultation.
11. Always be empathetic — hair loss is an emotional topic.
12. End with a helpful follow-up offer when appropriate.
"""
)

@traceable(run_type="retriever", name="Structured RAG Retrieve")
def retrieve_structured(query: str):
    return structured_rag.retrieve(query, k=4)

@traceable(run_type="retriever", name="Unstructured RAG Retrieve")
def retrieve_unstructured(query: str):
    return unstructured_rag.retrieve(query, k=4)


@traceable(                                           
    run_type="chain",
    name="HairRevive WhatsApp Agent",
    project_name="WhatsApp-Agent"
)
async def generate_reply(user_message: str) -> str:
    """
    Hybrid RAG pipeline:
    1. Query both structured and unstructured RAG simultaneously
    2. Merge and deduplicate context
    3. Send combined context to LLM once
    """

    # ── Step 1: Retrieve from both RAGs ───────────────────────────────────────
    structured_docs   = retrieve_structured(user_message)  
#    unstructured_docs = retrieve_unstructured(user_message)

    # ── Step 2: Build merged context with source labels ───────────────────────
    structured_context   = "\n\n".join([doc.page_content for doc in structured_docs])
 #   unstructured_context = "\n\n".join([doc.page_content for doc in unstructured_docs])

    # ── Step 3: Guard — if both empty, return fallback ────────────────────────
    if not structured_docs:
        return (
            "I don't have that information right now. "
            "Please contact us on WhatsApp +92-300-4247349 and our team will assist you."
        )

    pk_tz = pytz.timezone("Asia/Karachi")
    now = datetime.now(pk_tz)

    current_datetime = now.strftime("%A, %d %B %Y | %I:%M %p")
    current_day = now.strftime("%A")
    current_time = now.strftime("%I:%M %p")

    # ── Step 4: Inject merged context into prompt ─────────────────────────────
    prompt = f"""
    --- Current Date & Time ---
    Today is: {current_datetime}
    Day: {current_day}
    Current Time: {current_time}
    Timezone: Asia/Karachi (Pakistan)

    --- Clinic Reference Information ---
    {structured_context}

    --- Patient Question ---
    {user_message}

    If the patient asks about availability "now", 
    use the current time above to determine the answer.

    Answer based ONLY on the information above.
    """

     # ── Step 3: Run agent and capture usage ───────────────────────────────────
    result = await agent.run(prompt)

    # ── Step 4: Extract token usage from pydantic_ai result ───────────────────
    # pydantic_ai stores usage in result.usage()
    usage = result.usage()

    input_tokens  = usage.request_tokens  or 0
    output_tokens = usage.response_tokens or 0
    total_tokens  = usage.total_tokens    or (input_tokens + output_tokens)

    # ── Step 5: Log token usage to LangSmith as metadata ─────────────────────
    # LangSmith reads these specific keys to display token usage in dashboard
    try:
        from langsmith import get_current_run_tree
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.end(
                outputs={"output": result.output},
                metadata={
                    "usage": {
                        "input_tokens" : input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens" : total_tokens,
                    },
                    "model"  : "llama-3.3-70b-versatile",
                    "provider": "groq",
                }
            )
    except Exception:
        pass

    # Always print to console as backup
    print(f"[Tokens] Input: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}")

    return result.output