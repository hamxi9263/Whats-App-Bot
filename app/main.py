from fastapi import FastAPI, Request, Form
from app.config import settings 
from fastapi.responses import PlainTextResponse
from app.agent import generate_reply
from app.twilio_client import TwilioClient
from app.logger import logger
from app.security import validate_twilio_request
from twilio.twiml.messaging_response import MessagingResponse
import traceback
import logfire

# Stage Branch

app = FastAPI(title="Enterprise WhatsApp AI Bot (Groq + Twilio)")

twilio = TwilioClient()

logfire.configure(
    service_name="whatsappagentrag"
)

logfire.instrument_fastapi(app)
logfire.instrument_httpx()

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    try:
        # ✅ Read body only once
        body_bytes = await request.body()
        body_str = body_bytes.decode()

        # ✅ Parse form safely
        form = await request.form()

        incoming_msg = form.get("Body")
        sender = form.get("From")

        print("Message:", incoming_msg)

        # ✅ Ignore non-message events
        if not incoming_msg:
            print("No message body. Skipping LLM.")
            return PlainTextResponse(
                content=str(MessagingResponse()),
                media_type="application/xml"
            )

        # Generate reply from Groq
        reply = await generate_reply(incoming_msg)

        print(reply)

        # Twilio response
        twilio_response = MessagingResponse()
        twilio_response.message(reply)

        return PlainTextResponse(
            content=str(twilio_response),
            media_type="application/xml"
        )

    except Exception as e:
        traceback.print_exc()
        print("Webhook error:", str(e))
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "ok"}