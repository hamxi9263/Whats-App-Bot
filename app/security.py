from fastapi import Request, HTTPException
from twilio.request_validator import RequestValidator
from app.config import settings

validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)


async def validate_twilio_request(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Twilio-Signature")

    url = str(request.url)

    if not validator.validate(url, body.decode(), signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")