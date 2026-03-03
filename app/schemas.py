from pydantic import BaseModel

class TwilioWebhook(BaseModel):
    From: str
    Body: str