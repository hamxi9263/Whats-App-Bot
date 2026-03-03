import httpx
from app.config import settings
from app.logger import logger


class TwilioClient:

    def __init__(self):
        self.account_sid = settings.TWILIO_ACCOUNT_SID
        self.auth_token = settings.TWILIO_AUTH_TOKEN
        self.base_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"

    async def send_message(self, to: str, message: str):
        data = {
            "From": settings.TWILIO_WHATSAPP_NUMBER,
            "To": to,
            "Body": message
        }

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    self.base_url,
                    data=data,
                    auth=(self.account_sid, self.auth_token)
                )
                response.raise_for_status()
                logger.info("message_sent", to=to)
            except Exception as e:
                logger.error("twilio_send_failed", error=str(e))
                raise