import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Optional
from uuid import uuid4

from app.utils.save_chat_messages import save_chat_message
from app.features.bot.utils.response import ResponseCreator as RC
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketResponse:
    """
    A class that handles writing messages for the bot.
    """

    def __init__(
        self,
        *,
        chat_id: str,
        user_id: str = None,
        websocket: WebSocket,
        connection_manager=None,
        user_message_token=None,
    ):
        self.chat_id = chat_id
        self.user_id = user_id
        self.connection_manager = connection_manager
        self.websocket = websocket
        self.user_message_token = user_message_token

    async def async_word_generator(self, text: str) -> AsyncGenerator[str, None]:
        """Async generator yielding words from the given text."""
        words = re.split(r"(\s+)", text)
        for word in words:
            yield word

    # Websocket Sending
    async def create_bot_response(
        self,
        text: AsyncGenerator[str, None] | str,
        time_zone: Optional[str] = None,
        from_lang_chain: Optional[bool] = False,
        user_question: Optional[str] = None,
        *,
        msg_type: Optional[str] = "normal_msg",
        doc_type: Optional[str] = "public",
        message_context: Optional[str] = None,
    ) -> Optional[str]:
        """Creates bot message and sends it. Returns full text."""
        now = datetime.utcnow()
        current_time = now.strftime("%H:%M")
        final_text = ""
        uuid = str(uuid4())
        print(doc_type,"doc_typedoc_typedoc_typedoc_typedoc_typedoc_type",message_context, msg_type)
        if not text:
            logger.error(f"Could not generate text for {self.chat_id}")
            if msg_type != "followup_msg":
                await self.connection_manager.send_personal_message(
                    self.websocket,
                    {
                        "mt": (
                            "followup_message"
                            if (msg_type == "followup_msg")
                            else "message_upload_confirm"
                        ),
                        "chatId": self.chat_id,
                        "uuid": uuid,
                        "message": "I'm having difficulty comprehending your message."
                        " Could you please phrase it differently?",
                        "time": current_time,
                        "message_context": doc_type if doc_type else "public",
                        "isBot": True,
                    },
                )
                return None

        if self.connection_manager is None:
            logger.error("Websocket is not initialized.")
            return None
        if msg_type != "followup_msg":
            await self.connection_manager.send_personal_message(
                self.websocket,
                {
                    "mt": (
                        "followup_message"
                        if (msg_type == "followup_msg")
                        else "chat_message_bot_partial"
                    ),
                    "chatId": self.chat_id,
                    "start": uuid,
                },
            )

        if isinstance(text, str):
            text = self.async_word_generator(text)

        if from_lang_chain:
            for t in text:
                if t and t.content:
                    final_text += t.content
                    if msg_type != "followup_msg":
                        await self.connection_manager.send_personal_message(
                            self.websocket,
                            {
                                "mt": (
                                    "followup_message"
                                    if (msg_type == "followup_msg")
                                    else "chat_message_bot_partial"
                                ),
                                "message_context": doc_type if doc_type else "public",
                                "chatId": self.chat_id,
                                "uuid": uuid,
                                "partial": t.content,
                            },
                        )
        else:
            async for t in text:
                if t:
                    final_text += t
                    if msg_type != "followup_msg":
                        await self.connection_manager.send_personal_message(
                            self.websocket,
                            {
                                "mt": (
                                    "followup_message"
                                    if (msg_type == "followup_msg")
                                    else "chat_message_bot_partial"
                                ),
                                "chatId": self.chat_id,
                                "message_context": doc_type if doc_type else "public",
                                "uuid": uuid,
                                "partial": t,
                            },
                        )
        if msg_type != "followup_msg":
            await self.connection_manager.send_personal_message(
                self.websocket,
                {
                    "mt": (
                        "followup_message"
                        if (msg_type == "followup_msg")
                        else "chat_message_bot_partial"
                    ),
                    "chatId": self.chat_id,
                    "stop": uuid,
                },
            )

        current_time_ms = int(datetime.now().timestamp() * 1000)
        token = f"{current_time_ms}_{self.chat_id}"

        await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": (
                    "followup_message"
                    if (msg_type == "followup_msg")
                    else "message_upload_confirm"
                ),
                "chatId": self.chat_id,
                "uuid": uuid,
                "message": {"result":final_text},
                "message_context": doc_type if doc_type else "public",
                "time": current_time,
                "isBot": True,
                "token": token,
            },
        )

        msg_status = None
        if (
            final_text
            == "I'm having trouble right now. You can try again."
        ):
            msg_status = "failed"

        if msg_type != "followup_msg":
            asyncio.create_task(
                save_chat_message(
                    self.chat_id,
                    final_text,
                    True,
                    time_zone,
                    msg_status,
                    token,
                    user_message_token=self.user_message_token,
                    message_context=doc_type if doc_type else "public",
                )
            )

        if msg_type == "normal_msg" and msg_status != "failed":
            await RC().get_followup_question(
                self.chat_id,
                self.websocket,
                self.connection_manager,
                user_question,
                final_text,
            )

        return final_text
