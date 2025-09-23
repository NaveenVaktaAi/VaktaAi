from typing import Optional
from sentry_sdk import capture_exception

from app.database import get_db
from app.models.chat_message import ChatMessage
from sqlalchemy import update


async def save_chat_message(chat_id, text_message, is_bot, time_zone, msg_type=None, token=None, *, user_message_token=None,message_context="public"):
    try:
        db_connection = next(get_db())
        # update status of user message to failed from user_message_token
        if msg_type == "failed" and user_message_token:
            updated_values = {
                 "type": "failed"
            }
            query = (
                update(ChatMessage)
                .where(ChatMessage.token == user_message_token)
                .values(updated_values)
            ) 
            db_connection.execute(query)
            db_connection.commit()

        new_message = ChatMessage(
            chat_id=chat_id,
            message=text_message,
            is_bot=is_bot,
            timezone=time_zone,
            message_context=message_context,
            type=msg_type,
            token=token,
        )
        
        db_connection.add(new_message)

        db_connection.commit()

    except Exception as e:
        print(e, "Exception")
        capture_exception(e)
