import asyncio
import json
from datetime import datetime
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from app.database import get_db
from app.features.bot.repository import (
    create_inactive_chat,
    delete_chat_by_chat_id,
    delete_users_all_chats,
    upload_edit_ques_from_chat_history,
    get_paginated_chat_messages,
    get_paginated_chats,
    mark_chat_as_active,
)
from app.features.bot.utils.response import ResponseCreator
from app.oauth import is_user_authorized
from app.models.employee import Employee
from app.utils.pagination import CursorPagination
from sentry_sdk import capture_exception

from app.config import env_variables
from app.features.bot.schemas import (
    DeleteSchema,
    MessageData,
    MessageUploadData,
    GenerateQuesFromChatHistory,
)
from app.features.bot.websocket_response import WebSocketResponse
from app.features.bot.message import BotMessage
from sqlalchemy.orm import Session
from app.request import Request
from app.utils.save_chat_messages import save_chat_message


env_data = env_variables()


class ConnectionManager:
    def __init__(self):
        self.client_and_connections = {}

    async def connect(self, websocket: WebSocket, chat_id: str):
        await websocket.accept()
        if chat_id not in self.client_and_connections:
            self.client_and_connections[chat_id] = []
        self.client_and_connections[chat_id].append(websocket)

    try:

        def disconnect(self, websocket: WebSocket, chat_id):
            if chat_id in self.client_and_connections:
                self.client_and_connections[chat_id].remove(websocket)

    except Exception as e:
        capture_exception(e)

    async def send_personal_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print("Error in send_personal_message", e)
            capture_exception(e)

    async def broadcast(self, message: dict):
        try:
            for connection in self.client_and_connections:
                await connection.send_text(json.dumps(message))
        except Exception as e:
            print("Error in broadcast", e)
            capture_exception(e)


manager = ConnectionManager()
running_tasks = {}

router = APIRouter(tags=["Chat"], prefix="/chat")


# Endpoint to upload or edit question answers from chat history.
@router.post("/edit-chat-history")
async def edit_question_from_chat_history(
    request: Request,
    data: GenerateQuesFromChatHistory,
    db: Session = Depends(get_db),
    current_user: dict = Depends(is_user_authorized),
):
    return await upload_edit_ques_from_chat_history(
        request, data, db, current_user["id"]
    )


@router.get("/{chat_id}/messages")
async def get_chat_messages(
    request: Request,
    chat_id: int,
    db: Session = Depends(get_db),
    cursor: str | None = CursorPagination.get_schema_fields()["cursor"],
    _: dict = Depends(is_user_authorized),
):
    return get_paginated_chat_messages(request, chat_id, db, cursor)


@router.get("/all")
async def get_chats(
    request: Request,
    db: Session = Depends(get_db),
    current_user: dict = Depends(is_user_authorized),
):

    return get_paginated_chats(request, db, current_user["id"])


@router.websocket("/ws/connection/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    try:
        await connect_websocket(websocket, chat_id)
        now = datetime.utcnow()
        current_time = now.strftime("%H:%M")
        isError = False

        while True:
            data = await websocket.receive_text()

            await process_received_data(data, chat_id, current_time, isError, websocket)

    except WebSocketDisconnect as e:
        print(e, "error in WebSocketDisconnect")
        await handle_websocket_disconnect(websocket, chat_id, current_time)
    except Exception as e:
        print(e, ">>>>>>>>>>>>>.error")
        capture_exception(e)


async def connect_websocket(websocket: WebSocket, chat_id: str):
    return await manager.connect(websocket, chat_id)


async def process_received_data(
    data, chat_id, current_time, isError, websocket: WebSocket
):
    received_data = json.loads(data)
    mt = received_data.get("mt", "")
    message = received_data.get("message", "")
    # user_input = received_data.get("user_input", "")
    message_context = None
    try:
        int(chat_id)
    except ValueError as e:
        print("ValueError", e)
        await handle_invalid_chat_id(websocket, chat_id)
        isError = True

    if (mt == "message_upload") and not isError:
        asyncio.create_task(handle_query_type(websocket, message_context))
        task = asyncio.create_task(
            handle_message_upload(received_data, chat_id, current_time, websocket)
        )
        running_tasks[websocket] = task

    elif mt == "stop":
        if websocket in running_tasks:
            task = running_tasks[websocket]
            task.cancel()
        current_time_ms = int(datetime.now().timestamp() * 1000)
        user_message_token = f"{current_time_ms}_{chat_id}"

        asyncio.create_task(
            save_chat_message(
                chat_id,
                message,
                True,
                received_data["timezone"],
                mt,
                user_message_token,
                message_context=message_context,
            )
        )
        await manager.send_personal_message(
            websocket,
            {
                "mt": "stopped_generation",
                "isBot": True,
                "message": message,
                "chatId": chat_id,
                "token": user_message_token,
            },
        )
        del running_tasks[websocket]


async def handle_invalid_chat_id(websocket, chat_id):
    error_message = f"Invalid client ID: {chat_id}"
    await manager.send_personal_message(
        websocket,
        {
            "mt": "bot_access_denied",
            "message": error_message,
            "chatId": chat_id,
        },
    )


async def handle_message_upload(
    received_data, chat_id, current_time, websocket: WebSocket
):
    try:

        db = next(get_db())
        upload_data = MessageUploadData(**received_data)
        current_time_ms = int(datetime.now().timestamp() * 1000)
        user_message_token = f"{current_time_ms}_{chat_id}"
        message = MessageData(
            time=current_time,
            chatId=chat_id,
            userId=upload_data.userId,
            message=upload_data.message,
            isBot=upload_data.isBot,
            token=user_message_token,
            languageCode=upload_data.selectedLanguage,
            mt="message_upload_confirm",
        )
        await manager.send_personal_message(websocket, message.dict())

        try:
            # Directly fetch the organization_id for the given user_id from Employee table
            org_id = db.query(Employee.organization_id).filter_by(user_id=upload_data.userId).scalar()
        except Exception as e:
            print("Error while fetching organization ID:", e)


        socket_response = WebSocketResponse(
            connection_manager=manager,
            user_id=upload_data.userId,
            chat_id=chat_id,
            websocket=websocket,
            user_message_token=user_message_token,
        )

        await mark_chat_as_active(
            db, chat_id, received_data["message"], received_data["userId"], org_id
        )

        bot_message = BotMessage(
            socket_response=socket_response,
            chat_id=chat_id,
            user_id=upload_data.userId,
            org_id=org_id,
            time_zone=received_data["timezone"],
            user_message_token=user_message_token,
            language_code=upload_data.selectedLanguage,
        )
        no_answer_found = await bot_message.send_bot_message(received_data["message"])

        await asyncio.sleep(1)

        current_time_ms = int(datetime.now().timestamp() * 1000)
        token = f"{current_time_ms}_{chat_id}"

        if no_answer_found:
            default_answer = (
                "I'm having trouble right now. You can try again."
            )
            message = MessageData(
                time=current_time,
                chatId=chat_id,
                message=default_answer,
                userId=upload_data.userId,
                isBot=True,
                token=token,
                mt="message_upload_confirm",
            )
            message_context = await BotMessage.check_context_of_user_input(
                None, message
            )

            asyncio.create_task(
                save_chat_message(
                    chat_id,
                    default_answer,
                    True,
                    received_data["timezone"],
                    "failed",
                    token,
                    user_message_token=user_message_token,
                    message_context=message_context,
                )
            )
            await manager.send_personal_message(websocket, message.dict())
    except asyncio.CancelledError:
        print(f"Task for chat {chat_id} was cancelled.")
        # Handle task cancellation (e.g., send cancellation message to client)
        await manager.send_personal_message(
            websocket,
            {
                "mt": "generation_stopped",
                "message": "Bot response generation was stopped.",
                "chatId": chat_id,
            },
        )


async def handle_query_type(websocket: WebSocket, message_context):
    try:
        # await manager.send_personal_message(websocket, message_context)
        await manager.send_personal_message(
            websocket,
            {
                "mt": "message_context_type",
                "message_context": message_context,
            },
        )
    except Exception as e:
        capture_exception(e)


async def handle_websocket_disconnect(websocket, chat_id, current_time):
    manager.disconnect(websocket, chat_id)
    message = {"time": current_time, "chatId": chat_id, "message": "Offline"}
    try:
        await manager.send_personal_message(websocket, message)
    except Exception as e:
        capture_exception(e)


@router.post("/delete-user-all-chat")
async def delete_all_chat_of_user(
    request: Request,
    db: Session = Depends(get_db),
    current_user: dict = Depends(is_user_authorized),
):
    return await delete_users_all_chats(db, current_user["id"])


@router.post("/delete-chat")
async def delete_chat_by_id(
    request: DeleteSchema,
    db: Session = Depends(get_db),
    current_user: dict = Depends(is_user_authorized),
):
    return await delete_chat_by_chat_id(db, request.chat_id, current_user["id"])


@router.post("/create-new-chat")
async def create_new_chat(
    request: Request,
    db: Session = Depends(get_db),
    current_user: dict = Depends(is_user_authorized),
):
    return await create_inactive_chat(db, current_user["id"])
