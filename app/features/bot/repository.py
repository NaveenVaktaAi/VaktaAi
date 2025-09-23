import datetime
import threading
from sentry_sdk import capture_exception
from sqlalchemy import delete, func, or_, text, update
from app.database import get_db
from app.features.admin.repository import (
    common_delete_fun,
    create_chunks,
    generate_keywords,
    insert_or_update_to_keywords_table,
)
from app.models.agent_ai_documents import AgentAIDocument
from app.models.chat import Chat
from app.models.chat_message import ChatMessage
from app.models.chunks import Chunk
from app.models.employee import Employee
from app.models.keywords import Keywords
from app.models.legal_documnet_chunks import LegalDocumentChunks
from app.models.question_answer import QuestionAnswer
from app.models.organization import Organization
from app.models.user import User
from app.request import Request
from sqlalchemy.orm import Session

from app.utils.milvus.operations.crud import (
    delete_keywords_from_milvus,
    insert_data_into_milvus,
)
from app.utils.pagination import CursorPagination
from app.features.bot.schemas import (
    GenerateQuesFromChatHistory,
)

from keybert import KeyBERT


def get_paginated_chat_messages(
    request: Request,
    chat_id: int,
    db: Session,
    cursor: str | None = CursorPagination.get_schema_fields()["cursor"],
):
    # query = db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id)
    query = (
        db.query(
            ChatMessage,
            func.concat(Employee.first_name, " ", Employee.last_name).label("employee_name"),
            # User.profile_img # TODO: Not in base model and sign up page
        )
        .join(Chat, ChatMessage.chat_id == Chat.id)
        .join(Employee, Chat.user_id == Employee.user_id)
        # .join(User, User.id == Employee.user_id)
        .filter(ChatMessage.chat_id == chat_id)
    )
     # Check if there are any messages for the given chat_id
    if not query.first():
        return {
            "data": {
                "next": None,
                "previous": None,
                "results": []
            },
            "success": True
        }

    paginator = CursorPagination()
    paginator.ordering = "-chat_messages.id"
    paginator.page_size = 10
    page = paginator.paginate_queryset(query, request)
    return {"data": paginator.get_paginated_response_admin_chats(page), "success": True}


def get_paginated_chats(
    request: Request,
    db: Session,
    user_id: str,
):
    subquery = None
    query = None
    org_id = None
    status = request.query_params.get("status")
    # Fetch user details
    user_details = db.query(User.role_type).filter(User.id == user_id).first()
    if not user_details:
        return {"data": [], "success": False, "message": "User not found"}
    user_type = user_details.role_type

    # Fetch organization ID for organization users
    if user_type == "organization":
        if status == "":
            status = "like"
        org_details = db.query(Organization.id).filter(Organization.user_id == user_id).first()
        if not org_details:
            return {"data": [], "success": False, "message": "Organization not found"}
        org_id = org_details.id
        query = db.query(Chat).filter(Chat.status == "active", Chat.organization_id == org_id)
    else:
        query = db.query(Chat).filter(Chat.status == "active", Chat.user_id == user_id)

    # Filter by status
    # status = request.query_params.get("status")
    if status == "failed":
        subquery = (
            db.query(ChatMessage.chat_id)
            .filter(ChatMessage.type == "failed")
            .distinct()
        ).subquery()
    elif status in ["like", "dislike"]:
        reaction = True if status == "like" else False
        subquery = (
            db.query(ChatMessage.chat_id)
            .filter(ChatMessage.reaction == reaction)
            .distinct()
        ).subquery()

    # Join Chat with the subquery if applicable
    if subquery is not None:
        query = query.join(subquery, Chat.id == subquery.c.chat_id)

    # Pagination
    paginator = CursorPagination()
    paginator.ordering = "-id"
    paginator.page_size = 15
    try:
        page = paginator.paginate_queryset(query, request)
        return {"data": paginator.get_paginated_response(page), "success": True}
    except Exception as e:
        return {"data": [], "success": False, "message": str(e)}



async def save_previous_answer(result):
    if not result.relevant_answer:
        result.relevant_answer = []

    temp_relevant_answers = result.relevant_answer.copy()
    print(temp_relevant_answers, "temp_relevant_answers====================") 
    if result.answer:
        temp_relevant_answers.append(result.answer)

    if len(temp_relevant_answers) > 3:
        # keep latest 3 answers only
        temp_relevant_answers = temp_relevant_answers[-3:]

    result.relevant_answer = temp_relevant_answers


async def upload_edit_ques_from_chat_history(
    request: Request, data: GenerateQuesFromChatHistory, db: Session, user_id: str
):
    try:
        """
        Endpoint to add a new question or edit an existing question based on chat history.
        :param request: Request object containing user message, bot message, and optional message id
        :return: Response indicating success or failure of the operation
        """

         
        org_id = db.query(Organization.id).filter(Organization.user_id == user_id).scalar() if user_id else None

         
        if not data.messageId:
            question_id = -1
            question_value = None
            existing_question = (
                db.query(QuestionAnswer)
                .filter(
                    QuestionAnswer.training_document_id == None,
                    func.lower(QuestionAnswer.question) == func.lower(data.userMessage),
                    QuestionAnswer.organization_id == org_id,
                )
                .first()
            )

            if existing_question:
                await save_previous_answer(existing_question)

                existing_question.answer = data.botMessage
                existing_question.question = data.currentUserMessage.lower()
                existing_question.question_edit_count += 1
                existing_question.updated_ts = datetime.datetime.now()
                question_value = existing_question
                question_id = existing_question.id
            else:
                new_question = QuestionAnswer(
                    question=data.currentUserMessage.lower(),
                    answer=data.botMessage,
                    question_edit_count=1,
                    position=0,
                    relevant_answer=[data.prevBotMessage],
                    organization_id=org_id,
                    created_ts=datetime.datetime.now(),
                    updated_ts=datetime.datetime.now(),
                )
                db.add(new_question)
                db.flush()  
                question_value = new_question
                question_id = new_question.id
 
            db.query(ChatMessage).filter(ChatMessage.id == data.chat_messages_id).update(
                {ChatMessage.is_edited: True, ChatMessage.message: data.botMessage}
            )

            db.commit()   

            
            milvus_resp = await insert_data_into_milvus(
                question_id, question_value.vector_question_ids, data.currentUserMessage
            )

            db.query(QuestionAnswer).filter(QuestionAnswer.id == question_id).update(
                {QuestionAnswer.vector_question_ids: milvus_resp.primary_keys[0]}
            )
            db.commit()

            threading.Thread(
                target=train_bot_from_edited_answer,
                args=(data, question_id, org_id),
            ).start()

            return {"message": "Question added successfully",
            "success": True,          }

        # If messageId exists, update the question
        existing_question = db.query(QuestionAnswer).filter(QuestionAnswer.id == data.messageId).first()

        if not existing_question:
            return {"message": "Question not found", "success": False}

        await save_previous_answer(existing_question)

        db.query(QuestionAnswer).filter(QuestionAnswer.id == data.messageId).update(
            {
                "answer": data.botMessage,
                "question_edit_count": existing_question.question_edit_count + 1,
                "updated_ts": datetime.datetime.now(),
            }
        )

        db.commit() 
        # Train bot asynchronously
        threading.Thread(
            target=train_bot_from_edited_answer,
            args=(data, existing_question.id, org_id),
        ).start()

        return {"message": "Question updated successfully", "success": True}

    except Exception as e:
        capture_exception(e)
        return {"message": "Something went wrong", "success": False}

async def mark_chat_as_active(db: Session, chat_id, user_input, user_id, org_id):
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if chat.status == "active":
            return chat
        chat.status = "active"
        chat.title = user_input
        chat.user_id = user_id
        chat.organization_id = org_id
        db.commit()

        return chat
    except Exception as e:
        print(e, "error associate_chat_with_user")
        capture_exception(e)


def train_bot_from_edited_answer(request, question_id, org_id):
    try:
        # Initialize KeyBERT model for keyword extraction
        keyword_model = KeyBERT()
        db = next(get_db()) 
        question_id  = int(question_id)
        training_document_id = db.query(QuestionAnswer.training_document_id).filter(QuestionAnswer.id == question_id).scalar() 
        doc_type = db.query(AgentAIDocument.type).filter(AgentAIDocument.id == training_document_id).scalar() 
        # create new chunks and keywords from new answer
        if not request.messageId:
            chunks_data = create_chunks(request.botMessage, 200)
            for chunk_info in chunks_data:
                keywords_string, keyword_list = generate_keywords(
                    (request.userMessage + chunk_info),
                    keyword_model,
                )

                # if doc_type.lower() == "legal":
                #     chunk = LegalDocumentChunks(
                #         detail=request.botMessage,
                #         keywords=keywords_string,
                #         meta_summary="",
                #         chunk=chunk_info,
                #         organization_id=org_id,
                #         question_id=question_id,
                #         training_document_id=None,
                #         created_ts=datetime.now(),
                #         updated_ts=datetime.now(),
                #     )
                #     db.add(chunk)
                    
                # else:
                chunk = Chunk(
                    detail=request.botMessage,
                    keywords=keywords_string,
                    meta_summary="",
                    chunk=chunk_info,
                    organization_id=org_id,
                    question_id=question_id,
                    training_document_id=None,
                    created_ts=datetime.now(),
                    updated_ts=datetime.now(),
                )
                db.add(chunk)
                # # Insert or update keywords associated with the chunk
                insert_or_update_to_keywords_table(db, keyword_list, chunk.id, doc_type, chunk.organization_id)
        else:
            # If messageId exists, delete old chunks and create new based on edited answer
            train_bot_from_update_answer_from_question_listing(db, request , org_id , doc_type)

    except Exception as e:
        print("error in train_bot_from_edited_answer", e)
        capture_exception(e)
        return {"message": "Something went wrong", "success": False}


def train_bot_from_update_answer_from_question_listing(db: Session, request , org_id , doc_type):
    keyword_model = KeyBERT()
    try:
        get_chunk_id_query = (
            db.query(Chunk.id).filter(Chunk.question_id == request.messageId).all()
        )
        chunk_ids = [chunk[0] for chunk in get_chunk_id_query]
        # Update chunk IDs in the Keywords table
        if chunk_ids:

            common_delete_fun(db, chunk_ids)
            # Delete chunks associated with the provided question ID
            delete_chunk_query = delete(Chunk).where(
                Chunk.question_id == request.messageId
            )
            db.execute(delete_chunk_query)
            db.commit()

            # create new chunks
            chunks_data = create_chunks(request.botMessage, 200)
            for chunk_info in chunks_data:
                keywords_string, keyword_list = generate_keywords(
                    (request.userMessage + chunk_info),
                    keyword_model,
                )
                # if doc_type.lower() == "legal":
                #     chunk = LegalDocumentChunks(
                #         detail=request.botMessage,
                #         keywords=keywords_string,
                #         meta_summary="",
                #         chunk=chunk_info,
                #         question_id=request.messageId,
                #         created_ts=datetime.datetime.now(),
                #         updated_ts=datetime.datetime.now(),
                #     )
                #     db.add(chunk) 
                #     db.flush
                # else:
                chunk = Chunk(
                    detail=request.botMessage,
                    keywords=keywords_string,
                    meta_summary="",
                    chunk=chunk_info,
                    question_id=request.messageId,
                    created_ts=datetime.datetime.now(),
                    updated_ts=datetime.datetime.now(),
                )
                db.add(chunk)
                db.flush()

                # Insert or update keywords associated with the chunk
                insert_or_update_to_keywords_table(db, keyword_list, chunk.id , org_id)

    except Exception as e:
        print("error in train_bot_from_update_answer_from_question_listing", e)
        capture_exception(e)
        return {"message": "Something went wrong", "success": False}

async def delete_users_all_chats(db: Session, user_id: int):
    try:
        # Step 1: Get all chat IDs for the user
        chat_ids_query = db.query(Chat.id).filter(Chat.user_id == user_id)
 
        # Step 2: Delete all messages for the user's chats
        delete_messages_query = db.query(ChatMessage).filter(
            ChatMessage.chat_id.in_(chat_ids_query)
        )
        if delete_messages_query.count() > 0:
            delete_messages_query.delete(synchronize_session=False)
            db.commit()
            print(f"Deleted all chat messages for user_id: {user_id}")
 
        # Step 3: Delete all chats for the user
        delete_chats_query = db.query(Chat).filter(Chat.user_id == user_id)
        if delete_chats_query.count() > 0:
            delete_chats_query.delete(synchronize_session=False)
            db.commit()
            print(f"Deleted all chats for user_id: {user_id}")
 
            return {
                "message": "All chats and associated messages deleted successfully.",
                "success": True,
            }
 
        return {
            "message": "No chats found for the user.",
            "success": False,
        }
 
    except Exception as e:
        db.rollback()
        print("Error in delete_users_all_chats:", e)
        capture_exception(e)
        return {"message": "Something went wrong", "success": False}


async def create_inactive_chat(db: Session, user_id):
    try:
        chat = Chat(
            user_id=user_id,
            status="inactive",
        )
        db.add(chat)
        db.commit()
        print(chat, "chat============================")
        return {
            "message": "chat created successfully",
            "success": True,
            "data": chat.to_dict(),
        }
    except Exception as e:
        print("error in create_inactive_chat", e)
        capture_exception(e)
        return {"message": "Something went wrong", "success": False}


async def delete_chat_by_chat_id(db: Session, chat_id: int, user_id: int):
    try:
        print(chat_id, "delete_messages_query 1", user_id)
 
        # Delete all messages of the chat first
        delete_messages_query = db.query(ChatMessage).filter(
            ChatMessage.chat_id == chat_id
        )
        print(delete_messages_query, "delete_messages_query")
        if delete_messages_query.count() > 0:
            delete_messages_query.delete(synchronize_session=False)
            db.commit()
 
        print("Messages deleted successfully.")
 
        # Delete the chat
        delete_query = db.query(Chat).filter(
            Chat.id == chat_id, Chat.user_id == user_id
        )
        if delete_query.count() > 0:
            delete_query.delete(synchronize_session=False)
            db.commit()
            print("Chat deleted successfully.")
            return {
                "message": "Chat and associated messages deleted successfully",
                "success": True,
            }
 
        return {
            "message": "Chat not found",
            "success": False,
        }
    except Exception as e:
        db.rollback()
        print("Error in delete_all_chats", e)
        return {"message": "Something went wrong", "success": False}
