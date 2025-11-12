from typing import Tuple, Dict, Any
from pymongo.database import Database
from bson import ObjectId
from datetime import datetime


def get_collections(db: Database):
    documents = db["docSathi_ai_documents"]
    chunks = db["chunks"]
    chats = db["chats"]
    chat_messages = db["chat_messages"]
    student_quizs = db["student_quizs"]
    question_answers = db["question_answers"]
    users = db["users"]
    students = db["students"]

    # Ensure common indexes
    documents.create_index("user_id")
    documents.create_index("status")
    chunks.create_index("training_document_id")
    chunks.create_index("question_id")
    
    # Chat collection indexes
    chats.create_index("user_id")
    chats.create_index("document_id")
    chats.create_index("status")
    chats.create_index("created_at")
    
    # Chat messages collection indexes
    chat_messages.create_index("chat_id")
    chat_messages.create_index("is_bot")
    chat_messages.create_index("created_ts")
    
    # Student quizs collection indexes
    student_quizs.create_index("related_doc_id")
    student_quizs.create_index("created_by")
    student_quizs.create_index("level")
    student_quizs.create_index("created_at")
    
    # Question answers collection indexes
    question_answers.create_index("quiz_id")
    question_answers.create_index("question_type")
    question_answers.create_index("created_at")
    
    # Users collection indexes
    users.create_index("phone_number", unique=True)
    users.create_index("email", unique=True, sparse=True)
    users.create_index("role")
    users.create_index("account_status")
    users.create_index("is_active")
    users.create_index("is_phone_verified")
    users.create_index("is_email_verified")
    users.create_index("created_at")
    users.create_index("last_login_at")
    users.create_index("referral_code", unique=True, sparse=True)
    users.create_index("referred_by")
    
    # Students collection indexes
    students.create_index("user_id", unique=True)
    students.create_index("current_class")
    students.create_index("board")
    students.create_index("exam_target")
    students.create_index("preferred_language")
    students.create_index("state")
    students.create_index("city")
    students.create_index("created_at")

    return documents, chunks, chats, chat_messages, student_quizs, question_answers, users, students


def create_document(db: Database, doc: Dict[str, Any]) -> str:
    documents, _, _, _, _, _, _, _ = get_collections(db)
    result = documents.insert_one(doc)
    return str(result.inserted_id)


def update_document_status(db: Database, document_id: str, fields: Dict[str, Any]) -> None:
    documents, _, _, _, _, _, _, _ = get_collections(db)
    documents.update_one({"_id": ObjectId(document_id)}, {"$set": fields})


def insert_chunk(db: Database, chunk_doc: Dict[str, Any]) -> str:
    _, chunks, _, _, _, _, _, _ = get_collections(db)
    result = chunks.insert_one(chunk_doc)
    return str(result.inserted_id)


# Chat CRUD operations
def create_chat(db: Database, chat_doc: Dict[str, Any]) -> str:
    """Create a new chat"""
    _, _, chats, _, _, _, _, _ = get_collections(db)
    result = chats.insert_one(chat_doc)
    return str(result.inserted_id)


def get_chat(db: Database, chat_id: str) -> Dict[str, Any]:
    """Get a chat by ID"""
    _, _, chats, _, _, _, _, _ = get_collections(db)
    return chats.find_one({"_id": ObjectId(chat_id)})


def get_user_chats(db: Database, user_id: str) -> list:
    """Get all chats for a user"""
    _, _, chats, _, _, _, _, _ = get_collections(db)
    # Handle both string (ObjectId) and integer user_id for backward compatibility
    try:
        # Try as string first (for ObjectId strings)
        if isinstance(user_id, str) and len(user_id) == 24 and all(c in '0123456789abcdefABCDEF' for c in user_id):
            # It's an ObjectId string, use as string
            query = {"user_id": user_id}
        else:
            # Try to convert to integer (for backward compatibility)
            query = {"user_id": int(user_id)}
    except (ValueError, TypeError):
        # If conversion fails, use as string
        query = {"user_id": user_id}
    
    return list(chats.find(query).sort("created_at", -1))


def update_chat(db: Database, chat_id: str, fields: Dict[str, Any]) -> None:
    """Update a chat"""
    _, _, chats, _, _, _, _, _ = get_collections(db)
    chats.update_one({"_id": ObjectId(chat_id)}, {"$set": fields})


def delete_chat(db: Database, chat_id: str) -> None:
    """Delete a chat and all its messages"""
    _, _, chats, chat_messages, _, _, _, _ = get_collections(db)
    
    # Delete all messages for this chat
    chat_messages.delete_many({"chat_id": ObjectId(chat_id)})
    
    # Delete the chat
    chats.delete_one({"_id": ObjectId(chat_id)})


# Chat Messages CRUD operations
def create_chat_message(db: Database, message_doc: Dict[str, Any]) -> str:
    """Create a new chat message"""
    _, _, _, chat_messages, _, _, _, _ = get_collections(db)
    result = chat_messages.insert_one(message_doc)
    return str(result.inserted_id)


def get_chat_messages(db: Database, chat_id: str, limit: int = 50, skip: int = 0) -> list:
    """Get messages for a chat with pagination"""
    _, _, _, chat_messages, _, _, _, _ = get_collections(db)
    return list(chat_messages.find({"chat_id": ObjectId(chat_id)})
                .sort("created_ts", 1)
                .skip(skip)
                .limit(limit))


def get_chat_message(db: Database, message_id: str) -> Dict[str, Any]:
    """Get a specific chat message"""
    _, _, _, chat_messages, _, _, _, _ = get_collections(db)
    return chat_messages.find_one({"_id": ObjectId(message_id)})


def update_chat_message(db: Database, message_id: str, fields: Dict[str, Any]) -> None:
    """Update a chat message"""
    _, _, _, chat_messages, _, _, _, _ = get_collections(db)
    chat_messages.update_one({"_id": ObjectId(message_id)}, {"$set": fields})


def delete_chat_message(db: Database, message_id: str) -> None:
    """Delete a chat message"""
    _, _, _, chat_messages, _, _, _, _ = get_collections(db)
    chat_messages.delete_one({"_id": ObjectId(message_id)})


def get_chat_message_count(db: Database, chat_id: str) -> int:
    """Get total message count for a chat"""
    _, _, _, chat_messages, _, _, _, _ = get_collections(db)
    return chat_messages.count_documents({"chat_id": ObjectId(chat_id)})


# Student Quiz CRUD operations
def create_student_quiz(db: Database, quiz_doc: Dict[str, Any]) -> str:
    """Create a new student quiz"""
    _, _, _, _, student_quizs, _, _, _ = get_collections(db)
    result = student_quizs.insert_one(quiz_doc)
    return str(result.inserted_id)


def get_student_quiz(db: Database, quiz_id: str) -> Dict[str, Any]:
    """Get a student quiz by ID"""
    _, _, _, _, student_quizs, _, _, _ = get_collections(db)
    return student_quizs.find_one({"_id": ObjectId(quiz_id)})


def get_student_quizs_by_doc(db: Database, doc_id: str) -> list:
    """Get all quizzes for a document"""
    _, _, _, _, student_quizs, _, _, _ = get_collections(db)
    return list(student_quizs.find({"related_doc_id": ObjectId(doc_id)}).sort("created_at", -1))


def get_student_quizs_by_user(db: Database, user_id: str) -> list:
    """Get all quizzes created by a user"""
    _, _, _, _, student_quizs, _, _, _ = get_collections(db)
    return list(student_quizs.find({"created_by": ObjectId(user_id)}).sort("created_at", -1))


def update_student_quiz(db: Database, quiz_id: str, fields: Dict[str, Any]) -> None:
    """Update a student quiz"""
    _, _, _, _, student_quizs, _, _, _ = get_collections(db)
    student_quizs.update_one({"_id": ObjectId(quiz_id)}, {"$set": fields})


def delete_student_quiz(db: Database, quiz_id: str) -> None:
    """Delete a student quiz and all its questions"""
    _, _, _, _, student_quizs, question_answers, _, _ = get_collections(db)
    
    # Delete all questions for this quiz
    question_answers.delete_many({"quiz_id": ObjectId(quiz_id)})
    
    # Delete the quiz
    student_quizs.delete_one({"_id": ObjectId(quiz_id)})


# Question Answer CRUD operations
def create_question_answer(db: Database, question_doc: Dict[str, Any]) -> str:
    """Create a new question answer"""
    _, _, _, _, _, question_answers, _, _ = get_collections(db)
    result = question_answers.insert_one(question_doc)
    return str(result.inserted_id)


def get_question_answer(db: Database, question_id: str) -> Dict[str, Any]:
    """Get a question answer by ID"""
    _, _, _, _, _, question_answers, _, _ = get_collections(db)
    return question_answers.find_one({"_id": ObjectId(question_id)})


def get_questions_by_quiz(db: Database, quiz_id: str) -> list:
    """Get all questions for a quiz"""
    _, _, _, _, _, question_answers, _, _ = get_collections(db)
    return list(question_answers.find({"quiz_id": ObjectId(quiz_id)}).sort("created_at", 1))


def update_question_answer(db: Database, question_id: str, fields: Dict[str, Any]) -> None:
    """Update a question answer"""
    _, _, _, _, _, question_answers, _, _ = get_collections(db)
    question_answers.update_one({"_id": ObjectId(question_id)}, {"$set": fields})


def delete_question_answer(db: Database, question_id: str) -> None:
    """Delete a question answer"""
    _, _, _, _, _, question_answers, _, _ = get_collections(db)
    question_answers.delete_one({"_id": ObjectId(question_id)})


# Users CRUD operations
def create_user(db: Database, user_doc: Dict[str, Any]) -> str:
    """Create a new user"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    result = users.insert_one(user_doc)
    return str(result.inserted_id)


def get_user(db: Database, user_id: str) -> Dict[str, Any]:
    """Get a user by ID"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    return users.find_one({"_id": ObjectId(user_id)})


def get_user_by_phone(db: Database, phone_number: str, phone_country_code: str = "+91") -> Dict[str, Any]:
    """Get a user by phone number"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    return users.find_one({
        "phone_number": phone_number,
        "phone_country_code": phone_country_code
    })


def get_user_by_email(db: Database, email: str) -> Dict[str, Any]:
    """Get a user by email"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    return users.find_one({"email": email})


def get_users_by_role(db: Database, role: str, limit: int = 50, skip: int = 0) -> list:
    """Get users by role with pagination"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    return list(users.find({"role": role})
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit))


def update_user(db: Database, user_id: str, fields: Dict[str, Any]) -> None:
    """Update a user"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    users.update_one({"_id": ObjectId(user_id)}, {"$set": fields})


def update_user_login(db: Database, user_id: str) -> None:
    """Update user's last login timestamp"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    users.update_one(
        {"_id": ObjectId(user_id)}, 
        {"$set": {"last_login_at": datetime.utcnow()}}
    )


def update_user_active_status(db: Database, user_id: str, is_active: bool) -> None:
    """Update user's active status and last active timestamp"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    update_fields = {"is_active": is_active}
    if is_active:
        update_fields["last_active_at"] = datetime.utcnow()
    
    users.update_one({"_id": ObjectId(user_id)}, {"$set": update_fields})


def delete_user(db: Database, user_id: str) -> None:
    """Delete a user and their associated student profile"""
    _, _, _, _, _, _, users, students = get_collections(db)
    
    # Delete student profile if exists
    students.delete_many({"user_id": ObjectId(user_id)})
    
    # Delete the user
    users.delete_one({"_id": ObjectId(user_id)})


def verify_phone_number(db: Database, user_id: str) -> None:
    """Mark phone number as verified"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    users.update_one(
        {"_id": ObjectId(user_id)}, 
        {"$set": {"is_phone_verified": True}}
    )


def verify_email(db: Database, user_id: str) -> None:
    """Mark email as verified"""
    _, _, _, _, _, _, users, _ = get_collections(db)
    users.update_one(
        {"_id": ObjectId(user_id)}, 
        {"$set": {"is_email_verified": True}}
    )


# Students CRUD operations
def create_student(db: Database, student_doc: Dict[str, Any]) -> str:
    """Create a new student profile"""
    _, _, _, _, _, _, _, students = get_collections(db)
    result = students.insert_one(student_doc)
    return str(result.inserted_id)


def get_student(db: Database, student_id: str) -> Dict[str, Any]:
    """Get a student profile by ID"""
    _, _, _, _, _, _, _, students = get_collections(db)
    return students.find_one({"_id": ObjectId(student_id)})


def get_student_by_user_id(db: Database, user_id: str) -> Dict[str, Any]:
    """Get student profile by user ID"""
    _, _, _, _, _, _, _, students = get_collections(db)
    return students.find_one({"user_id": ObjectId(user_id)})


def get_students_by_class(db: Database, current_class: str, limit: int = 50, skip: int = 0) -> list:
    """Get students by class with pagination"""
    _, _, _, _, _, _, _, students = get_collections(db)
    return list(students.find({"current_class": current_class})
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit))


def get_students_by_board(db: Database, board: str, limit: int = 50, skip: int = 0) -> list:
    """Get students by board with pagination"""
    _, _, _, _, _, _, _, students = get_collections(db)
    return list(students.find({"board": board})
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit))


def get_students_by_exam_target(db: Database, exam_target: str, limit: int = 50, skip: int = 0) -> list:
    """Get students by exam target with pagination"""
    _, _, _, _, _, _, _, students = get_collections(db)
    return list(students.find({"exam_target": exam_target})
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit))


def update_student(db: Database, student_id: str, fields: Dict[str, Any]) -> None:
    """Update a student profile"""
    _, _, _, _, _, _, _, students = get_collections(db)
    fields["updated_at"] = datetime.utcnow()
    students.update_one({"_id": ObjectId(student_id)}, {"$set": fields})


def update_student_by_user_id(db: Database, user_id: str, fields: Dict[str, Any]) -> None:
    """Update student profile by user ID"""
    _, _, _, _, _, _, _, students = get_collections(db)
    fields["updated_at"] = datetime.utcnow()
    students.update_one({"user_id": ObjectId(user_id)}, {"$set": fields})


def delete_student(db: Database, student_id: str) -> None:
    """Delete a student profile"""
    _, _, _, _, _, _, _, students = get_collections(db)
    students.delete_one({"_id": ObjectId(student_id)})


def delete_student_by_user_id(db: Database, user_id: str) -> None:
    """Delete student profile by user ID"""
    _, _, _, _, _, _, _, students = get_collections(db)
    students.delete_one({"user_id": ObjectId(user_id)})


