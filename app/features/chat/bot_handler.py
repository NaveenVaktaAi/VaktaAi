import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import WebSocket
import numpy as np
from sqlalchemy import Integer, func, or_
from bson import ObjectId
from app.features.chat.utils.response import ResponseCreator
from app.schemas.milvus.collection.milvus_collections import chunk_msmarcos_collection
from app.schemas.schemas import ChunkBase, DocSathiAIDocumentBase
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from translate import Translator

from app.features.chat.repository import ChatRepository
from app.features.chat.websocket_manager import ChatWebSocketResponse
from app.features.chat.schemas import ChatMessageCreate
from app.database.session import get_db
from app.features.chat.semantic_search import BotGraphMixin


logger = logging.getLogger(__name__)


class MongoDBBotHandler:
    """
    Advanced bot message handler with full RAG functionality.
    Adapted from the bot system to work with MongoDB chat collections.
    """

    def __init__(
        self,
        chat_repository: ChatRepository,
        websocket_response: ChatWebSocketResponse,
        chat_id: str,
        user_id: str,
        document_id: str = None,
        time_zone: str = "UTC",
        language_code: str = "en"
    ):
        self.chat_repository = chat_repository
        self.websocket_response = websocket_response
        self.chat_id = chat_id
        self.user_id = user_id
        self.document_id = document_id
        self.time_zone = time_zone
        self.language_code = language_code
        self.db = next(get_db())

    async def handle_user_message(self, user_message: str) -> bool:
        """
        Handle a user message with full RAG functionality.
        Adapted from the bot system's bot_handler method.
        """
        try:
            # Create user message in MongoDB
            
            user_message_data = ChatMessageCreate(
                chat_id=self.chat_id,
                message=user_message,
                is_bot=False,
                type="text"
            )
            
            user_message_id = await self.chat_repository.create_chat_message(user_message_data)
            logger.info(f"User message created: {user_message_id}")

            # Check for auto-replies first (greetings, thanking, confirmation)
            if await self.auto_reply_handler(user_message, "greeting"):
                return True
            if await self.auto_reply_handler(user_message, "thanking"):
                return True
            if await self.auto_reply_handler(user_message, "confirmation"):
                return True
        
            
            # Process user input for keywords and language detection
            message_response = (
                await self.get_multi_response_from_gpt_based_on_user_input(
                    user_input=user_message,
                )
            )
            
            chat_history_flag = False
            if message_response:
                message_data = self.extract_json_from_text(message_response)
                processed_user_message = message_data["translated_input"]
                language_code = message_data["language_code"]
                chat_history_flag = message_data["follow_up"]
                
                # Update language code if detected
                self.language_code = language_code
            
            # Handle chat history for follow-up questions
            if chat_history_flag:
                chat_history_text = await self.get_chat_history()
                if chat_history_text:
                    user_message = f"{user_message}\nChat history: {chat_history_text}"

            # Process bot response with RAG
            is_result_found = await self.process_bot_response(
                self.db, 
                processed_user_message, 
                # is_outside_from_document,\
                language_code
            )
            
            if is_result_found:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error in handle_user_message: {e}")
            await self.send_error_response(str(e))
            return True

    # Auto-reply functionality from bot system
    async def auto_reply_handler(self, last_user_text: str, auto_reply_type: str) -> bool:
        """
        Handles if the bot can reply directly to the user.
        Adapted from the bot system.
        """
        print(f"LAST_USER_TEXT {last_user_text}\n")

        # Determine if the bot should auto-reply
        should_auto_reply = False
        if auto_reply_type == "greeting":
            should_auto_reply = await self.check_message_auto_reply(last_user_text)
        elif auto_reply_type == "thanking":
            should_auto_reply = await self.check_appreciation_message_auto_reply(last_user_text)
        elif auto_reply_type == "confirmation":
            should_auto_reply = await self.check_confirmation_message_auto_reply(last_user_text)
        elif auto_reply_type == "normal_chat":
            should_auto_reply = await self.check_normal_chat_message_auto_reply(last_user_text)
            print(f"SHOULD_AUTO_REPLY normal chat ---------- {should_auto_reply}\n")
        print(f"AUTO_REPLY_TYPE {auto_reply_type}, SHOULD_AUTO_REPLY {should_auto_reply}\n")

        if not should_auto_reply:
            return False
        
        # Set response type and message based on the auto-reply type
        if auto_reply_type == "greeting":
            msg_type = "greetings_msg"
            msg = self.create_greetings_response()
        elif auto_reply_type == "thanking":
            msg_type = "thanking_msg"
            msg = self.create_thanking_response()
        elif auto_reply_type == "confirmation":
            msg_type = "confirmation_msg"
            msg = self.create_confirmation_response()

        # Send the bot's response
        await self.websocket_response.create_streaming_response(
            msg, 
            f"auto_reply_{auto_reply_type}",
            "public",
            True
        )

        return True

    async def check_message_auto_reply(self, user_input: str) -> bool:
        """Check if message is a greeting using GPT"""
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that determines whether a user's input is a casual greeting or conversational statement. "
                    "Return 'true' if the input is a standalone greeting (e.g., 'Hi', 'Hello', 'Hey', 'What's up') or a greeting followed by casual conversation (e.g., 'Hi, how are you?', 'Hello, what's up?'). "
                    "Return 'false' if the input is a greeting followed by an informational or factual question (e.g., 'Hi, what is bike insurance?', 'Hello, what is banking?'). "
                    "Only respond with 'true' or 'false'."
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]
        gpt_res = await ResponseCreator().gpt_response_without_stream(prompt)
        return gpt_res.strip().lower() == "true"
    
    async def check_confirmation_message_auto_reply(self, user_input: str) -> bool:
        """Check if message is a confirmation using GPT"""
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that determines whether a user's input is an Acknowledgment or Confirmation Response. "
                    "An Acknowledgment/Confirmation Response includes short replies that indicate agreement, acceptance, or recognition, such as: 'ok,' 'got it,' 'noted,' 'understood,' 'done,' or similar phrases. "
                    "Also return 'true' if the acknowledgment includes polite closings like 'ok, bye' or 'ok, have a great day.' "
                    "Return 'false' if the response contains an Acknowledgment/Confirmation **followed by a further question or request for explanation**, such as 'ok, explain' or 'ok, what is bike insurance.' "
                    "Return 'false' if the response contains an Acknowledgment/Confirmation **followed by a statement requesting more details or information, even without question words**, such as 'great, principles of hotel management' or 'got it, insurance policies for bike.' "
                    "Only respond with 'true' or 'false.'"
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]
        gpt_res = await ResponseCreator().gpt_response_without_stream(prompt)
        return gpt_res.strip().lower() == "true"

    async def check_appreciation_message_auto_reply(self, user_input: str) -> bool:
        """Check if message is an appreciation using GPT"""
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that determines whether a user's input is an Appreciation Response. "
                    "An Appreciation Response includes short messages that express gratitude or satisfaction, such as: 'thank you,' 'appreciated,' 'perfect,' 'that's it,' 'great,' 'excellent,' or similar phrases. "
                    "Also return 'true' if the appreciation includes polite closings like 'thank you, have a nice day.' "
                    "Return 'false' if the response contains an appreciation **followed by a further question or request for information**, such as 'thank you, what is insurance?' or 'appreciated, now tell me about banking.' "
                    "Return 'false' if the response contains an appreciation **followed by a statement requesting more details or information, even without question words**, such as 'thank you, role of hotel management' or 'great, insurance policies for bike.' "
                    "Only respond with 'true' or 'false.'"
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]
        gpt_res = await ResponseCreator().gpt_response_without_stream(prompt)
        return gpt_res.strip().lower() == "true"

    async def check_normal_chat_message_auto_reply(self, user_input: str) -> bool:
        """Check if message is a normal chat using GPT"""
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that determines whether a user's input is a normal chat message. "
                     "like-  hello, how are you, what is your name, what are you doing, etc. "
                    "Return 'true' if the input is a normal chat message, otherwise return 'false'."
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]
        gpt_res = await ResponseCreator().gpt_response_without_stream(prompt)
        print(f"GPT_RES normal chat ---------- {gpt_res}\n")
        return gpt_res.strip().lower() == "true"

    def create_greetings_response(self) -> str:
        """Create greeting response"""
        greetings_responses = [
            "Hello! How can I help you today?",
            "Hi there! What would you like to know?",
            "Good day! How may I assist you?",
        ]
        import random
        return random.choice(greetings_responses)

    def create_thanking_response(self) -> str:
        """Create thanking response"""
        thanks_responses = [
            "You're welcome! I'm here to help.",
            "My pleasure! Feel free to ask if you need anything else.",
            "Happy to help! Is there anything else you'd like to know?",
        ]
        import random
        return random.choice(thanks_responses)

    def create_confirmation_response(self) -> str:
        """Create confirmation response"""
        return "Got it! Is there anything else you'd like to know?"

    # Advanced RAG functionality from bot system
    async def get_multi_response_from_gpt_based_on_user_input(
        self, user_input: str
    ) -> str:
        """Extract language from user input using GPT"""
     

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an advanced AI designed to extract the most precise keywords, synonyms, and related terms from a user query, "
                    "ensuring high relevance and contextual accuracy for search optimization. Your goal is to maximize search precision "
                    "by focusing on intent-based keyword extraction.\n\n"

                    "### **Instructions:**\n"
                    "🔹 **Detect the language and translate to English if necessary.**\n"
                   
                    "🔹 **Return 'follow_up': true if the query is a follow-up question, otherwise return 'follow_up': false**.\n\n"

                    "### **Output Format (JSON):**\n"
                    "{\n"
                    '  "detected_language": "<detected language>",\n'
                    '  "translated_input": "<translated English text>",\n'
                    '  "language_code": "<ISO 639-1 language code>",\n'
                    '  "follow_up": <true/false>\n'
                    "}\n\n"

                    f"User input: {user_input}"
                )
            }
        ]

        return await ResponseCreator().gpt_response_without_stream(prompt)

    async def process_bot_response(self, db, user_message, language_code):
        """
        Processes the bot response by searching answers, translating if needed, and generating GPT response.
        """
        try:
            bot_graph_mixin = BotGraphMixin(db=db)
            answer_ids = await bot_graph_mixin.search_answers(
                [user_message.lower()], chunk_msmarcos_collection, self.document_id
            )
            print(f"SEARCH_ANSWERS IDs: {answer_ids}\n")
            print("self.document_id-------------------doc--------", self.document_id)

            if answer_ids:
                try:
                    pipeline = [
                        {
                            "$match": {
                                "training_document_id": ObjectId(self.document_id),
                                "_id": {"$in": [ObjectId(aid) for aid in answer_ids]}
                            }
                        },
                        {
                            "$lookup": {
                                "from": "docSathi_ai_documents",
                                "localField": "training_document_id",
                                "foreignField": "_id",
                                "as": "document"
                            }
                        },
                        {"$unwind": "$document"},
                        {
                            "$project": {
                                "chunk": 1,
                                "doc_type": "$document.type",
                                "doc_id": "$document._id"
                            }
                        }
                    ]

                    chunk_results = list(db["chunks"].aggregate(pipeline))
                    print(f"CHUNK_RESULTS: {chunk_results}\n")

                    if not chunk_results:
                        print("No relevant chunks found.\n")
                        return False
                 
                    formatted_results = [
                        {
                            "context_data": result["chunk"], 
                            "document_type": result["doc_type"], 
                            "summary": "Default summary",
                            "document_id": str(result["doc_id"])
                        }
                        for result in chunk_results
                    ]
                except Exception as e:
                    print(f"ERROR_IN_CHUNK_QUERY: {str(e)}\n")
                    return False
                print("formatted_results>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", formatted_results)
                
                # Get GPT response from chunks
                gpt_resp = await self.get_chunk_response_from_gpt(user_message, formatted_results)

                response_data = self.extract_json_from_text(gpt_resp)
                print("response_data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",response_data)
                answer = response_data.get("answer")
                document_type = response_data.get("document_type")
                gpt_flag = response_data.get("GPT_FLAG")
                print(f"GPT_FLAG >>> {gpt_flag}\n")
                print(f"ANSWER >>> {answer}\n")

                if answer:
                    translator = Translator(to_lang=language_code)
                    translated_chunk_answer = translator.translate(answer)
                    print("inside the if condition>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    
                    # Streaming response generator
                    async def string_to_generator(data):
                        for char in data:
                            yield char
                            await asyncio.sleep(0.01)

                    generator = string_to_generator(translated_chunk_answer)
                    print(f"GENERATOR >>> {generator}\n")

                    # Send response to frontend
                    await self.websocket_response.create_streaming_response(
                        generator,
                        f"rag_response_{datetime.now().timestamp()}",
                        document_type,
                        True
                    )

                    return True

            # Fallback to GPT response if no answer found
            gpt_resp = await ResponseCreator().get_gpt_response(
                user_message,
                language_code,
                self.chat_id,
            )
            print("gpt_resp>>>>>>>>>>>>>>>>>>>>",gpt_resp)
            if not gpt_resp:
                return False
            print("gpt_resp>>>>>>>>after if condition>>>>>>>>>>>>",gpt_resp)
            if hasattr(gpt_resp, "__aiter__"):
                gpt_resp_text = "".join([chunk async for chunk in gpt_resp])
                print("gpt_resp_text>>>>>>>>in if condition>>>>>>>>>>>>",gpt_resp_text)
            else:
                gpt_resp_text = gpt_resp
                print(f"Final GPT Response: {gpt_resp_text}\n")

            # Streaming response generator
            async def string_to_generator(data):
                for char in data:
                    yield char
                    await asyncio.sleep(0.01)

            generator = string_to_generator(gpt_resp_text)
            print("generator>>>>>>>>final generator>>>>>>>>>>>>>",generator)

            # Send response to frontend
            await self.websocket_response.create_streaming_response(
                generator,
                f"gpt_response_{datetime.now().timestamp()}",
                'public',
                True
            )

            return True   

        except Exception as e:
            print(f"Error in process_bot_response: {e}\n")
            return False

    async def get_chunk_response_from_gpt(self, user_query: str, chunk_info: list) -> str:
        """Get GPT response based on chunk information"""
        try:
            if not isinstance(user_query, str) or not user_query.strip():
                print("⚠️ Warning: Invalid user query. Proceeding with a fallback response.")
                user_query = "Provide general information based on available data."
 
            if not isinstance(chunk_info, list) or len(chunk_info) == 0:
                print("⚠️ Warning: No context data provided. Proceeding with a fallback response.")
                return json.dumps({
                    "answer": "No relevant context was found, so here is a general response.",
                    "document_type": None,
                    "GPT_FLAG": "General Knowledge"
                })
 
            # Generate formatted context data and summary references
            formatted_chunks, summary_tracker = self.format_context_data(chunk_info)
            print("formatted_chunks>>>>>>>>>>>>>>>>>>>>>>>>>.",formatted_chunks)
            print("summary_tracker>>>>>>>>>>>>>>>>>>>>>>>>>.",summary_tracker)
 
            if not formatted_chunks:
                print("⚠️ Warning: No valid context found. Proceeding with a fallback response.")
                return json.dumps({
                    "answer": "No relevant context was found, so here is a general response.",
                    "document_type": None,
                    "GPT_FLAG": "General Knowledge"
                })
 
            # Construct the prompt with document_id linking
            prompt_text = f"""
            You are an intelligent AI assistant that answers user queries based on provided dynamic context data.
 
            ### User Query:
            {user_query}
 
            ### Context Data:
            {''.join(formatted_chunks)}
 
            ### Summaries (Reference by Document ID):
            {json.dumps(summary_tracker, indent=2)}
 
            ### Instructions:
            - Use the **summary from the reference section** for each `document_id` when generating answers.
            - If a **summary exists**, prioritize it for accuracy.
            - If no summary is available, rely only on **context_data**.
            - If no relevant context is found, provide a **general knowledge-based response**.
            - Respond in **JSON format**.
            - Find the **best matching** context chunk using similarity scoring.
            - Select **one document_type** based on the highest relevance.
            - If a **summary exists**, use it for better accuracy in generating answers.
            - If no summary is provided, rely only on **context_data** (do not generate a summary).
            - If no relevant context is found, provide a **general knowledge** answer.
            - Return the response in JSON format.
 
            ### JSON Response Format:
            ```json
            {{
                "answer": "<Generated Answer>",
                "document_type": "<Matched Document or null>",
                "GPT_FLAG": "<null or 'General Knowledge'>"
            }}
            ```
            """
 
            # Create proper message format for OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
 
            try:
                # Get response from GPT
                gpt_res = await ResponseCreator().gpt_response_without_stream(messages)
 
                if not gpt_res:
                    print("⚠️ Warning: Empty response from GPT API. Returning fallback.")
                    return json.dumps({
                        "answer": "I couldn't process your request at the moment.",
                        "document_type": None,
                        "GPT_FLAG": "General Knowledge"
                    })
 
                print("✅ Raw GPT response:", gpt_res)
                return gpt_res
 
            except Exception as api_error:
                print(f"🚨 GPT API Error: {api_error} - Returning fallback response.")
                return json.dumps({
                    "answer": "I'm sorry, I couldn't process your request.",
                    "document_type": None,
                    "GPT_FLAG": "General Knowledge"
                })
 
        except Exception as e:
            print(f"🚨 Unexpected error: {e} - Proceeding with fallback response.")
            return json.dumps({
                "answer": "An unexpected error occurred, but I am still able to assist with general knowledge.",
                "document_type": None,
                "GPT_FLAG": "General Knowledge"
            })

    def format_context_data(self, chunk_info):
        """Format context data for GPT processing"""
        formatted_chunks = []
        summary_tracker = {}  # Stores summaries by document_id
 
        for ctx in chunk_info:
            try:
                # Ensure ctx is parsed correctly
                if isinstance(ctx, str):
                    ctx = json.loads(ctx)  # Safe JSON parsing
 
                # Extract necessary fields with default fallbacks
                document_id = ctx.get("document_id", None)
                print("document_id===========================================>", document_id)
                document_type = ctx.get("document_type", "Unknown Document")
                context_data = ctx.get("context_data", "No context available.")
                summary = ctx.get("summary", "").strip()
 
                if not document_id:
                    print("⚠️ Warning: Missing 'document_id' in context. Skipping this chunk.")
                    continue  # Skip chunk but continue execution
 
                # Store summary separately if it's not already stored
                if document_id not in summary_tracker and summary:
                    summary_tracker[document_id] = summary  # Store unique summary
 
                # Build the formatted chunk with `document_id`
                formatted_chunk = f"Document ID: {document_id}\nDocument Type: {document_type}\nContext: {context_data}"
                formatted_chunks.append(formatted_chunk)
 
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                print(f"🚨 Error parsing context chunk: {e} - Skipping this chunk.")
                continue  # Skip invalid chunks but continue execution
 
        return formatted_chunks, summary_tracker  # Return both chunks & summary references

    def extract_json_from_text(self, text: str):
        """Extract complete JSON data from the given text."""
        # Find JSON content using regex
        pattern = r"\{.*\}"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            json_text = match.group(0)
            try:
                json_data = json.loads(json_text)  # Convert JSON string to Python object
                return json_data
            except json.JSONDecodeError as e:
                print("JSON Parsing Error:", e)
                return None
        return None

    async def get_chat_history(self) -> str:
        """Get chat history from MongoDB for follow-up questions"""
        try:
            # Get recent messages from MongoDB chat
            messages = await self.chat_repository.get_chat_messages(self.chat_id, page=1, limit=2)
            
            if messages:
                chat_history = []
                for msg in messages:
                    sender = "User" if not msg.get("is_bot", False) else "AI"
                    chat_history.append(f"{sender}: {msg.get('message', '')}")
                
                return "\n".join(chat_history)
            return ""
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return ""

    def is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        return any(greeting in message.lower() for greeting in greetings)

    def is_thanking(self, message: str) -> bool:
        """Check if message is thanking"""
        thanks = ["thank", "thanks", "appreciate", "grateful"]
        return any(thank in message.lower() for thank in thanks)

    def is_question(self, message: str) -> bool:
        """Check if message is a question"""
        return message.strip().endswith("?") or message.lower().startswith(("what", "how", "why", "when", "where", "who"))

    async def handle_greeting(self, message: str, language_code: str) -> str:
        """Handle greeting messages"""
        greetings_responses = {
            "en": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Good day! How may I assist you?",
            ],
            "hi": [
                "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
                "हैलो! आज मैं आपके लिए क्या कर सकता हूँ?",
                "नमस्कार! मैं आपकी सहायता कैसे कर सकता हूँ?",
            ]
        }
        
        responses = greetings_responses.get(language_code, greetings_responses["en"])
        import random
        return random.choice(responses)

    async def handle_thanking(self, message: str, language_code: str) -> str:
        """Handle thanking messages"""
        thanks_responses = {
            "en": [
                "You're welcome! I'm here to help.",
                "My pleasure! Feel free to ask if you need anything else.",
                "Happy to help! Is there anything else you'd like to know?",
            ],
            "hi": [
                "आपका स्वागत है! मैं मदद के लिए यहाँ हूँ।",
                "खुशी की बात है! अगर आपको और कुछ चाहिए तो बताइए।",
                "मदद करके खुशी हुई! क्या आपको और कुछ जानना है?",
            ]
        }
        
        responses = thanks_responses.get(language_code, thanks_responses["en"])
        import random
        return random.choice(responses)

    async def handle_question(self, message: str, language_code: str) -> str:
        """Handle question messages - integrate with your RAG system here"""
        # This is where you would integrate with your document search/RAG system
        # For now, return a generic response
        
        if language_code == "hi":
            return f"यह एक अच्छा सवाल है: '{message}' मैं आपके दस्तावेजों में इसकी जानकारी खोजने की कोशिश करता हूँ।"
        else:
            return f"That's a great question: '{message}' Let me search through your documents for relevant information."

    async def handle_general_message(self, message: str, language_code: str) -> str:
        """Handle general messages"""
        if language_code == "hi":
            return f"मैं समझ गया कि आप कहना चाहते हैं: '{message}' क्या आपको कोई विशिष्ट जानकारी चाहिए?"
        else:
            return f"I understand you're saying: '{message}' Is there anything specific you'd like to know about?"

    async def send_error_response(self, error_message: str):
        """Send error response to user"""
        try:
            error_response = ChatMessageCreate(
                chat_id=self.chat_id,
                message=f"I'm having trouble right now: {error_message}. Please try again.",
                is_bot=True,
                type="error"
            )
            
            await self.chat_repository.create_chat_message_with_websocket(
                error_response,
                self.websocket_response
            )
        except Exception as e:
            logger.error(f"Error sending error response: {e}")

    def calculate_token_count(self, text: str) -> int:
        """Calculate approximate token count for text"""
        # Simple word-based token estimation
        # You can replace this with actual tokenization if needed
        return len(text.split())

    async def send_typing_indicator(self):
        """Send typing indicator to show bot is thinking"""
        await self.websocket_response.connection_manager.send_to_chat(
            self.chat_id,
            {
                "mt": "typing_indicator",
                "chatId": self.chat_id,
                "isBot": True,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def stop_typing_indicator(self):
        """Stop typing indicator"""
        await self.websocket_response.connection_manager.send_to_chat(
            self.chat_id,
            {
                "mt": "stop_typing_indicator",
                "chatId": self.chat_id,
                "isBot": True,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def send_follow_up_questions(self, user_question: str, bot_response: str):
        """Send follow-up questions based on the conversation"""
        # This could be integrated with your existing follow-up question system
        follow_up_questions = [
            "Would you like to know more about this topic?",
            "Do you have any related questions?",
            "Is there anything else you'd like to explore?"
        ]
        
        # For now, just log that follow-up questions could be sent
        logger.info(f"Follow-up questions could be sent for: {user_question}")


class MongoDBBotMessage:
    """
    Main bot message handler class that integrates with the existing bot system patterns.
    Now includes full RAG functionality from the bot system.
    """
    
    def __init__(
        self,
        chat_id: str,
        user_id: str,
        document_id: str = None,
        websocket_response: ChatWebSocketResponse = None,
        timezone: str = "UTC",
        language_code: str = "en"
    ):
        self.chat_id = chat_id
        self.user_id = user_id
        self.document_id = document_id
        self.websocket_response = websocket_response
        self.timezone = timezone
        self.language_code = language_code
        
        # Initialize chat repository with WebSocket manager
        from app.features.chat.websocket_manager import websocket_manager
        self.chat_repository = ChatRepository(websocket_manager)
        
        # Initialize advanced bot handler with RAG functionality
        self.bot_handler = MongoDBBotHandler(
            self.chat_repository,
            self.websocket_response,
            self.chat_id,
            self.user_id,
            self.document_id,
            self.timezone,
            self.language_code
        )

    async def send_bot_message(self, user_message: str) -> bool:
        """
        Send bot message in response to user message with full RAG functionality.
        Returns True if successful, False if no answer found.
        """
        try:
            # Show typing indicator
            await self.bot_handler.send_typing_indicator()
            
            # Handle the user message with advanced RAG processing
            is_result_found = await self.bot_handler.handle_user_message(user_message)
            
            # Stop typing indicator
            await self.bot_handler.stop_typing_indicator()
            
            # Return whether result was found
            return not is_result_found  # Invert because True means no answer found in bot system
            
        except Exception as e:
            logger.error(f"Error in send_bot_message: {e}")
            await self.bot_handler.stop_typing_indicator()
            return True  # Return True to indicate no answer found
