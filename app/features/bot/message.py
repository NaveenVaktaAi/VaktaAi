import ast
import asyncio
import json
import logging
from datetime import datetime
import re
from typing import List
import nltk
import numpy as np
from sqlalchemy import Integer
from app.features.bot.utils.confirmation_response import create_confirmation_response
from app.models.chat_message import ChatMessage

# Ensure the punkt tokenizer is available
nltk.download("punkt")
from sqlalchemy.sql import exists
from requests import Session
from sentry_sdk import capture_exception
from sqlalchemy import func, or_
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from translate import Translator
from app.database import get_db
from app.features.bot.semantic_search import BotGraphMixin
from app.features.bot.utils.response import ResponseCreator
from app.models.chunks import Chunk
from app.models.industry import Industry
from app.models.keywords import Keywords
from app.models.agent_ai_documents import AgentAIDocument
from app.models.legal_document_keywords import LegalDocumentKeywords
from app.models.legal_documnet_chunks import LegalDocumentChunks
from app.models.organization import Organization
from app.models.question_answer import QuestionAnswer
from app.utils.save_chat_messages import save_chat_message

from .utils.greetings_response import create_greetings_response
from .utils.thanking_response import create_thanking_response
from .utils.local_semantic_similarity import can_auto_reply
from .utils.local_semantic_similarity import can_auto_reply_thanking
from .websocket_response import WebSocketResponse

# from keybert import KeyBERT
from app.models.milvus.collection.milvus_collections import (
    chunk_msmarcos_collection,
    questions_msmarcos_collection,
    keywords_msmarcos_collection,
)


logger = logging.getLogger(__name__)


def printf():
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(timestamp, "timestamp")


class BotMessage:
    """
    A class that handles writing messages for the bot.
    """

    def __init__(
        self,
        *,
        chat_id: str,
        user_id: str,
        org_id: str,
        socket_response: WebSocketResponse,
        time_zone: str,
        user_message_token: str,
        language_code: str,
    ):
        self.chat_id = chat_id
        self.user_id = user_id
        self.org_id = org_id
        self.socket_response = socket_response
        self.time_zone = time_zone
        self.user_message_token = user_message_token
        self.language_code = language_code

    # Auto Replying to requests
    async def auto_reply_handler(
        self, last_user_text: str, auto_reply_type: str
    ) -> bool:
        """
        Handles if the bot can reply directly to the user.
        """

        print(f"LAST_USER_TEXT {last_user_text}\n")

        # Determine if the bot should auto-reply
        should_auto_reply = False
        if auto_reply_type == "greeting":
            should_auto_reply = await self.check_message_auto_reply(last_user_text)
            # can_auto_reply(
            #     last_user_text
            # ) or 
        elif auto_reply_type == "thanking":
            should_auto_reply = await self.check_appreciation_message_auto_reply(last_user_text) # can_auto_reply_thanking(last_user_text)
        
            
        elif auto_reply_type == "confirmation":
            should_auto_reply = await self.check_confirmation_message_auto_reply(
                last_user_text
            )
        print(f"AUTO_REPLY_TYPE {auto_reply_type},  SHOULD_AUTO_REPLY {should_auto_reply}\n")
 
        if not should_auto_reply:
            return False
        
        # Set response type and message based on the auto-reply type
        if auto_reply_type == "greeting":
            msg_type = "greetings_msg"
            msg = create_greetings_response()
        elif auto_reply_type == "thanking":
            msg_type = "thanking_msg"
            msg = create_thanking_response()
        elif auto_reply_type == "confirmation":
            msg_type = "confirmation_msg"
            msg = create_confirmation_response()

        # Save the chat message asynchronously
        asyncio.create_task(
            save_chat_message(
                self.chat_id,
                last_user_text,
                False,
                self.time_zone,
                None,  # `msg_type` is passed separately in the socket response
                self.user_message_token,
            )
        )

        # Send the bot's response
        await self.socket_response.create_bot_response(
            msg, self.time_zone, False, None, msg_type=msg_type
        )

        return True

    async def check_message_auto_reply(self, user_input: str) -> str:
        # prompt = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are an assistant that evaluates whether a user's input is a casual greeting or conversational text. "
        #             "Reply with 'true' if it is, or 'false' otherwise. Only respond with 'true' or 'false'."
        #         ),
        #     },
        #     {
        #         "role": "user",
        #         "content": user_input,
        #     },
        # ]
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that determines whether a user's input is a casual greeting or conversational statement. "
                    "Return 'true' if the input is a standalone greeting (e.g., 'Hi', 'Hello', 'Hey', 'What’s up') or a greeting followed by casual conversation (e.g., 'Hi, how are you?', 'Hello, what's up?'). "
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
    
    async def check_confirmation_message_auto_reply(self, user_input: str) -> str:
        # prompt = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are an assistant that evaluates whether a user's input is an Acknowledgment or Confirmation Response. "
        #             "An Acknowledgment/Confirmation Response includes replies such as 'ok,' 'got it,' 'noted,' 'understood,' or similar phrases indicating agreement, acceptance, or recognition. "
        #             "Reply with 'true' if the user's input is an Acknowledgment/Confirmation Response, otherwise reply with 'false.' "
        #             "Only respond with 'true' or 'false.'"
        #         ),
        #     },
        #     {
        #         "role": "user",
        #         "content": user_input,
        #     },
        # ]

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

    async def check_appreciation_message_auto_reply(self, user_input: str) -> str:
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

    async def send_bot_message(self, text: str) -> bool:
        """
        Sends a bot message to the user
        :param text: the text the user sent
        :returns: whether the bot has sent the response or not
        """
        # Task for handling the main bot logic
        return await asyncio.create_task(self.bot_handler(text))

        # is_answer_found_and_sent = await is_answer_found_and_sent_main_task
        # return is_answer_found_and_sent_main_task

    async def check_context_of_user_input(self, user_input: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an Intent identification assistant who has tell the intent of a query from this array [ Legal, Non legal] based on user input just return the type of query from this array ,,, Make sure the question is entirely legal rather than a conversational question. The intent should be identified only on the basis of extracting the legal info not based on user's intent to discuss legal matters"
                    f"Here is the current user input: {user_input}\n"
                ),
            },
        ]
        gpt_res = await ResponseCreator().gpt_response_without_stream(prompt)
        return gpt_res
    
    async def check_question_type(self,user_input: str) -> str:
        prompt = [
    {
        "role": "system",
        "content": (
            "You are an intelligent assistant. Analyze the user input and classify it into one of the following categories:\n"
            "- 'CONVERSATIONAL' if it is a normal chat or general discussion.\n"
            "- 'LEGAL/PROTECTED' if it contains legal, sensitive, or industry-specific content.\n"
            "\n"
            "Respond with either 'CONVERSATIONAL' or 'LEGAL/PROTECTED'.\n"
            "\n"
            f"User Input: {user_input}\n"
            "Classification:"
        )
    }
]


        chat_type = await ResponseCreator().gpt_response_without_stream(prompt)
        return chat_type 

    async def get_multi_response_from_gpt_based_on_user_input(
        self, user_input: str, industry_type: str, is_outside_from_industry: bool
    ) -> str:
        # Define industry context
        industry_context = (
            f"related to the {industry_type} industry."
            if is_outside_from_industry
            else "related to the user's input."
        )

        # Simplified and concise prompt

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an advanced AI designed to extract the most precise keywords, synonyms, and related terms from a user query, "
                    "ensuring high relevance and contextual accuracy for search optimization. Your goal is to maximize search precision "
                    "by focusing on intent-based keyword extraction.\n\n"

                    "### **Instructions:**\n"
                    "🔹 **Detect the language and translate to English if necessary.**\n"
                    "🔹 **Extract a maximum of 6-8 precise keywords**, avoiding irrelevant terms.**\n"
                    "🔹 **For each keyword, provide up to 7 contextually relevant synonyms** that match the user's intent and industry context.\n"  
                    "🔹 **Generate up to 7 related terms** that enhance the meaning of each keyword. Ensure the terms are related to the **user's intent, while avoiding redundancy.**\n"  
                    "🔹 **Prioritize synonyms & related terms** that match the **industry** and **user's query intent**.\n"  
                    "🔹 **Remove generic words** like 'the', 'in', 'what', etc., unless they are contextually necessary.\n"
                    "🔹 **Ensure extracted keywords align with the industry** context of the query, distinguishing between personal and business-related keywords.\n"  
                    "🔹 **Dynamically identify the query's intent** (e.g., personal name, business, industry) and segment keywords accordingly.\n"
                    "🔹 **Avoid redundancy** in synonyms and related terms. If a term has already been used, skip repeating it.\n"
                    "🔹 **Detect if the user query is a follow-up question** (i.e., it refers to a previously mentioned topic, seeks further details, or follows up on a past query).\n"
                    "🔹 **Return 'follow_up': true if the query is a follow-up question, otherwise return 'follow_up': false**.\n\n"

                    "### **Output Format (JSON):**\n"
                    "{\n"
                    '  "detected_language": "<detected language>",\n'
                    '  "translated_input": "<translated English text>",\n'
                    '  "language_code": "<ISO 639-1 language code>",\n'
                    '  "follow_up": <true/false>\n'
                    "}\n\n"

                    "### **Follow-Up Query Detection Rules:**\n"
                    "✔ The query is a follow-up if it:\n"
                    "  - Refers to something mentioned previously (e.g., 'Can you explain more?','give more','explain more','give more details' ,'What about its side effects?', 'And what are the costs?').\n"
                    "  - Asks for additional details about a previous response (e.g., 'How does it work?', 'Tell me more.').\n"
                    "  - Uses context-dependent phrases like 'What else?', 'Also?', 'And regarding that?'.\n"
                    "✔ The query is NOT a follow-up if it:\n"
                    "  - Introduces a new topic (e.g., 'What is machine learning?').\n"
                    "  - Is a generic greeting or standalone conversational phrase (e.g., 'Hello!', 'That makes sense.').\n\n"

                    "### **Example Output for Different Queries:**\n"

                    "#### **Follow-Up Query Example:**\n"
                    "{\n"
                    '  "detected_language": "English",\n'
                    '  "translated_input": "What about its side effects?",\n'
                    '  "language_code": "en",\n'
                    '  "follow_up": true\n'
                    "}\n\n"

                    "#### **New Topic Query Example:**\n"
                    "{\n"
                    '  "detected_language": "English",\n'
                    '  "translated_input": "What are the tax implications of capital gains?",\n'
                    '  "language_code": "en",\n'
                    '  "follow_up": false\n'
                    "}\n\n"

                    "#### **Conversational Statement Example:**\n"
                    "{\n"
                    '  "detected_language": "English",\n'
                    '  "translated_input": "That makes sense, thank you!",\n'
                    '  "language_code": "en",\n'
                    '  "follow_up": false\n'
                    "}\n\n"

                    f"User input: {user_input}"
                )
            }
        ]



        # Fetch response using the optimized GPT response function
        return await ResponseCreator().gpt_response_without_stream(prompt)
    
    
    
    async def process_bot_response(self, db, user_message, is_outside_from_document, is_outside_from_industry, industry_type, language_code):
        """
        Processes the bot response by searching answers, translating if needed, and generating GPT response.
        """
        try:
            bot_graph_mixin = BotGraphMixin(db=db)
            answer_ids = await bot_graph_mixin.search_answers(
                [user_message.lower()], chunk_msmarcos_collection, self.org_id
            )
            print(f"SEARCH_ANSWERS IDs: {answer_ids}\n")

            if answer_ids:
                chunk_results = (
                    db.query(Chunk.chunk, AgentAIDocument.type, AgentAIDocument.summary, AgentAIDocument.id)
                    .join(AgentAIDocument, Chunk.training_document_id == AgentAIDocument.id)
                    .filter(
                        Chunk.organization_id == str(self.org_id),
                        Chunk.id.in_(answer_ids)
                    )
                    .distinct()
                    .all()
                )

                if not chunk_results:
                    print("No relevant chunks found.\n")
                    return False
             
                print(f"CHUNK_RESULTS: {chunk_results}\n")
                formatted_results = [
                    {"context_data": chunk, "document_type": doc_type, "summary": summary,"document_id":id}
                    for chunk, doc_type, summary, id in chunk_results
                ]
                print("formatted_results>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", formatted_results)
                # Get GPT response from chunks
                gpt_resp = await self.get_chunk_response_from_gpt(user_message,
                    formatted_results
                )

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
                    await self.socket_response.create_bot_response(
                        generator,
                        self.time_zone,
                        False,
                        user_message,
                        msg_type="normal_msg",
                        doc_type=document_type,
                    )

                    return True

            # Fallback to GPT response if no answer found (or answer_ids is None)
            gpt_resp = await ResponseCreator().get_gpt_response(
                user_message,
                is_outside_from_document,
                is_outside_from_industry,
                None,
                industry_type,
                None,
                language_code,
                self.chat_id,
            )
            print("gpt_resp>>>>>>>>>>>>>>>>>>>>",gpt_resp)
            if not gpt_resp:
                return False

            if hasattr(gpt_resp, "__aiter__"):
                gpt_resp_text = "".join([chunk async for chunk in gpt_resp])
            else:
                gpt_resp_text = gpt_resp
                print(f"Final GPT Response: {gpt_resp_text}\n")


            # Streaming response generator
            async def string_to_generator(data):
                for char in data:
                    yield char
                    await asyncio.sleep(0.01)

            generator = string_to_generator(gpt_resp_text)

            # Send response to frontend
            await self.socket_response.create_bot_response(
                generator,
                self.time_zone,
                False,
                user_message,
                msg_type="normal_msg",
                doc_type='public',
            )

            return True   

        except Exception as e:
            print(f"Error in process_bot_response: {e}\n")
            capture_exception(e)
            return False

    
    
    def format_context_data(self, chunk_info):
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
 
    async def get_chunk_response_from_gpt(self, user_query: str, chunk_info: list) -> str:
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
    
    
    
    # Bot Handler
    async def bot_handler(self, user_input: str) -> bool:
        """
        Respond to the user input, optimizing the process of querying GPT-4.
        """
        try:
            db = next(get_db())
            if await self.auto_reply_handler(user_input, "greeting"):
                db.commit()
                return False

            if await self.auto_reply_handler(user_input, "thanking"):
                db.commit()
                return False
            
            if await self.auto_reply_handler(user_input, "confirmation"):
                db.commit()
                return False

            organization_details = (
                db.query(
                    Organization.outside_industry,
                    Organization.outside_document,
                    Industry.title.label("industry_type"),
                    Organization.industry_name.label("custom_industry_name"),
                )
                .outerjoin(Industry, Industry.id == Organization.industry_id)
                .filter(Organization.id == self.org_id)
                .first()
            )
    
            if organization_details:
                (
                    is_outside_from_industry,
                    is_outside_from_document,
                    industry_type,
                    custom_industry_name,
                ) = organization_details

            industry_type = (
                custom_industry_name
                if industry_type == "other" and custom_industry_name
                else industry_type
            )

            message_response = (
                await self.get_multi_response_from_gpt_based_on_user_input(
                    user_input=user_input,
                    industry_type=industry_type,
                    is_outside_from_industry=is_outside_from_industry,
                )
            )
            
            chat_history_flag = False
            if message_response:
                message_data = self.extract_json_from_text(message_response)
                user_message = message_data["translated_input"]
                language_code = message_data["language_code"]
                industry_type = None
                msg_type = None
                chat_history_flag = message_data["follow_up"]
            
            
            print("chat_history_flag=============================",chat_history_flag)            
            chat_history_text = None
            if chat_history_flag:
                results = (
                        db.query(ChatMessage)
                        .filter(ChatMessage.chat_id == self.chat_id)
                        .order_by(ChatMessage.id.desc())
                        .limit(2)
                        .all()
                    )

                # Format chat history
                chat_history_text = "\n".join(
                    [f"{'User' if not msg.is_bot else 'AI'}: {msg.message}" for msg in results]
                )
                user_input = f"{user_input}\nChat history: {chat_history_text}"
                

            asyncio.create_task(
                save_chat_message(
                    self.chat_id,
                    user_input,
                    False,
                    self.time_zone,
                    msg_type,
                    self.user_message_token,
                    message_context=None,
                )
            )
            
                     
            is_result_found = await self.process_bot_response(
                db, user_message, is_outside_from_document, is_outside_from_industry, industry_type, language_code
            )
            if is_result_found:
                db.commit()
                return False
            else:
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Error in bot_handler: {e}")
            capture_exception(e)
            db.rollback()
            return True

    

    async def handle_user_query_from_db(
    self,
    keyword_list: list[str],
    db,
    user_input: str,
    industry_type: str,
    language_code: str,
    is_outside_from_document: bool,
    is_outside_from_industry: bool,
):
        """
        Handles user queries by searching for matching chunks based on keywords.
        Uses BM25 ranking for relevance and translation before generating a response via LangChain.
        Falls back to GPT if no relevant chunks are found.
        """
        try:
            
            print("===============INSIDE HANDLE USER QUERY FROM DB==============") 
            
            question_flag = False
            query = db.query(
                QuestionAnswer.answer,
                QuestionAnswer.question,
                QuestionAnswer.id,
                AgentAIDocument.type,
            ).filter(
                QuestionAnswer.organization_id == self.org_id,
                or_(QuestionAnswer.question.ilike(f"%{user_input}%"))
            ).join(AgentAIDocument, QuestionAnswer.training_document_id == AgentAIDocument.id)
            
            results = query.all()
            if results:
                question_flag = True
            
            if not results:
                if not keyword_list:
                    return False  # Early exit if no keywords provided
                
                query = db.query(
                    Chunk.detail, Chunk.id, Chunk.keywords, AgentAIDocument.type
                ).filter(
                    Chunk.organization_id == self.org_id,
                    or_(*[Chunk.keywords.like(f"%{term}%") for term in keyword_list])
                ).join(AgentAIDocument, Chunk.training_document_id == AgentAIDocument.id)
                
                results = query.all()
                print("results================================>",results)
                if not results:
                    query = db.query(
                        Chunk.chunk, Chunk.id, Chunk.keywords, AgentAIDocument.type
                    ).filter(
                        Chunk.organization_id == self.org_id,
                        or_(*[Chunk.chunk.ilike(f"%{term}%") for term in keyword_list])
                    ).join(AgentAIDocument, Chunk.training_document_id == AgentAIDocument.id)
                    
                    results = query.all()
            
            if results:
                candidate_chunks = [row[0] for row in results if len(row) >= 4]
                
                print("candidate_chunks======================>", candidate_chunks)
                document_references = [row[3] for row in results if len(row) >= 4]
                selected_chunk, selected_document = None, None
                
                if not question_flag and candidate_chunks:
                    tokenized_corpus = [word_tokenize(text.lower()) for text in candidate_chunks]
                    tokenized_query = word_tokenize(" ".join(keyword_list).lower())
                    
                    bm25 = BM25Okapi(tokenized_corpus)
                    scores = bm25.get_scores(tokenized_query)
                    print("BM25 Scores:======================>", scores)
                    
                    if scores is None or len(scores) == 0 or np.all(scores < 1.0):
                        print("No highly relevant chunk found.")
                    else:
                        best_match_indices = np.where(scores == np.max(scores))[0]
                        selected_chunk = [candidate_chunks[i] for i in best_match_indices]
                        selected_documents = [document_references[i] for i in best_match_indices]
                        selected_document = selected_documents[0] if selected_documents else None
                        selected_chunk = Translator(to_lang=language_code).translate(selected_chunk)
                else:
                    selected_chunk = Translator(to_lang=language_code).translate(candidate_chunks)
                    selected_document = document_references[0]
                
                lang_chain_resp = await ResponseCreator().get_response_from_langchain(
                    user_input,
                    selected_chunk,
                    industry_type,
                    language_code,
                    is_outside_from_document,
                    is_outside_from_industry,
                    self.chat_id
                )
                
                if lang_chain_resp:
                    await self.socket_response.create_bot_response(
                        lang_chain_resp,
                        self.time_zone,
                        True,
                        user_input,
                        msg_type="normal_msg",
                        doc_type=selected_document
                    )
                    return True
            
            if not results and is_outside_from_document:
                gpt_resp = await ResponseCreator().get_gpt_response(
                    user_input,
                    is_outside_from_document,
                    is_outside_from_industry,
                    [],
                    industry_type,
                    keyword_list,
                    language_code,
                    self.chat_id
                )
                
                if gpt_resp:
                    async def string_to_generator(data):
                        async for item in data:
                            for char in item:
                                yield char
                                await asyncio.sleep(0.01)
                    
                    generator = string_to_generator(gpt_resp)
                    await self.socket_response.create_bot_response(
                        generator,
                        self.time_zone,
                        False,
                        user_input,
                        msg_type="normal_msg",
                    )
                    return True
            
            return False
        
        except Exception as e:
            print(f"Error in handle_user_query_from_db: {e}")
            capture_exception(e)
            return False 

    @staticmethod
    def parse_milvus_response(response):
        """Parses Milvus response and extracts entities safely."""
        if not response:
            print("⚠️ parse_milvus_response: Received None or Empty response!")
            return []  # Return empty list instead of None

        parsed_results = []
        
        try:
            for item in response:
                if isinstance(item, str):  # ✅ Convert string to JSON if needed
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON Decode Error: {e} | Item: {item}")
                        continue  # Skip this item

                if isinstance(item, dict) and "entity" in item:
                    parsed_results.append(item["entity"])
                elif isinstance(item, int):  # Handle IDs directly if needed
                    parsed_results.append(item)

            if not parsed_results:
                print("⚠️ parse_milvus_response: No valid entities found in response!")

            return parsed_results

        except Exception as e:
            print(f"❌ Error in parse_milvus_response: {e}")
            return []


    async def execute_search(self, db, questions_list, keywords_list, message_context, org_id):
        try:
            bot_graph_mixin = BotGraphMixin(db=db)
            keyword_collection = keywords_msmarcos_collection

            # ✅ Removed `await` from scalar() since it's a synchronous function
            question_exists = db.query(
                exists().where(func.lower(QuestionAnswer.question) == func.lower(questions_list[0]))
            ).scalar()

            print(f"🔍 Searching for question: {questions_list}")

            # ✅ Making async calls
            question_search_task = bot_graph_mixin.search_questions(
                questions_list, questions_msmarcos_collection, org_id
            )
            print("question_search_taskquestion_search_task",question_search_task)
            keyword_search_task = bot_graph_mixin.search_questions(
                keywords_list, keyword_collection, org_id
            )
            print("keyword_search_taskkeyword_search_task",keyword_search_task)
            question_search_result, keyword_search_result = await asyncio.gather(
                question_search_task, keyword_search_task
            )

            print(f"✅ Question Exists: {question_exists}")
            print(f"🟠 Raw Milvus Response: {question_search_result}")

            parsed_result = BotMessage.parse_milvus_response(question_search_result)

            if question_exists:
                return "question_milvus_task", parsed_result

            result_data = parsed_result if parsed_result else keyword_search_result
            result_source = "question_milvus_task" if result_data == parsed_result else "keyword_milvus_task"

            return result_source, result_data

        except Exception as e:
            print(f"❌ Error in execute_search: {e}")
            return None, None



    async def suggest_answer_from_previous_conversion(
        self, db: Session, chat_id: int, user_input: str
    ):
        """
        Suggest an answer from previous conversations if available.

        :param db: The database connection
        :param chat_id: The ID of the chat
        :param user_input: The user's input text
        :return: True if no answer is suggested, False if an answer is suggested and sent
        """
        try:
            # Get a response from the language chain if an answer is present in the chat history
            response = await ResponseCreator().get_response_from_langchain_if_answer_is_present_in_chat(
                db,
                chat_id,
                user_input,
            )
            # If a response is obtained
            if response:
                # Parse the response data
                parsed_response_data = self.extract_json_from_text(response)

                # Check if an answer is present in the parsed response data
                if (
                    "is_answer_present" in parsed_response_data
                    and parsed_response_data["is_answer_present"]
                ):
                    # Send the answer to the socket
                    await self.socket_response.create_bot_response(
                        parsed_response_data["answer"],
                        self.time_zone,
                        False,
                        user_input,
                        msg_type="normal_msg",
                    )
                    return False  # Answer suggested and sent

            return True  # No answer suggested from previous conversations
        except Exception as e:
            print(e, "suggest_answer_from_previous_conversion")
            capture_exception(e)
            return True  # Return false in case of any error

    
    async def handle_keywords_milvus_response(
            self,
            db: Session,
            keyword_ids: list[str],
            user_message: str,
            message_context: str,
            industry_type: str,
            keywords_list: list[str],
            language_code: str,
            is_outside_from_document: bool,
            is_outside_from_industry: bool,
        ) -> bool:
            """
            Handle responses from Milvus based on keyword IDs and stream the answer line by line.
            """
            try:
                print("Processing keyword IDs:", keyword_ids)

                # Fetch keyword-related chunks
                keyword_split = (
                    db.query(
                        Keywords.id.label("keyword_id"),
                        Keywords.organization_id,
                        func.unnest(func.string_to_array(Keywords.chunk_id, ",")).label("chunk_id_split")
                    )
                    .filter(
                        Keywords.organization_id == str(self.org_id),
                        func.coalesce(Keywords.chunk_id, '') != '',
                        func.coalesce(Keywords.chunk_id, '') != 'None'
                    )
                    .subquery()
                )

                chunk_results = (
                    db.query(Chunk.id, Chunk.chunk, AgentAIDocument.type)
                    .join(keyword_split, Chunk.id == func.cast(func.nullif(keyword_split.c.chunk_id_split, 'None'), Integer))
                    .join(AgentAIDocument, Chunk.training_document_id == AgentAIDocument.id)
                    .filter(
                        Keywords.id.in_(keyword_ids),
                        Keywords.organization_id == str(self.org_id),
                        Chunk.organization_id == str(self.org_id),
                    )
                    .distinct()
                    .all()
                )

                if not chunk_results:
                    print("No relevant chunks found.")
                    return False

                # Extract chunk details
                chunk_details = [(result.id, result.chunk, result.type) for result in chunk_results]
                print("Fetched Chunks:", chunk_details)

                # Tokenize chunks and query
                tokenized_corpus = [word_tokenize(chunk.lower()) for _, chunk, _ in chunk_details]
                tokenized_query = word_tokenize(" ".join(keywords_list).lower())

                # Compute BM25 scores
                bm25 = BM25Okapi(tokenized_corpus)
                scores = bm25.get_scores(tokenized_query)
                
                # Find the best-matching chunk based on highest BM25 score
                best_match_index = np.argmax(scores) if scores.any() else None
                
                if best_match_index is None:
                    print("No relevant chunk matched based on BM25.")
                    return False
                
                print("chunk_details[best_match_index]chunk_details[best_match_index]===========>",chunk_details[best_match_index])
                best_chunk_text, document_type = chunk_details[best_match_index]

                print("Best Matching Chunk:", best_chunk_text)
                print("Document Type:", document_type)

                # Translate if needed
                translator = Translator(to_lang=language_code)
                translated_chunk_text = translator.translate(best_chunk_text)
                print(f"Translated Chunk: {translated_chunk_text}")

                # Get GPT response
                gpt_resp = await ResponseCreator().get_gpt_response(
                    user_message,
                    is_outside_from_document,
                    is_outside_from_industry,
                    translated_chunk_text,
                    industry_type,
                    keywords_list,
                    language_code,
                    self.chat_id
                )

                if not gpt_resp:
                    return False

                # Convert async generator response to string
                gpt_resp_text = "".join([chunk async for chunk in gpt_resp])
                print("Final GPT Response:", gpt_resp_text)

                # Streaming response generator
                async def string_to_generator(data):
                    for char in data:
                        yield char
                        await asyncio.sleep(0.01)

                generator = string_to_generator(gpt_resp_text)

                # Send response to frontend
                await self.socket_response.create_bot_response(
                    generator,
                    self.time_zone,
                    False,
                    user_message,
                    msg_type="normal_msg",
                    doc_type=document_type,
                )

                return True

            except Exception as e:
                print(f"Error: {e}")
                capture_exception(e)
                return False

    async def handle_questions_milvus_response(
        self,
        db: Session,
        questions_ids: list[str],
        user_message: str,
        industry_type: str,
        language_code: str,
        message_context: str,
        is_outside_from_document: bool,
        is_outside_from_industry: bool,
    ) -> bool:
        """
        Handle responses from Milvus based on question chunks and stream the answer line by line in real-time.
        """
        try:
            print("==============INSIDE HANDLE QUESTIONS MILVUS RESPONSE==============")
            # Fetch question details
            question_details = (
                db.query(
                    QuestionAnswer.answer,
                    QuestionAnswer.training_document_id,
                    QuestionAnswer.id,
                    AgentAIDocument.type.label("document_type"),
                )
                .outerjoin(AgentAIDocument, QuestionAnswer.training_document_id == AgentAIDocument.id)
                .filter(
                    QuestionAnswer.organization_id == int(self.org_id),
                    QuestionAnswer.id.in_(questions_ids),
                )
                .all()
            )

            print('question_details', question_details)
            if not question_details:
                return False

            chunk_data = [q.answer for q in question_details]
            message_context_type = question_details[0][3]  # Document type

            if not chunk_data:
                return False

            print("chunk_data in Milvus:", chunk_data)

            # Get GPT response once
            gpt_resp = await ResponseCreator().get_gpt_response(
                user_message, is_outside_from_document, is_outside_from_industry, chunk_data, industry_type, language_code
            )

            if isinstance(gpt_resp, str):
                if language_code:
                    gpt_resp = Translator(to_lang=language_code).translate(gpt_resp)

            elif hasattr(gpt_resp, "__aiter__"):  # If async generator
                response_str = ""
                async for chunk in gpt_resp:
                    response_str += chunk

                if language_code:
                    response_str = Translator(to_lang=language_code).translate(response_str)
                
                gpt_resp = response_str

            # Stream response line by line
            async def string_to_generator(data: str):
                for i in range(0, len(data), 10):  # Send in chunks of 10 chars
                    yield data[i : i + 10]
                    await asyncio.sleep(0.01)

            generator = string_to_generator(gpt_resp)

            if generator:
                await self.socket_response.create_bot_response(
                    generator,
                    self.time_zone,
                    False,
                    user_message,
                    msg_type="normal_msg",
                    doc_type=message_context_type,
                )
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error in handle_questions_milvus_response: {e}")
            capture_exception(e)
            return False

    # def extract_json_from_text(self, text: str):
    #     # Define regular expression pattern to extract JSON-like content
    #     pattern = r"{[^{}]+}"
    #     # Search for JSON-like content in the text
    #     match = re.search(pattern, text)

    #     if match:
    #         # Extract the JSON-like content
    #         json_text = match.group(0)
    #         # Load the JSON data
    #         json_data = json.loads(json_text)
    #         return json_data
    #     else:
    #         return None


    def extract_json_from_text(self, text: str):
        """
        Extract complete JSON data from the given text.
        """
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

    # Todo remove this in future
    def format_question_answers(self, qa_list):
        output = ""
        flag = 1
        for qa_dict in qa_list:
            for question, answer in qa_dict.items():
                if flag:
                    output += f"**{answer.capitalize()}**\n"  # Capitalize the first letter of the answer
                    flag = 0
                else:
                    output += f"{answer.capitalize()}\n\n"
                    flag = 1
        return output
