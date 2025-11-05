import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
import hashlib
from functools import lru_cache
from fastapi import WebSocket
import numpy as np
from sqlalchemy import Integer, func, or_
from bson import ObjectId
from app.features.chat.utils.response import ResponseCreator
from app.schemas.milvus.collection.milvus_collections import chunk_msmarcos_collection
from app.schemas.schemas import ChunkBase, DocSathiAIDocumentBase
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
# from translate import Translator  # Replaced with OpenAI translation
from deep_translator import GoogleTranslator

from app.features.chat.repository import ChatRepository
from app.features.chat.websocket_manager import ChatWebSocketResponse
from app.features.chat.schemas import ChatMessageCreate
from app.database.session import get_db
from app.features.chat.semantic_search import BotGraphMixin
from langchain_community.tools import DuckDuckGoSearchResults

import os
from langchain_community.tools.tavily_search import TavilySearchResults
TAVILY_API_KEY="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"
tool=TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
from tavily import TavilyClient

tavily = TavilyClient(api_key=TAVILY_API_KEY)


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
        # Simple in-memory cache for frequent queries
        self._query_cache = {}
        self._cache_max_size = 100

    async def handle_user_message(self, user_message: str, use_web_search: bool = False) -> bool:
        """
        Handle a user message with full RAG functionality.
        Adapted from the bot system's bot_handler method.
        
        Args:
            user_message: The user's message
            use_web_search: Whether to enable web search (default: False, uses only document RAG)
        """
        try:
            # Debug: Check if user message already exists to prevent duplicates
            print(f"[DEBUG] Attempting to save user message: '{user_message}' for chat: {self.chat_id}")
            
            # Check for recent duplicate within last 5 seconds
            from datetime import datetime, timedelta
            recent_time = datetime.now() - timedelta(seconds=5)
            
            existing_messages = await self.chat_repository.get_chat_messages(
                self.chat_id, page=1, limit=5
            )
            
            # Check if this exact message was saved recently
            for msg in existing_messages:
                if (msg.message == user_message and 
                    msg.created_ts and 
                    msg.created_ts > recent_time):
                    print(f"[DEBUG] Duplicate user message detected, skipping save: {msg.id}")
                    user_message_id = msg.id
                    break
            else:
                # Create user message in MongoDB
                print(f"[DEBUG] Creating new user message in database")
                user_message_data = ChatMessageCreate(
                    chat_id=self.chat_id,
                    message=user_message,
                    is_bot=False,
                    type="text"
                )
                
                user_message_id = await self.chat_repository.create_chat_message(user_message_data)
                logger.info(f"User message created: {user_message_id}")

            # nornal_response = await self.get_normal_response(user_message)
            # if nornal_response:
            #     return True
            # Check for auto-replies first (greetings, thanking, confirmation)
            # if await self.auto_reply_handler(user_message, "greeting"):
            #     return True
            # if await self.auto_reply_handler(user_message, "thanking"):
            #     return True
            # if await self.auto_reply_handler(user_message, "confirmation"):
            #     return True
            chat_history_text = "There is no chat history"    
            chat_history_text = await self.get_chat_history()
            raw = await self.get_goforward_decision_and_response(user_message, chat_history_text)
            
            # Parse the raw response to get goforward decision
            try:
                if not raw or raw.strip() == "":
                    print("Empty response from goforward decision, continuing with RAG")
                    raise ValueError("Empty response")
                
                # Clean the response to extract JSON
                json_text = raw.strip()
                if json_text.startswith('```json'):
                    json_text = json_text[7:]
                if json_text.endswith('```'):
                    json_text = json_text[:-3]
                json_text = json_text.strip()
                
                # Parse JSON
                parsed_response = json.loads(json_text)
                
                # Check if goforward is false
                if not parsed_response.get("goforward", True):
                    # Use the immediate response from decision function
                    immediate_response = parsed_response.get("response", "")
                    if immediate_response:
                        print(f"💬 Using immediate response (goforward=False): {immediate_response}")
                        
                        # Create ultra-fast streaming generator for immediate response
                        async def string_to_generator(data):
                            chunk_size = 10  # Chunk size for streaming
                            delay = 0.01  # Small delay for smooth streaming
                            for i in range(0, len(data), chunk_size):
                                chunk = data[i:i+chunk_size]
                                yield chunk
                                await asyncio.sleep(delay)
                        
                        generator = string_to_generator(immediate_response)
                        print(f"🎬 Starting immediate response streaming...")
                        
                        # Send immediate response and get final text (immediate responses are AI-generated)
                        final_response = await self.websocket_response.create_streaming_response(
                            generator,
                            f"immediate_{datetime.now().timestamp()}",
                            True,
                            citation_source="ai"  # ✅ Immediate responses are AI-generated
                        )
                        
                        # Save bot message to database after streaming completes
                        try:
                            bot_message_data = ChatMessageCreate(
                                chat_id=self.chat_id,
                                message=final_response,
                                is_bot=True,
                                type="text"
                            )
                            bot_message_id = await self.chat_repository.create_chat_message(bot_message_data)
                            print(f"💾 Immediate response saved to DB: {bot_message_id}")
                        except Exception as save_error:
                            print(f"⚠️ Error saving immediate response to DB: {save_error}")
                        
                        return True
                    else:
                        print("No immediate response provided, continuing with RAG")
                
                # If goforward=True, continue with RAG processing
                print(f"Proceeding with RAG (goforward=True)")
                
            except (json.JSONDecodeError, ValueError) as parse_error:
                print(f"JSON parsing failed: {parse_error}")
                print(f"Raw response was: {raw}")
                # Fallback: continue with RAG processing
            except Exception as e:
                print(f"Error in goforward decision processing: {e}")
                # Fallback: continue with RAG processing
            
            # Process user input for keywords and language detection
            try:
                message_response = (
                    await self.get_multi_response_from_gpt_based_on_user_input(
                        user_input=user_message,
                        chat_history_text=chat_history_text,
                    )
                )
            
                processed_user_message = user_message  # Default to original message
                language_code = "en"  # Default language
                detected_language = "English"  # Default language
                
                if message_response:
                        try:
                            message_data = self.extract_json_from_text(message_response)
                            processed_user_message = message_data.get("translated_input", user_message)
                            language_code = message_data.get("language_code", "en")
                            detected_language = message_data.get("detected_language", "English")
                    
                # Update language code if detected
                            self.language_code = language_code
                        except Exception as lang_error:
                            print(f"Error in language detection: {lang_error}")
                        # Use defaults
                        
            except Exception as e:
                print(f"Error in message processing: {e}")
                processed_user_message = user_message
                language_code = "en"
                detected_language = "English"
            # Start web search in parallel ONLY if enabled by user
            web_search_task = None
            if use_web_search:
                print("🌐 Web Search ENABLED - Starting parallel web search...")
                
                async def fetch_web_results():
                    try:
                        # Run web search in thread pool since tool.invoke() is sync
                        import asyncio
                        loop = asyncio.get_event_loop()
                        results = await loop.run_in_executor(None, tool.invoke, processed_user_message)
                        if results:
                            print("results>>>>>>>tavily>>>>>>>>>>>>>>>>>>>>>>", results[0].get('title', 'No title'))
                            print("results>>>>>>>content>>>>>>>>>>>>>>>>>>>>>>", results[0].get('content', 'No content'))
                        return results
                    except Exception as e:
                        print(f"Error fetching web search results: {e}")
                        return []
                
                # Start web search task immediately
                web_search_task = asyncio.create_task(fetch_web_results())
            else:
                print("📄 Document-Only Mode - Web search DISABLED (using RAG only)")
            
            # Process bot response with RAG (this will also run in parallel)
            is_result_found = await self.process_bot_response(
                self.db, 
                processed_user_message,
                language_code,
                detected_language,
                web_search_task,  # Pass the task, not the results yet
                use_web_search=use_web_search  # ✅ Pass web search flag for citation logic
            )
            
            if is_result_found:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error in handle_user_message: {e}")
            print(f"Error in handle_user_message: {e}")
            await self.send_error_response(str(e))
            return True



    async def get_goforward_decision_and_response(
            self,
            user_input: str,
            chat_history_text: str
        ) -> Dict[str, Any]:
        """
        Ask the LLM to decide whether to proceed to RAG or handle immediately.

        Returns a dict with keys:
        - goforward: bool
        - response: str | None
        - detected_language: str
        - language_code: str
        - translated_input: str

        Behavior:
        - Single LLM call, strict JSON-only output required from LLM.
        - If parsing fails or LLM output doesn't conform, returns conservative fallback: goforward=True.
        """
        if not user_input or not user_input.strip():
            return {
                "goforward": True,
                "response": None,
                "detected_language": "unknown",
                "language_code": "",
                "translated_input": ""
            }

        system_prompt = f"""
    You are a classifier+assistant. Inspect the provided chat_history (the last two messages: previous user message and previous assistant reply) and the new user message.
    Decide whether the user's message should be handled immediately by a short assistant reply (auto-reply) OR escalated to the retrieval/RAG pipeline.

    REQUIREMENTS:
    - Use user_input and chat_history_text to resolve references when needed.
    - Detect language (English, Hindi, Hinglish, Marathi, Gujarati).
    - Produce a concise English normalized query in 'translated_input' that combines current user_input and the last two messages if necessary.

    OUTPUT RULES (MUST FOLLOW EXACTLY):
    Return ONLY ONE valid JSON object (no explanations, no markdown, no code fences) with these keys:

    {{
    "goforward": true|false,              // true -> caller should run RAG; false -> LLM's 'response' can be used immediately
    "response": "<short reply or null>", // if goforward=false, provide a short conversational response (<= 40 tokens). If goforward=true, set null.
    "detected_language": "<language name>",
    "language_code": "<ISO 639-1 code>",
    "translated_input": "<combined, standardized English query>"
    }}

    GUIDELINES:
    - If the user's message is casual chit-chat, greetings, thanks, or a short acknowledgement and does not require retrieval, set goforward=false and provide a short response.
    - If the user asks an informational/factual question, requests detailed steps, or continues a prior informational thread (follow-up), set goforward=true and set response to null.
    - If ambiguous, you may either (A) set goforward=false and include a short clarification in 'response', or (B) set goforward=true and leave response null. Use your judgment.
    - No chain-of-thought. Keep response short and helpful.

    EXAMPLES (ONLY JSON):
    # Greeting -> handle locally
    {{"goforward": false, "response": "Hi! How can I help you today?", "detected_language":"English", "language_code":"en", "translated_input": "User greets the assistant."}}

    # Follow-up informational -> go to RAG
    {{"goforward": true, "response": null, "detected_language":"Hindi", "language_code":"hi", "translated_input": "Is the vaccination slot open today for Pune?"}}
    """

        user_payload = (
            f"CHAT_HISTORY_LAST_TWO:\n{chat_history_text}\n\n"
            f"USER_INPUT:\n{user_input}\n\n"
            "Respond with the single JSON object described above and nothing else."
        )

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ]

        try:
            raw = await ResponseCreator().gpt_response_without_stream(prompt)
            print("raw>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",raw)
        except Exception as e:
            # LLM call failed — conservatively continue to RAG
            print("[get_goforward_decision_and_response] LLM call failed:", e)
            return {
                "goforward": True,
                "response": None,
                "detected_language": "unknown",
                "language_code": "",
                "translated_input": ""
            }
        return raw






    # Advanced RAG functionality from bot system
    async def get_multi_response_from_gpt_based_on_user_input(
        self, user_input: str, chat_history_text: str
    ) -> str:
        """Extract language, translate, and detect follow-up from user input using GPT"""
     

        prompt = [
            {
                "role": "system",
                "content": f"""You are an AI assistant whose job is to DETECT THE LANGUAGE of a user's message and to PRODUCE
a single, standardized ENGLISH query that combines the current user input with the last two
chat messages (the previous user message and the assistant reply).

REQUIREMENTS:
1) Input sources you must use: the provided user_input string and the provided chat_history
   (which contains the last two messages: previous user message and previous assistant message).
   Use those to resolve references (like 'uske', 'it', 'that') when possible and create a single
   clear, self-contained English query that captures the user's intent.
2) Detect the language of the user_input. Supported languages: English, Hindi, Hinglish, Marathi, Gujarati.
   Provide the language name in detected_language and the ISO 639-1 code in language_code.
3) Translate and rewrite: produce translated_input — one concise, natural-sounding English sentence or
   short paragraph that represents the user's intended request after incorporating relevant context
   from the last two chat messages. Expand vague references using the chat history.

OUTPUT RULES (MUST BE FOLLOWED EXACTLY):
- Return ONLY valid JSON (no surrounding text, no markdown) with exactly these three keys:
{{
  "detected_language": "<language name>",
  "translated_input": "<combined, standardized English query>",
  "language_code": "<ISO 639-1 code>"
}}

    ADDITIONAL GUIDELINES:
    - Keep translated_input complete and actionable.if user_input is independent then keep it as it is else use chat history if needed, else only user input.
      EXAMPLE: 
         in chat history: user_input -  "what is the frequency and sound of the radio wave".
                          assistant_reply -  "The frequency of a radio wave is the number of oscillations per second, and the sound of a radio wave is the pitch of the sound it makes when it is received by a radio receiver."
         now is user_input -  "Find the following for given wave equation.
 P = 0.02 sin [(3000 t – 9 x] (all quantities are in S.I. units.)
 (a) Frequency (b) Wavelength (c) Speed of sound wave
 (d) If the equilibrium pressure of air is in 105
 Pa then find maximum and minimum pressure. ".
        then translated_input should not combine user_input and chat history, it should be only user_input because user_input is independent.

    - If user explicitly asks for the response in a particular language, set `detected_language` to that language.
    - and include a note in `translated_input` specifying the required response language.
    - Preserve intent and parameters from user input.

Payload:
USER_INPUT: {user_input}
CHAT_HISTORY_LAST_TWO: {chat_history_text}
"""
            }
       ]



        return await ResponseCreator().gpt_response_without_stream(prompt)

    def _get_cache_key(self, query: str, document_id: str = None) -> str:
        """Generate cache key for query"""
        cache_string = f"{query.lower().strip()}_{document_id or 'default'}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str):
        """Get result from cache"""
        if cache_key in self._query_cache:
            print(f"🎯 Cache HIT for query")
            return self._query_cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, result):
        """Set result in cache with size limit"""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = result
        print(f"💾 Cached query result")

    async def process_bot_response(self, db, user_message, language_code, detected_language, web_search_task=None, use_web_search=False):
        """
        Processes the bot response by searching answers, translating if needed, and generating GPT response.
        """
        try:
            print("user_message>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",user_message)
            bot_graph_mixin = BotGraphMixin(db=db)
            # Optimized semantic search with caching and timing
            cache_key = self._get_cache_key(user_message, self.document_id)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result:
                answer_ids = cached_result
                print(f"🎯 Using cached search results: {len(answer_ids)} IDs")
            else:
                print("🔍 Starting semantic search...")
                search_start = asyncio.get_event_loop().time()
                answer_ids = await bot_graph_mixin.search_answers(
                [user_message.lower()], chunk_msmarcos_collection, self.document_id
            )
                search_end = asyncio.get_event_loop().time()
                print(f"⚡ Search completed in {search_end - search_start:.3f}s")
                self._set_cache(cache_key, answer_ids)
            
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
                                "doc_id": "$document._id",
                                "doc_summary": "$document.summary"
                            }
                        }
                    ]

                    chunk_results = list(db["chunks"].aggregate(pipeline))

                    if not chunk_results:
                        print("No relevant chunks found.\n")
                        return False
                 
                    formatted_results = [
                        {
                            "context_data": result["chunk"],
                            "summary": result.get("doc_summary", "No summary available"),
                            "document_id": str(result["doc_id"])
                        }
                        for result in chunk_results
                    ]
                except Exception as e:
                    print(f"ERROR_IN_CHUNK_QUERY: {str(e)}\n")
                    return False
                # print("formatted_results>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", formatted_results)
                
                # Get web search results if task is provided
                web_results = []
                if web_search_task:
                    try:
                        web_results = await web_search_task
                        print(f"✅ Web search completed with {len(web_results)} results")
                    except Exception as e:
                        print(f"❌ Web search failed: {e}")
                        web_results = []
                
                # Rule 1: If user enabled web search, always use "web_search" citation
                # For other cases, we'll determine citation AFTER GPT response (based on actual usage)
                initial_citation_source = None
                if use_web_search:
                    initial_citation_source = "web_search"
                    print(f"📚 Citation source: {initial_citation_source} (user enabled web search)")
                
                # Get GPT response from chunks with web results (optimized)
                print("🤖 Starting GPT response generation...")
                gpt_start = asyncio.get_event_loop().time()
                
                # Get the generator (don't await it!)
                gpt_resp_generator = self.get_chunk_response_from_gpt(
                    user_message, 
                    formatted_results, 
                    detected_language, 
                    web_results
                )
                
                print("📡 Sending streaming response to frontend...")
                # Create a wrapper generator that will determine citation after response
                response_text_container = {"text": ""}
                message_id = f"rag_response_{datetime.now().timestamp()}"
                
                async def wrapped_generator():
                    async for chunk in gpt_resp_generator:
                        response_text_container["text"] += chunk
                        yield chunk
                
                # Send response to frontend - pass the wrapped generator
                # This returns the complete text after streaming
                final_response = await self.websocket_response.create_streaming_response(
                    wrapped_generator(),
                    message_id,
                    True,
                    citation_source=None  # Will be determined after response
                )
                
                # ✅ Determine actual citation source based on GPT response analysis
                # Check if GPT actually used chunks or relied on its own knowledge (70-80% threshold)
                actual_response = response_text_container["text"] or final_response
                if initial_citation_source:
                    citation_source = initial_citation_source  # Web search already set
                else:
                    citation_source = self._determine_citation_from_response(
                        actual_response,
                        formatted_results,
                        user_message,
                        web_results
                    )
                    print(f"📚 Final citation source determined: {citation_source}")
                
                # ✅ Send citation source update via WebSocket metadata
                # Update the stop message with correct citation source
                try:
                    await self.websocket_response.connection_manager.send_personal_message(
                        self.websocket_response.websocket,
                        {
                            "mt": "chat_message_bot_partial",
                            "chatId": self.chat_id,
                            "stop": message_id,
                            "citation_source": citation_source,  # Add citation to stop message
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    print(f"📡 Citation source update sent: {citation_source}")
                except Exception as e:
                    print(f"⚠️ Failed to send citation update: {e}")
                
                gpt_end = asyncio.get_event_loop().time()
                print(f"✅ GPT streaming completed in {gpt_end - gpt_start:.3f}s")
                
                # Save bot message to database after streaming completes
                try:
                    bot_message_data = ChatMessageCreate(
                        chat_id=self.chat_id,
                        message=final_response,
                        is_bot=True,
                        type="text"
                    )
                    bot_message_id = await self.chat_repository.create_chat_message(bot_message_data)
                    print(f"💾 Bot message saved to DB: {bot_message_id}")
                except Exception as save_error:
                    print(f"⚠️ Error saving bot message to DB: {save_error}")
                    # Don't fail the request if DB save fails

                return True

            # Get web search results for fallback case too
            web_results = []
            if web_search_task:
                try:
                    web_results = await web_search_task
                    print(f"✅ Web search completed for fallback with {len(web_results)} results")
                except Exception as e:
                    print(f"❌ Web search failed in fallback: {e}")
                    web_results = []

            # ✅ Determine citation source for fallback
            # If user enabled web search, always use "web_search"
            if use_web_search:
                citation_source = "web_search"
            elif web_results and len(web_results) > 0:
                citation_source = "web_search"  # Web search available but no documents
            else:
                citation_source = "ai"  # Pure AI response
            print(f"📚 Fallback citation source: {citation_source}")

            # Fallback to GPT response if no answer found
            print("📝 No relevant chunks found, using fallback GPT response...")
            gpt_resp = await ResponseCreator().get_gpt_response(
                user_message,
                language_code,
                self.chat_id,
            )
            if not gpt_resp:
                print("⚠️ Fallback GPT response also failed")
                return False
                
            if hasattr(gpt_resp, "__aiter__"):
                gpt_resp_text = "".join([chunk async for chunk in gpt_resp])
            else:
                gpt_resp_text = gpt_resp
            
            print(f"✅ Fallback GPT Response length: {len(gpt_resp_text)} chars")

            # Ultra-fast streaming response generator
            async def string_to_generator(data):
                chunk_size = 10  # Chunk size for streaming
                delay = 0.01  # Small delay for smooth streaming
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size]
                    yield chunk
                    await asyncio.sleep(delay)

            generator = string_to_generator(gpt_resp_text)

            # Send response to frontend and get final text
            final_response = await self.websocket_response.create_streaming_response(
                generator,
                f"gpt_fallback_{datetime.now().timestamp()}",
                True,
                citation_source=citation_source  # ✅ Pass citation source
            )
            
            # Save bot message to database after streaming completes
            try:
                bot_message_data = ChatMessageCreate(
                    chat_id=self.chat_id,
                    message=final_response,
                    is_bot=True,
                    type="text"
                )
                bot_message_id = await self.chat_repository.create_chat_message(bot_message_data)
                print(f"💾 Fallback bot message saved to DB: {bot_message_id}")
            except Exception as save_error:
                print(f"⚠️ Error saving fallback bot message to DB: {save_error}")

            return True   

        except Exception as e:
            print(f"Error in process_bot_response: {e}\n")
            return False

    async def get_chunk_response_from_gpt(self, user_query: str, chunk_info: list, detected_language: str, web_results: list):
        """
        Get GPT response based on chunk information as an async generator.
        Yields tokens as they arrive for real-time streaming.
        """
        try:
            # Validate user query
            if not isinstance(user_query, str) or not user_query.strip():
                print("⚠️ Warning: Invalid user query. Proceeding with a fallback response.")
                user_query = "Provide general information based on available data."
 
            # Validate chunk info
            if not isinstance(chunk_info, list) or len(chunk_info) == 0:
                print("⚠️ Warning: No context data provided. Proceeding with a fallback response.")
                fallback_msg = "No relevant context was found. Let me provide a general response based on your question."
                # Yield fallback message in chunks for streaming effect
                for i in range(0, len(fallback_msg), 10):
                    yield fallback_msg[i:i+10]
                    await asyncio.sleep(0.01)
                return
            
            print("🔍 Processing query:", user_query)    
 
            # Generate formatted context data and summary references
            formatted_chunks, summary_tracker = self.format_context_data(chunk_info)
 
            if not formatted_chunks:
                print("⚠️ Warning: No valid context found. Proceeding with a fallback response.")
                fallback_msg = "No relevant context was found. Let me provide a general response based on your question."
                for i in range(0, len(fallback_msg), 10):
                    yield fallback_msg[i:i+10]
                    await asyncio.sleep(0.01)
                return
                
            # Construct the prompt with document_id linking
            # ✅ Add relevance check instruction for citation logic
            relevance_instruction = ""
            if not web_results:  # Only check relevance if web search is OFF
                relevance_instruction = """
                ### IMPORTANT RELEVANCE CHECK:
                - First, carefully analyze if the Context Chunks are actually relevant to the User Query.
                - If the chunks are NOT relevant (e.g., physics question asked but chemistry document chunks retrieved), 
                  IGNORE the chunks and provide a general knowledge answer based on your training.
                - Only use the Context Chunks if they are genuinely related to the user's query.
                - If chunks are irrelevant, your response should indicate you're using general knowledge, not document content.
                """
            
            prompt_text = f"""
                You are an intelligent AI assistant that answers the user's query.

                ### User Query:
                {user_query}

                ### Context Chunks:
                {''.join(formatted_chunks)}

                ### Summaries:
                {json.dumps(summary_tracker, indent=2)}

                ### Web Search Results:
                {web_results[0].get('content', 'No additional web context available') if web_results else "No additional web context available"}

                {relevance_instruction}

                ### Instructions:
                - Respond in the hinglish or english user's language: {detected_language}.
                - Use summaries if available; otherwise rely on context chunks (ONLY if they are relevant to the query).
                - If web search results are available, use them for additional context.
                - Provide an answer combining your knowledge, context chunks (if relevant), and web results.
                - If detailed answer is needed, provide a detailed response using all available sources.
                - If no relevant info in chunks, provide a detailed general knowledge answer.
                - Respond as a teacher would to a student.
                """
 
            # Create proper message format for OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds as a teacher would to a student."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
 
            # Stream response from GPT
            print("🤖 Starting GPT streaming...")
            token_count = 0
            
            try:
                async for token in ResponseCreator().get_streaming_response(messages):
                    if token:
                        token_count += 1
                        if token_count % 50 == 0:  # Log every 50 tokens
                            print(f"📝 Streamed {token_count} tokens...")
                        yield token
                
                if token_count == 0:
                    print("⚠️ Warning: Empty response from GPT API. Sending fallback.")
                    fallback_msg = "I couldn't generate a proper response at the moment. Please try again."
                    for i in range(0, len(fallback_msg), 10):
                        yield fallback_msg[i:i+10]
                        await asyncio.sleep(0.01)
                else:
                    print(f"✅ Streaming completed successfully with {token_count} tokens")
 
            except asyncio.TimeoutError:
                print("🚨 GPT API Timeout - Sending timeout message")
                error_msg = "The response took too long. Please try again with a shorter query."
                for i in range(0, len(error_msg), 10):
                    yield error_msg[i:i+10]
                    await asyncio.sleep(0.01)
                    
            except Exception as api_error:
                print(f"🚨 GPT API Error during streaming: {api_error}")
                error_msg = "I encountered an error while generating the response. Please try again."
                for i in range(0, len(error_msg), 10):
                    yield error_msg[i:i+10]
                    await asyncio.sleep(0.01)
 
        except Exception as e:
            print(f"🚨 Unexpected error in get_chunk_response_from_gpt: {e}")
            import traceback
            traceback.print_exc()
            error_msg = "An unexpected error occurred. Please try again later."
            for i in range(0, len(error_msg), 10):
                yield error_msg[i:i+10]
                await asyncio.sleep(0.01)

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

    def _determine_citation_from_response(
        self,
        gpt_response: str,
        formatted_chunks: list,
        user_query: str,
        web_results: list
    ) -> str:
        """
        Determine citation source by analyzing if GPT actually used chunks or relied on own knowledge.
        Uses keyword matching and chunk content comparison to check if response is 70-80% from chunks.
        
        Args:
            gpt_response: The GPT-generated response
            formatted_chunks: List of document chunks that were provided
            user_query: Original user query
            web_results: Web search results (if any)
            
        Returns:
            "document" if chunks were used (70%+ from chunks)
            "web_search" if web results were used
            "ai" if GPT used mostly its own knowledge
        """
        try:
            # Rule 1: If no chunks provided, definitely AI
            if not formatted_chunks or len(formatted_chunks) == 0:
                return "ai"
            
            # Rule 2: Check if response contains indicators that chunks were ignored
            response_lower = gpt_response.lower()
            ignore_keywords = [
                "not found in the document",
                "not in the provided context",
                "not mentioned in",
                "general knowledge",
                "based on my knowledge",
                "the document doesn't contain",
                "not available in the context",
                "outside the scope",
                "irrelevant"
            ]
            
            for keyword in ignore_keywords:
                if keyword in response_lower:
                    print(f"📚 GPT indicated chunks were ignored (keyword: {keyword})")
                    return "ai"
            
            # Rule 3: Check content similarity - extract key phrases from chunks
            # Simple approach: Check if significant chunk content appears in response
            chunk_content_used = False
            chunk_phrases = []
            
            for chunk_info in formatted_chunks:
                if isinstance(chunk_info, dict):
                    chunk_text = chunk_info.get("context_data", "")
                else:
                    chunk_text = str(chunk_info)
                
                # Extract key phrases from chunk (simple: first 50-100 words)
                words = chunk_text.split()[:50]
                key_phrase = " ".join(words).lower()
                chunk_phrases.append(key_phrase)
                
                # Check if this phrase appears in response
                if len(key_phrase) > 20 and key_phrase in response_lower:
                    chunk_content_used = True
                    break
            
            # Rule 4: Check response length vs chunk content
            # If response is very long but doesn't match chunks, likely AI-generated
            response_length = len(gpt_response.split())
            if response_length > 100 and not chunk_content_used:
                # Check if response contains unique terms not in chunks
                response_words = set(response_lower.split())
                chunk_words = set()
                for phrase in chunk_phrases:
                    chunk_words.update(phrase.split())
                
                # If > 70% of response words are NOT in chunks, likely AI
                unique_ratio = len(response_words - chunk_words) / len(response_words) if response_words else 0
                if unique_ratio > 0.7:
                    print(f"📚 Response has {unique_ratio*100:.1f}% unique content (not from chunks)")
                    return "ai"
            
            # Rule 5: If chunks were provided and response seems to use them, return "document"
            if chunk_content_used:
                return "document"
            
            # Rule 6: Default to "document" if chunks provided, but with caution
            # If response is very short or generic, likely AI
            if response_length < 30:
                print(f"📚 Response too short ({response_length} words), likely AI")
                return "ai"
            
            # If we have chunks but couldn't verify usage, assume document (conservative)
            # But log a warning
            print(f"⚠️ Could not definitively determine citation source, defaulting to 'document'")
            return "document"
            
        except Exception as e:
            print(f"🚨 Error in citation determination: {e}")
            # Fallback: if chunks provided, assume document
            if formatted_chunks and len(formatted_chunks) > 0:
                return "document"
            return "ai"
    
    def extract_json_from_text(self, text: str):
        """Extract complete JSON data from the given text."""
        # Find JSON content using regex
        pattern = r"\{.*\}"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            json_text = match.group(0)
            try:
                json_data = json.loads(json_text)
                return json_data
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error: {e}")
                try:
                    # Extract fields using regex to avoid escape sequence issues
                    answer_match = re.search(r'"answer"\s*:\s*"(.*?)"(?=\s*,\s*")', json_text, re.DOTALL)
                    doc_type_match = re.search(r'"document_type"\s*:\s*(".*?"|null)', json_text)
                    gpt_flag_match = re.search(r'"GPT_FLAG"\s*:\s*(".*?"|null)', json_text)
                    if answer_match:
                        answer = answer_match.group(1)
                        doc_type = doc_type_match.group(1).strip('"') if doc_type_match and doc_type_match.group(1) != "null" else None
                        gpt_flag = gpt_flag_match.group(1).strip('"') if gpt_flag_match and gpt_flag_match.group(1) != "null" else None
                        
                        return {
                            "answer": answer,
                            "document_type": doc_type,
                            "GPT_FLAG": gpt_flag,
                        }
                except Exception as e2:
                    print(f"Regex extraction failed: {e2}")
                return None
        return None

    async def get_chat_history(self) -> str:
        """Get last 2 chat messages from MongoDB for follow-up questions"""
        try:
            print("inside get_chat_history>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # Get last 2 messages using a special method
            messages = await self.chat_repository.get_last_chat_messages(
                self.chat_id, limit=2
            )
            print("messages>>>>>>>>>>history>>>>>>>>>>>>>>>.", messages)

            if messages:
                chat_history = []
                for msg in messages:
                    sender = "User" if not msg.is_bot else "AI"
                    chat_history.append(f"{sender}: {msg.message}")

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

    async def send_bot_message(self, user_message: str, use_web_search: bool = False) -> bool:
        """
        Send bot message in response to user message with full RAG functionality.
        Returns True if successful, False if no answer found.
        
        Args:
            user_message: The user's message
            use_web_search: Whether to enable web search (default: False)
        """
        try:
            # Show typing indicator
            await self.bot_handler.send_typing_indicator()
            
            # Handle the user message with advanced RAG processing
            is_result_found = await self.bot_handler.handle_user_message(user_message, use_web_search=use_web_search)
            print("is_result_found>>>>>>>>>vmvm,mxv,mx>>>>>>>>>>>>>>>>>>>>",is_result_found)
            # Stop typing indicator
            await self.bot_handler.stop_typing_indicator()
            
            # Return whether result was found
            return is_result_found  # Invert because True means no answer found in bot system
            
        except Exception as e:
            logger.error(f"Error in send_bot_message: {e}")
            await self.bot_handler.stop_typing_indicator()
            return True  # Return True to indicate no answer found
