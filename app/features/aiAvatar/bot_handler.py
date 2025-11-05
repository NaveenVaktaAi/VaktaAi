import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import WebSocket

from app.features.aiAvatar.repository import AITutorRepository
from app.features.chat.utils.response import ResponseCreator
from app.features.aiAvatar.websocket_manager import AITutorWebSocketResponse
from app.features.aiAvatar.pdf_extractor import extract_text_from_pdf, summarize_pdf_content

logger = logging.getLogger(__name__)


class AITutorBotHandler:
    """
    AI Tutor bot message handler - Pure AI conversation without RAG/document search.
    Messages are kept in memory and saved to conversation at the end.
    """

    def __init__(
        self,
        ai_tutor_repository: AITutorRepository,
        websocket_response: AITutorWebSocketResponse,
        conversation_id: str,
        user_id: str,
        time_zone: str = "UTC",
        language_code: str = "en",
        selected_subject: str = "",  # ✅ Subject parameter
        selected_topic: str = ""  # ✅ Topic parameter
    ):
        self.ai_tutor_repository = ai_tutor_repository
        self.websocket_response = websocket_response
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.time_zone = time_zone
        self.language_code = language_code
        self.selected_subject = selected_subject  # ✅ Store subject
        self.selected_topic = selected_topic  # ✅ Store topic
        
        # In-memory messages list for this conversation session
        self.messages: List[Dict[str, Any]] = []

    async def handle_user_message(self, user_message: str, isAudio: bool = True, images: Optional[List[str]] = None, pdfs: Optional[List[str]] = None) -> bool:
        """
        Handle a user message with AI tutoring functionality.
        No RAG/document search - pure AI conversation.
        Messages are kept in memory, not saved to DB immediately.
        Supports image analysis via GPT-4 Vision API and PDF text extraction.
        
        Args:
            user_message: The user's message text
            isAudio: If True, generate audio + summary. If False, only text response.
            images: Optional list of image URLs for vision analysis
            pdfs: Optional list of PDF URLs for document analysis
        """
        try:
            print(f"[AI_TUTOR] Processing user message: '{user_message}' for conversation: {self.conversation_id}")
            if images:
                print(f"[AI_TUTOR] 🖼️ Images received: {len(images)} image(s)")
                print(f"[AI_TUTOR] 🖼️ Image URLs: {images}")
            else:
                print(f"[AI_TUTOR] 📝 No images in this message")
            
            if pdfs:
                print(f"[AI_TUTOR] 📄 PDFs received: {len(pdfs)} PDF(s)")
                print(f"[AI_TUTOR] 📄 PDF URLs: {pdfs}")
            else:
                print(f"[AI_TUTOR] 📝 No PDFs in this message")
            
            # Add user message to in-memory list
            user_message_dict = {
                "message": user_message,
                "is_bot": False,
                "type": "text",
                "reaction": None,
                "token": None,
                "is_edited": False,
                "created_ts": datetime.utcnow(),
                "updated_ts": datetime.utcnow()
            }
            self.messages.append(user_message_dict)
            print(f"[AI_TUTOR] User message added to in-memory list. Total messages: {len(self.messages)}")

            # Get chat history from in-memory messages
            chat_history_text = self.get_conversation_history_from_memory()
            
            # If images or PDFs are present, skip goforward decision and directly process with AI
            if (images and len(images) > 0) or (pdfs and len(pdfs) > 0):
                print(f"🖼️📄 Images or PDFs detected - Skipping goforward check, directly processing with AI...")
                # Skip goforward decision, directly use AI tutoring pipeline
                goforward = True
                processed_user_message = user_message
                language_code = self.language_code
                detected_language = "English"
            else:
                # Get goforward decision and response (only for text-only queries)
                raw = await self.get_goforward_decision_and_response(user_message, chat_history_text)
            
                # Parse the raw response to get goforward decision (only for text-only queries)
                try:
                    if not raw or raw.strip() == "":
                        print("Empty response from goforward decision, continuing with AI response")
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
                    goforward = parsed_response.get("goforward", True)
                    
                    if not goforward:
                        # Use the immediate response from decision function
                        immediate_response = parsed_response.get("response", "")
                        if immediate_response:
                            print(f"💬 Using immediate response (goforward=False): {immediate_response}")
                            
                            # Create natural streaming generator (not too slow now - real-time audio!)
                            async def string_to_generator(data):
                                chunk_size = 10  # Natural chunk size
                                delay = 0.015  # 15ms per chunk - smooth & natural
                                for i in range(0, len(data), chunk_size):
                                    chunk = data[i:i+chunk_size]
                                    yield chunk
                                    await asyncio.sleep(delay)
                            
                            generator = string_to_generator(immediate_response)
                            
                            if isAudio:
                                print(f"🎬 Starting immediate response streaming WITH audio...")
                            else:
                                print(f"📝 Starting immediate response streaming (text-only, no audio)...")
                            
                            # Send immediate response with conditional audio
                            final_response = await self.websocket_response.create_streaming_response_with_audio(
                                generator,
                                f"immediate_{datetime.utcnow().timestamp()}",
                                is_final=True,
                                language_code=self.language_code,
                                enable_tts=isAudio  # ✅ Use isAudio flag
                            )
                            
                            # Add bot response to in-memory list
                            bot_message_dict = {
                                "message": final_response,
                                "is_bot": True,
                                "type": "text",
                                "reaction": None,
                                "token": None,
                                "is_edited": False,
                                "created_ts": datetime.utcnow(),
                                "updated_ts": datetime.utcnow()
                            }
                            self.messages.append(bot_message_dict)
                            print(f"💾 Bot message added to in-memory list. Total messages: {len(self.messages)}")
                            
                            return True
                        else:
                            print("No immediate response provided, continuing with AI response")
                    
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                        print(f"Error parsing goforward response: {e}, continuing with AI response")
                        goforward = True
                        processed_user_message = user_message
                        language_code = self.language_code
                        detected_language = "English"
            
            # If goforward=True (or images present), send thinking indicator to frontend
            if goforward or (images and len(images) > 0):
                print(f"✨ Proceeding with AI tutoring (goforward=True) - Sending thinking indicator...")
                
                # Send thinking indicator with audio if isAudio=True
                if isAudio:
                    await self.websocket_response.send_thinking_indicator(
                        message="Hi Let me think about that for a moment...",  # Audio (teacher speaks)
                        display_message="Analyzing your question...",     # UI text
                        status="thinking",
                        language_code=self.language_code,
                        enable_audio=True
                    )
                else:
                    await self.websocket_response.send_thinking_indicator(
                        message="Analyzing your question...",
                        status="thinking",
                        language_code=self.language_code,
                        enable_audio=False
                    )
            
                # Now get language/translation for the user input
                print("📝 Getting language detection and translation...")
                multi_response_json = await self.get_multi_response_from_gpt_based_on_user_input(
                    user_message,
                    chat_history_text
                )
                
                # Process language detection response
                try:
                    processed_user_message = user_message  # Default
                    language_code = self.language_code  # Default
                    detected_language = "English"  # Default
                
                    if multi_response_json:
                        message_data = self.extract_json_from_text(multi_response_json)
                        if message_data:
                            processed_user_message = message_data.get("translated_input", user_message)
                            language_code = message_data.get("language_code", self.language_code)
                            detected_language = message_data.get("detected_language", "English")
                            self.language_code = language_code
                except Exception as lang_error:
                        print(f"Error in language detection: {lang_error}")
                processed_user_message = user_message
                language_code = self.language_code
                detected_language = "English"
            
            # Start web search in parallel ONLY if no images/PDFs
            web_search_task = None
            if not images and not pdfs:
                print("🚀 Starting parallel web search (no images/PDFs detected)...")
            
            async def fetch_web_results():
                try:
                    from langchain_community.tools.tavily_search import TavilySearchResults
                    TAVILY_API_KEY = "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"
                    tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
                    
                    import asyncio
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(None, tool.invoke, processed_user_message)
                    if results:
                        print("Web search results:", results[0].get('title', 'No title'))
                    return results
                except Exception as e:
                    print(f"Error fetching web search results: {e}")
                    return []
            
            # Start web search task immediately
            web_search_task = asyncio.create_task(fetch_web_results())
            
            # Send appropriate indicator based on content type
            if images or pdfs:
                # If images/PDFs present, show analyzing indicator
                if isAudio:
                    await self.websocket_response.send_thinking_indicator(
                        message="Let me analyze the content you've shared...",  # Audio
                        display_message="Analyzing images/documents...",         # UI text
                        status="analyzing",
                        language_code=self.language_code,
                        enable_audio=True
                    )
                else:
                    await self.websocket_response.send_thinking_indicator(
                        message="Analyzing images/documents...",
                        status="analyzing",
                        language_code=self.language_code,
                        enable_audio=False
                    )
            else:
                # Otherwise, show searching indicator for web search
                if isAudio:
                    await self.websocket_response.send_thinking_indicator(
                        message="Let me search for the most relevant information on this topic...",  # Audio
                        display_message="Searching for relevant information...",                   # UI text
                        status="searching",
                        language_code=self.language_code,
                        enable_audio=True
                    )
                else:
                    await self.websocket_response.send_thinking_indicator(
                        message="Searching for relevant information...",
                        status="searching",
                        language_code=self.language_code,
                        enable_audio=False
                    )
            
            # Process bot response with AI (no RAG)
            is_result_found = await self.process_ai_tutor_response(
                processed_user_message,
                language_code,
                detected_language,
                chat_history_text,
                web_search_task,
                isAudio=isAudio,  # ✅ Pass audio flag
                images=images,  # ✅ Pass images for vision analysis
                pdfs=pdfs  # ✅ Pass PDFs for document analysis
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
    ) -> str:
        """
        Ask the LLM to decide whether to proceed to AI response or handle immediately.
        Includes subject-aware greetings.
        """
        if not user_input or not user_input.strip():
            return json.dumps({
                "goforward": True,
                "response": None,
                "detected_language": "unknown",
                "language_code": "",
                "translated_input": ""
            })

        # Build STRONG subject + topic context for personalized greetings - AI is ALWAYS the teacher
        subject_greeting_context = ""
        if self.selected_subject:
            topic_part = f" on {self.selected_topic}" if self.selected_topic else ""
            example_greeting = f"Hello! I'm your {self.selected_subject} teacher{topic_part}. I have complete knowledge of {self.selected_subject}{f' and expertise in {self.selected_topic}' if self.selected_topic else ''}. How can I help you{' with {}'.format(self.selected_topic) if self.selected_topic else ' today'}?"
            
            subject_greeting_context = f"""
🎓 YOUR ROLE AS TEACHER:
**YOU ARE THE TEACHER OF {self.selected_subject.upper()}{f' WITH EXPERTISE IN {self.selected_topic.upper()}' if self.selected_topic else ''}.**
- You have COMPLETE knowledge of {self.selected_subject}{f' and specialized expertise in {self.selected_topic}' if self.selected_topic else ''}.
- REMEMBER: You are ALWAYS acting as the {self.selected_subject} teacher in every conversation.

📚 SUBJECT CONTEXT: The student has selected **{self.selected_subject}**{f' with focus on **{self.selected_topic}**' if self.selected_topic else ''}.

**GREETING RULE**: When user says greetings (hello, hi, namaste, etc.) or casual conversation:
- ALWAYS introduce yourself as their {self.selected_subject} teacher{topic_part}
- Emphasize that you have complete knowledge of {self.selected_subject}{f' and expertise in {self.selected_topic}' if self.selected_topic else ''}
- Example: "{example_greeting}"
- Keep it warm, encouraging, and specific to their learning focus
- Respond in the user's detected language
- Always maintain your identity as the {self.selected_subject} teacher
"""

        system_prompt = f"""
You are a classifier+assistant for an AI tutoring system. Inspect the chat_history and the new user message.
Decide whether the user's message should be handled immediately by a short assistant reply (auto-reply) OR escalated to the AI tutoring pipeline.
{subject_greeting_context}

REQUIREMENTS:
- Use user_input and chat_history_text to resolve references when needed.
- Detect language (English, Hindi, Hinglish, Marathi, Gujarati).
- Produce a concise English normalized query in 'translated_input'.

OUTPUT RULES (MUST FOLLOW EXACTLY):
Return ONLY ONE valid JSON object with these keys:

{{
"goforward": true|false,
"response": "<short reply or null>",
"detected_language": "<language name>",
"language_code": "<ISO 639-1 code>",
"translated_input": "<combined, standardized English query>"
}}

GUIDELINES:
- If the user's message is casual chit-chat, greetings, thanks, set goforward=false and provide a short subject-aware response.
- If the user asks an educational/informational question, set goforward=true and set response to null.
- Keep response short, warm, and helpful.
{f"- CRITICAL: For greetings: ALWAYS introduce as their {self.selected_subject} TEACHER{f' with expertise in {self.selected_topic}' if self.selected_topic else ''}. Emphasize that you have COMPLETE knowledge of {self.selected_subject}." if self.selected_subject else ""}
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
            print("raw>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", raw)
            return raw
        except Exception as e:
            print("[get_goforward_decision_and_response] LLM call failed:", e)
            return json.dumps({
                "goforward": True,
                "response": None,
                "detected_language": "unknown",
                "language_code": "",
                "translated_input": ""
            })

    async def get_multi_response_from_gpt_based_on_user_input(
        self, user_input: str, chat_history_text: str
    ) -> str:
        """Extract language, translate, and detect follow-up from user input using GPT"""

        prompt = [
            {
                "role": "system",
                "content": f"""You are an AI assistant for language detection and translation in an AI tutoring system.

REQUIREMENTS:
1) Use the provided user_input and chat_history to create a standardized English query.
2) Detect the language of the user_input. Supported: English, Hindi, Hinglish, Marathi, Gujarati.
3) Translate and rewrite into translated_input - one concise English query.

OUTPUT RULES:
Return ONLY valid JSON with exactly these three keys:
{{
  "detected_language": "<language name>",
  "translated_input": "<combined, standardized English query>",
  "language_code": "<ISO 639-1 code>"
}}

GUIDELINES:
- Keep translated_input complete and actionable.
- If user_input is independent, keep it as is; else use chat history if needed.
- If user asks for response in a particular language, include that note in translated_input.

USER_INPUT: {user_input}
CHAT_HISTORY_LAST_TWO: {chat_history_text}
"""
            }
        ]

        return await ResponseCreator().gpt_response_without_stream(prompt)

    async def process_ai_tutor_response(
        self, 
        user_message, 
        language_code, 
        detected_language, 
        chat_history_text,
        web_search_task=None,
        isAudio: bool = True,
        images: Optional[List[str]] = None,
        pdfs: Optional[List[str]] = None
    ):
        """
        Process AI tutor response with optional audio generation, image analysis, and PDF text extraction.
        
        Args:
            isAudio: If True, generate parallel text + summary + audio.
                     If False, only generate and stream text response.
            images: Optional list of image URLs for vision analysis.
            pdfs: Optional list of PDF URLs for document analysis.
        
        Flow:
        - isAudio=True: Parallel text + summary → Stream text + Generate audio
        - isAudio=False: Only text response → Stream text only
        """
        try:
            print("user_message>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", user_message)
            
            # Get web search results if task is provided
            web_results = []
            if web_search_task:
                try:
                    web_results = await web_search_task
                    print(f"✅ Web search completed with {len(web_results)} results")
                except Exception as e:
                    print(f"❌ Web search failed: {e}")
                    web_results = []
            
            # Extract text from PDFs if present
            pdf_texts = []
            if pdfs and len(pdfs) > 0:
                print(f"📄 Extracting text from {len(pdfs)} PDF(s)...")
                for pdf_url in pdfs:
                    # Convert relative URL to absolute path
                    if pdf_url.startswith('/static/'):
                        pdf_path = f".{pdf_url}"
                        print(f"📄 Processing PDF: {pdf_path}")
                        extracted_text = extract_text_from_pdf(pdf_path)
                        if extracted_text:
                            pdf_texts.append(extracted_text)
                            summary = summarize_pdf_content(extracted_text, 200)
                            print(f"✅ PDF text extracted: {summary}")
                        else:
                            print(f"⚠️ Failed to extract text from PDF: {pdf_url}")
                
                if pdf_texts:
                    print(f"✅ Successfully extracted text from {len(pdf_texts)} PDF(s)")
                else:
                    print(f"⚠️ No text extracted from PDFs")
            
            gpt_start = asyncio.get_event_loop().time()
            
            # Initialize final_response to None
            final_response = None
            
            if isAudio:
                # 🔥 AUDIO MODE: Generate text + summary
                print("🔗 AUDIO MODE: Starting text + summary generation with context...")
                
                # Send generating indicator with audio
                await self.websocket_response.send_thinking_indicator(
                    message="Now, let me prepare a detailed explanation for you...",  # Audio
                    display_message="✨ Generating detailed response and audio...",    # UI text
                    status="generating",
                    language_code=language_code,
                    enable_audio=True  # ✅ Audio mode - speak the status
                )
                
                # If images/PDFs present, use vision-capable GPT (streaming)
                if images or pdfs:
                    print("🖼️📄 Using VISION-CAPABLE GPT for images/PDFs in audio mode...")
                    
                    # Get streaming response with vision support
                    gpt_resp_generator = self.get_ai_tutor_response_from_gpt(
                        user_message, 
                        detected_language,
                        chat_history_text,
                        web_results,
                        images=images,
                        pdf_texts=pdf_texts
                    )
                    
                    # Collect full response for summary generation
                    text_response = ""
                    async for chunk in gpt_resp_generator:
                        text_response += chunk
                    
                    print(f"✅ Vision-based response complete: {len(text_response)} chars")
                    
                    # Generate summary from full response
                    summary_for_audio = await self._generate_teacher_summary(
                        user_message, text_response, language_code, chat_history_text, web_results
                    )
                else:
                    # No images/PDFs - use parallel generation for speed
                    response_creator = ResponseCreator()
                    parallel_result = await response_creator.generate_parallel_response_and_summary(
                        user_query=user_message,
                        language_code=language_code,
                        chat_history=chat_history_text,
                        web_results=web_results,
                        selected_subject=self.selected_subject,  # ✅ Pass subject context
                        selected_topic=self.selected_topic  # ✅ Pass topic context
                    )
                    
                    text_response = parallel_result["text"]
                    summary_for_audio = parallel_result["summary"]
                    
                    print(f"✅ Parallel generation complete!")
                    print(f"📝 Text response length: {len(text_response)}")
                    print(f"🎤 Audio summary: {summary_for_audio[:100]}...")
                
                # Debug info (outside if/else)
                if web_results:
                    print(f"🌐 Web context included: {len(web_results)} results")
                if chat_history_text:
                    print(f"💬 Chat history context included: {len(chat_history_text)} chars")
                
                # Validate that text_response and summary_for_audio are set and not empty
                if not text_response or not text_response.strip():
                    print("⚠️ ERROR: text_response is empty or invalid!")
                    raise ValueError("Failed to generate text response")
                
                if not summary_for_audio or not summary_for_audio.strip():
                    print("⚠️ ERROR: summary_for_audio is empty or invalid!")
                    raise ValueError("Failed to generate audio summary")
                
                # ✅ Use faster streaming approach (like DocSathi) - minimal delay for immediate start
                async def fast_text_generator(text):
                    """Fast streaming generator with minimal delay (similar to DocSathi)"""
                    words = text.split()
                    for word in words:
                        # Add space after word except last
                        yield word + (" " if word != words[-1] else "")
                        # Minimal delay for smooth streaming (same as DocSathi: 10ms)
                        await asyncio.sleep(0.01)
                
                text_generator = fast_text_generator(text_response)
                
                # 🔥 TRULY PARALLEL: Start text streaming AND audio generation simultaneously
                print("📡 Starting PARALLEL: Fast text streaming + Audio generation...")
                
                # Start audio generation in background (don't await)
                audio_task = asyncio.create_task(
                    self.websocket_response.generate_and_send_audio_chunks(
                        summary_for_audio,
                        f"summary_audio_{datetime.utcnow().timestamp()}",
                        language_code=language_code
                    )
                )
                print("🎤 Audio generation started in background...")
                
                # Stream text to frontend using FAST streaming (no audio delay overhead)
                print("📝 Fast text streaming started...")
                final_response = await self.websocket_response.create_streaming_response(
                    text_generator,
                    f"ai_tutor_response_{datetime.utcnow().timestamp()}",
                    is_final=True
                )
                
                # Wait for audio generation to complete
                await audio_task
                print("🎤 Audio generation completed!")
                
                gpt_end = asyncio.get_event_loop().time()
                print(f"✅ Complete in {gpt_end - gpt_start:.3f}s (parallel text + summary + audio)")
                
            else:
                # 📝 TEXT-ONLY MODE: No audio, no summary - just stream text response with FAST streaming
                print("📝 TEXT-ONLY MODE: Generating text response (no audio, no summary)...")
                
                # Send generating indicator for text-only mode (no audio)
                await self.websocket_response.send_thinking_indicator(
                    message="Generating response...",
                    display_message="✨ Generating response...",
                    status="generating",
                    language_code=language_code,
                    enable_audio=False  # ❌ Text mode - no audio
                )
                
                # Get the generator for AI tutor response (original method)
                gpt_resp_generator = self.get_ai_tutor_response_from_gpt(
                    user_message, 
                    detected_language,
                    chat_history_text,
                    web_results,
                    images=images,  # Pass images for vision analysis
                    pdf_texts=pdf_texts  # Pass PDF text content
                )
                
                print("📡 Streaming text-only response to frontend (FAST streaming)...")
                # ✅ Use FAST streaming (same as DocSathi) - no audio overhead, minimal delay
                final_response = await self.websocket_response.create_streaming_response(
                    gpt_resp_generator,
                    f"ai_tutor_response_{datetime.utcnow().timestamp()}",
                    is_final=True
                )
                
                gpt_end = asyncio.get_event_loop().time()
                print(f"✅ Complete in {gpt_end - gpt_start:.3f}s (text-only, no audio)")
            
            # Validate final_response is set
            if not final_response:
                print("⚠️ ERROR: final_response not set! Using fallback message.")
                final_response = "I apologize, but I encountered an error generating the response. Please try again."
            
            # Add bot response to in-memory list    
            bot_message_dict = {
                "message": final_response,
                "is_bot": True,
                "type": "text",
                "reaction": None,
                "token": None,
                "is_edited": False,
                "created_ts": datetime.utcnow(),
                "updated_ts": datetime.utcnow()
            }
            self.messages.append(bot_message_dict)
            print(f"💾 Bot message added to in-memory list. Total messages: {len(self.messages)}")

            return True

        except Exception as e:
            print(f"Error in process_ai_tutor_response: {e}\n")
            return False

    async def get_ai_tutor_response_from_gpt(
        self, 
        user_query: str, 
        detected_language: str,
        chat_history: str,
        web_results: list,
        images: Optional[List[str]] = None,
        pdf_texts: Optional[List[str]] = None
    ):
        """
        Get AI tutor GPT response as an async generator.
        Yields tokens as they arrive for real-time streaming.
        No RAG - pure AI tutoring with web search context.
        Supports image analysis via GPT-4 Vision API and PDF text extraction.
        
        Args:
            user_query: The user's question
            detected_language: Language detected from user input
            chat_history: Previous conversation context
            web_results: Web search results for additional context
            images: Optional list of image URLs for vision analysis
            pdf_texts: Optional list of extracted PDF text content
        """
        try:
            # Validate user query
            if not isinstance(user_query, str) or not user_query.strip():
                print("⚠️ Warning: Invalid user query.")
                user_query = "Please provide a general educational response."
 
            print("🔍 Processing AI tutor query:", user_query)    
            
            # Build PDF context section if PDFs are provided
            pdf_context = ""
            if pdf_texts and len(pdf_texts) > 0:
                print(f"📄 Including {len(pdf_texts)} PDF document(s) in context")
                pdf_context = "\n\n### Document Content (from uploaded PDFs):\n"
                for idx, pdf_text in enumerate(pdf_texts):
                    pdf_context += f"\n--- Document {idx + 1} ---\n{pdf_text}\n"
                pdf_context += "\nUse the above document content to answer the user's question accurately."
            
            # Build STRONG subject + topic context - AI is ALWAYS the teacher of selected subject
            subject_context = ""
            teacher_role = ""
            if self.selected_subject:
                topic_part = f", specifically on **{self.selected_topic}**" if self.selected_topic else ""
                subject_context = f"""
### 🎓 YOUR ROLE AS TEACHER:
**YOU ARE THE TEACHER OF {self.selected_subject.upper()}{topic_part.upper().replace('**', '') if self.selected_topic else ''}.**
- You have deep expertise and knowledge in **{self.selected_subject}**.
{f'- You have specialized knowledge in **{self.selected_topic}** and are an expert in this topic.' if self.selected_topic else ''}
- REMEMBER: Every conversation you have, you are acting as this {self.selected_subject} teacher.
- You have COMPLETE knowledge of {self.selected_subject}{f' and especially {self.selected_topic}' if self.selected_topic else ''}.
- Approach EVERY question from your perspective as the {self.selected_subject} teacher.
- Your identity and expertise as a {self.selected_subject} teacher should be reflected in your responses.
{f'- When explaining {self.selected_topic}, draw from your deep understanding as a {self.selected_topic} expert.' if self.selected_topic else ''}

### 📚 Selected Subject{f' - {self.selected_topic}' if self.selected_topic else ''}:
The student has chosen to study **{self.selected_subject}**{f' with focus on **{self.selected_topic}**' if self.selected_topic else ''} in this session.
You are their dedicated {self.selected_subject} teacher{f' with expertise in {self.selected_topic}' if self.selected_topic else ''}.
"""
            
            # Construct the prompt for AI tutoring with STRONG subject awareness
            prompt_text = f"""
{subject_context if subject_context else "You are an intelligent AI tutor that helps students learn and understand concepts."}

### User Query:
{user_query}

### Chat History:
{chat_history if chat_history else "No previous conversation"}

### Web Search Results (for additional context):
{web_results[0].get('content', 'No additional web context available') if web_results else "No additional web context available"}
{pdf_context}

### Instructions:
- Respond in the user's language: {detected_language}.
- Act as a patient, knowledgeable {f'{self.selected_subject} teacher' if self.selected_subject else 'tutor'} who explains concepts clearly.
{f'- Always remember: You are the {self.selected_subject} teacher. You have complete knowledge of {self.selected_subject}{f" and {self.selected_topic}" if self.selected_topic else ""}.' if self.selected_subject else ''}
- If web search results or PDF documents are available, use them for accurate context.
- Break down complex topics into simple, understandable parts.
- Use examples and analogies when helpful.
- Encourage learning by asking follow-up questions when appropriate.
- If you don't know something, be honest and guide the student to learn more.
- Keep your response educational, accurate, and engaging.

### 🎯 Subject Focus Strategy:
{f'''- CRITICAL: You are the teacher of **{self.selected_subject}**{f' with expertise in **{self.selected_topic}**' if self.selected_topic else ''}.
- You have COMPLETE knowledge of this subject and topic.
- If the question is about {self.selected_topic or self.selected_subject}, answer thoroughly using your expertise as a {self.selected_subject} teacher.
- If the question is OFF-TOPIC:
  * Still answer it completely and helpfully
  * At the end, GENTLY guide them back: "Since you're focusing on {self.selected_topic or self.selected_subject} today, would you also like to explore [related concept in {self.selected_topic or self.selected_subject}]?"
  * Make the redirection natural, encouraging, and optional - never forceful
- Whenever relevant, connect concepts back to {self.selected_topic or self.selected_subject} from your perspective as their teacher.
- Always maintain your identity as the {self.selected_subject} teacher throughout the conversation.''' if self.selected_subject else '- No specific subject focus. Answer all questions comprehensively without redirection.'}
"""
 
            # Create proper message format for OpenAI API with STRONG system message
            system_message_content = f"You are a helpful {f'{self.selected_subject} teacher' if self.selected_subject else 'AI tutor'} that teaches students with patience and clarity. You adapt to the student's level and explain concepts in an engaging way."
            if self.selected_subject:
                system_message_content += f" IMPORTANT: You are the teacher of {self.selected_subject}{f' with expertise in {self.selected_topic}' if self.selected_topic else ''}. You have complete knowledge of this subject and should always approach questions from this perspective."
            
            messages = [
                {
                    "role": "system",
                    "content": system_message_content
                }
            ]
            
            # Add user message with images if provided
            if images and len(images) > 0:
                print(f"🖼️ Processing {len(images)} image(s) for vision analysis")
                print(f"🖼️ Raw image URLs received: {images}")
                # Build image URLs (convert relative paths to full URLs if needed)
                image_contents = []
                for img_url in images:
                    if img_url.startswith('/static/'):
                        # Convert relative URL to full path - use file:// or actual server URL
                        import os
                        # Get absolute path for the image
                        abs_path = os.path.abspath(f".{img_url}")
                        print(f"🖼️ Converting {img_url} to absolute path: {abs_path}")
                        
                        # For OpenAI Vision API, we need to use data URLs or accessible HTTP URLs
                        # Let's encode the image as base64
                        try:
                            import base64
                            with open(abs_path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                # Detect image type
                                ext = abs_path.split('.')[-1].lower()
                                mime_type = f"image/{ext}" if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp'] else "image/jpeg"
                                data_url = f"data:{mime_type};base64,{img_data}"
                                image_contents.append({"type": "image_url", "image_url": {"url": data_url}})
                                print(f"✅ Image encoded as base64 data URL (length: {len(data_url)})")
                        except Exception as e:
                            print(f"❌ Error encoding image {img_url}: {e}")
                            # Fallback to HTTP URL
                            full_url = f"http://localhost:5000{img_url}"
                            image_contents.append({"type": "image_url", "image_url": {"url": full_url}})
                    else:
                        print(f"🖼️ Using direct URL: {img_url}")
                        image_contents.append({"type": "image_url", "image_url": {"url": img_url}})
                
                if not image_contents:
                    print("⚠️ No valid images to send to Vision API, falling back to text-only")
                    messages.append({
                        "role": "user",
                        "content": prompt_text
                    })
                else:
                    # Add text and images to user message
                    user_content = [{"type": "text", "text": prompt_text}] + image_contents
                    messages.append({
                        "role": "user",
                        "content": user_content
                    })
                    print(f"✅ Vision message created with text and {len(image_contents)} images")
            else:
                # Text-only message
                print("📝 No images provided, using text-only mode")
                messages.append({
                    "role": "user",
                    "content": prompt_text
                })
 
            # Stream response from GPT
            print("🤖 Starting AI tutor GPT streaming...")
            token_count = 0
            
            try:
                async for token in ResponseCreator().get_streaming_response(messages):
                    if token:
                        token_count += 1
                        if token_count % 50 == 0:
                            print(f"📝 Streamed {token_count} tokens...")
                        yield token
                
                if token_count == 0:
                    print("⚠️ Warning: Empty response from GPT API.")
                    fallback_msg = "I couldn't generate a proper response. Please try again."
                    for i in range(0, len(fallback_msg), 10):
                        yield fallback_msg[i:i+10]
                        await asyncio.sleep(0.01)
                else:
                    print(f"✅ Streaming completed successfully with {token_count} tokens")
 
            except asyncio.TimeoutError:
                print("🚨 GPT API Timeout")
                error_msg = "The response took too long. Please try again."
                for i in range(0, len(error_msg), 10):
                    yield error_msg[i:i+10]
                    await asyncio.sleep(0.01)
                    
            except Exception as api_error:
                print(f"🚨 GPT API Error during streaming: {api_error}")
                error_msg = "I encountered an error. Please try again."
                for i in range(0, len(error_msg), 10):
                    yield error_msg[i:i+10]
                    await asyncio.sleep(0.01)
 
        except Exception as e:
            print(f"🚨 Unexpected error in get_ai_tutor_response_from_gpt: {e}")
            import traceback
            traceback.print_exc()
            error_msg = "An unexpected error occurred. Please try again later."
            for i in range(0, len(error_msg), 10):
                yield error_msg[i:i+10]
                await asyncio.sleep(0.01)

    def extract_json_from_text(self, text: str):
        """Extract complete JSON data from the given text."""
        pattern = r"\{.*\}"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            json_text = match.group(0)
            try:
                json_data = json.loads(json_text)
                return json_data
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error: {e}")
                return None
        return None

    def get_conversation_history_from_memory(self) -> str:
        """Get last 8 conversation messages from in-memory list for better context"""
        try:
            print("Getting conversation history from in-memory messages...")
            
            if not self.messages:
                return ""
            
            # Get last 8 messages for better context and continuity
            last_messages = self.messages[-8:] if len(self.messages) > 8 else self.messages
            
            chat_history = []
            for msg in last_messages:
                sender = "User" if not msg.get("is_bot", False) else "AI"
                chat_history.append(f"{sender}: {msg.get('message', '')}")

            return "\n".join(chat_history)
        except Exception as e:
            logger.error(f"Error getting conversation history from memory: {e}")
            return ""

    async def send_error_response(self, error_message: str):
        """Send error response to user"""
        try:
            error_dict = {
                "message": f"I'm having trouble right now: {error_message}. Please try again.",
                "is_bot": True,
                "type": "error",
                "reaction": None,
                "token": None,
                "is_edited": False,
                "created_ts": datetime.utcnow(),
                "updated_ts": datetime.utcnow()
            }
            self.messages.append(error_dict)
        except Exception as e:
            logger.error(f"Error sending error response: {e}")

    async def send_typing_indicator(self):
        """Send typing indicator to show bot is thinking"""
        await self.websocket_response.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "typing_indicator",
                "conversationId": self.conversation_id,
                "isBot": True,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def stop_typing_indicator(self):
        """Stop typing indicator"""
        await self.websocket_response.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "stop_typing_indicator",
                "conversationId": self.conversation_id,
                "isBot": True,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    
    async def _generate_teacher_summary(
        self, 
        user_query: str, 
        full_response: str, 
        language_code: str,
        chat_history: str = "",
        web_results: list = None
    ) -> str:
        """
        Generate a teacher-style audio summary from the full response.
        Used when images/PDFs are present in audio mode.
        
        Args:
            user_query: The user's original question
            full_response: The complete AI response text
            language_code: Language code for response
            chat_history: Previous conversation context
            web_results: Web search results
            
        Returns:
            A brief, encouraging teacher-style summary for audio
        """
        try:
            from app.avatar_config import ai_settings
            
            # Build context
            history_context = ""
            if chat_history:
                history_context = f"\n\nConversation History:\n{chat_history}\n"
            
            web_context = ""
            if web_results and len(web_results) > 0:
                web_context = "\n\nWeb Context Available: Yes"
            
            # Create prompt for summary
            summary_prompt = f"""You are a warm, encouraging female teacher.

Given this detailed response to a student's question, create a brief conversational summary:

Student Question: {user_query}

Full Response:
{full_response}
{history_context}{web_context}

Create a brief (2-4 sentences), encouraging teacher-style audio summary:
- Use phrases like "Great question!", "Let me help you understand"
- Be warm and supportive
- Capture the KEY POINT
- Make it natural for speech/audio
- Keep it concise

Respond in the same language as the full response."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful teacher creating brief audio summaries."
                },
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ]
            
            # Get non-streaming response for summary
            response_creator = ResponseCreator()
            summary = await response_creator.gpt_response_without_stream(messages)
            
            print(f"✅ Generated teacher summary: {summary[:100]}...")
            return summary.strip()
            
        except Exception as e:
            print(f"[SUMMARY] Error generating teacher summary: {e}")
            # Fallback summary
            return "Let me help you understand this concept better."
    
    async def save_messages_to_db(self):
        """Save all in-memory messages to database (call this when conversation ends)"""
        try:
            if self.messages:
                await self.ai_tutor_repository.save_messages_to_conversation(
                    self.conversation_id, 
                    self.messages
                )
                print(f"💾 Saved {len(self.messages)} messages to conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"Error saving messages to DB: {e}")
            print(f"Error saving messages to DB: {e}")


class AITutorBotMessage:
    """
    Main AI Tutor bot message handler class.
    Pure AI tutoring without RAG/document search.
    """
    
    def __init__(
        self,
        conversation_id: str,
        user_id: str,
        websocket_response: AITutorWebSocketResponse = None,
        timezone: str = "UTC",
        language_code: str = "en",
        selected_subject: str = "",  # ✅ Subject parameter
        selected_topic: str = ""  # ✅ Topic parameter
    ):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.websocket_response = websocket_response
        self.timezone = timezone
        self.language_code = language_code
        self.selected_subject = selected_subject  # ✅ Store subject
        self.selected_topic = selected_topic  # ✅ Store topic
        
        # Initialize AI tutor repository
        self.ai_tutor_repository = AITutorRepository()
        
        # Initialize AI tutor bot handler with subject + topic context
        self.bot_handler = AITutorBotHandler(
            self.ai_tutor_repository,
            self.websocket_response,
            self.conversation_id,
            self.user_id,
            self.timezone,
            self.language_code,
            selected_subject,  # ✅ Pass subject to handler
            selected_topic  # ✅ Pass topic to handler
        )

    async def send_bot_message(self, user_message: str, isAudio: bool = True, images: Optional[List[str]] = None, pdfs: Optional[List[str]] = None) -> bool:
        """
        Send AI tutor bot message in response to user message.
        
        Args:
            user_message: The user's message text
            isAudio: If True, generate audio + summary. If False, text only.
            images: Optional list of image URLs for vision analysis.
            pdfs: Optional list of PDF URLs for document analysis.
        
        Returns True if successful, False if no answer found.
        """
        try:
            # Show typing indicator
            await self.bot_handler.send_typing_indicator()
            
            # Handle the user message with AI tutoring (pass isAudio flag, images, and pdfs)
            is_result_found = await self.bot_handler.handle_user_message(user_message, isAudio=isAudio, images=images, pdfs=pdfs)
            print("is_result_found>>>>>>>>>AI_TUTOR>>>>>>>>>>>>>>>>>>>>", is_result_found)
            
            # Stop typing indicator
            await self.bot_handler.stop_typing_indicator()
            
            return is_result_found
            
        except Exception as e:
            logger.error(f"Error in send_bot_message: {e}")
            await self.bot_handler.stop_typing_indicator()
            return True
    
    async def save_conversation_messages(self):
        """Save all messages from current session to database"""
        await self.bot_handler.save_messages_to_db()
