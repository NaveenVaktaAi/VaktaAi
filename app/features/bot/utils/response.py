from collections.abc import AsyncGenerator
import json
from typing import Any

import openai
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from requests import Session
from sentry_sdk import capture_exception
from app.database import get_db
from app.models.chat_message import ChatMessage
from app.utils.openai.openai import start_openai
from app.features.aws.secretKey import get_secret_keys
import asyncio
import openai
from typing import AsyncGenerator, Any, List
import tiktoken
from fastapi import WebSocket
import ast

keys = get_secret_keys()


class ResponseCreator:
    def __init__(self):
        self.client = start_openai()
        self.default_model = "gpt-4-0125-preview"
        self.fallback_model = "gpt-3.5-turbo"
        self.final_model = "gpt-3.5-turbo-instruct"
        self.max_tokens_gpt4 = 4096
        self.max_tokens_gpt3 = 4096
        self.max_tokens_final = 2049
        self.retry_attempts = 3  # Number of retries for each model
        self.openai_key = keys.get("OPENAI_API_KEY")

    async def get_gpt_response(
        self,
        last_user_text: str,
        language_code: str | None = None,
        chat_id: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Gets chat-based GPT response"""
        return await self.generate_response(
            last_user_text,
            language_code,
            chat_id
        )

    # Function to count tokens in a string
    def count_tokens(self, text: str, model: str) -> int:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

    async def generate_response(
        self,
        user_query,
        language_code: str | None,
        
        chat_id: int | None,
    ) -> AsyncGenerator[str, Any]:
        # Models with their context windows and max token output
        
        models = [
            ("gpt-3.5-turbo", 16385, 2730),
            ("gpt-3.5-turbo-0125", 16385, 2730),
            ("gpt-4o-mini", 128000, 4096),
            ("gpt-4o-2024-05-13", 128000, 4096),
            ("gpt-4-turbo", 128000, 2730),
            ("gpt-4", 8192, 2730),
        ]
        main_response = (
            "I'm having trouble right now. You can try again."
        )
        response = main_response

        # Function to split messages into manageable chunks
        def split_messages(example_res, context_window, model, max_output_tokens):
            if example_res:
                example_res_text = " ".join(example_res)
                example_tokens = self.count_tokens(example_res_text, model)
                total_token_limit = context_window - max_output_tokens

            # If the example fits within the context window, return as a single chunk
                if example_tokens <= total_token_limit:
                    return [example_res_text]

                # Calculate the number of segments needed
                num_segments = -(-example_tokens // total_token_limit)  # Ceiling division
                segment_length = len(example_res_text) // num_segments

                # Split the text into chunks of approximately equal size
                return [
                    example_res_text[i : i + segment_length]
                    for i in range(0, len(example_res_text), segment_length)
                ]
            
            return []


        async def handle_response():
            nonlocal main_response
            try:
                # Change time here
                await asyncio.wait_for(_handle_response_inner(), timeout=30)
            except asyncio.TimeoutError:
                print("Loop exceeded 30 seconds.")
            except Exception as e:
                print(f"An error occurred: {e}")

            return main_response

        async def _handle_response_inner():
            nonlocal response
            nonlocal main_response

            for modelName, context_window, max_output_tokens in models:
                if example_response is not None:
                    total_tokens = self.count_tokens(" ".join(example_response), modelName)
                    if total_tokens > context_window - max_output_tokens :
                        example_response_chunked = split_messages(
                            example_response, context_window, modelName, max_output_tokens
                        )
                else:
                    example_response_chunked = example_response
                    
                print("example_response_chunked>>>>>>>>>>>>>>>>>>>>>",example_response_chunked)

                try:
                    chat = ChatOpenAI(
                        model=modelName,
                        temperature=0.3,
                        max_tokens=max_output_tokens,
                        api_key=self.openai_key,
                    )
                    chat_history = ChatMessageHistory()

                    db = next(get_db())
                    results = (
                        db.query(ChatMessage)
                        .filter(ChatMessage.chat_id == chat_id)
                        .order_by(ChatMessage.id.desc())
                        .limit(2)
                        .all()
                    )

                    # Format chat history
                    chat_history_text = "\n".join(
                        [f"{'User' if not msg.is_bot else 'AI'}: {msg.message}" for msg in results]
                    )

                    chat_history_text += f"\nUser: {user_query}\n"
                    chat_history.add_user_message(user_query)
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            f"""
                            You are Agent AI, an expert assistant providing clear, detailed, and structured answers.
                            Provide your answer solely based on the user query.
                            Maintain politeness, accuracy, and clarity.
                            user query: {user_query}
                            
                            **Guidelines:**
                            
                            - **Answer Focus:** Your answer must be exclusively based on the user query.
                            - **Handling Specific Follow-Ups:** Maintain deep context throughout.
                            - **Response Format:** Avoid repetition; provide structured responses only if requested.
                            - **Conversation-Ending Statements:** Gracefully close when the user indicates they are done.
                            - **Data-Driven Responses:** Use provided references when applicable.
                            - **Response Structure & Formatting:** Use paragraphs by default.
                            - **Language Consistency:** Respond in the same language as input ({language_code}).
                            
                            {f'Responses should align with the {industry_type} industry.' if not is_outside_from_industry else 'Responses are not restricted to any industry.'}  
                            """
                        ]
                    )


                    print("================INSIDE PROMPT===============")
                    MessagesPlaceholder(variable_name="messages")
                    chain = prompt | chat
                    completion = chain.astream({"messages": chat_history.messages})
                    response = await process_completion(completion)
                    if (
                        response
                        != "I'm having trouble right now. You can try again."
                        and response
                        != "Resources are busy. Please try again later."
                    ):
                        main_response = response

                except asyncio.TimeoutError:
                    print("API call timed out.")
                    break  # Stop trying further models on timeout

                except openai.RateLimitError as e:
                    print("API Rate limit error", e)
                    if (
                        response
                        == "I'm having trouble right now. You can try again."
                    ):
                        response = "Resources are busy. Please try again later."
                    continue

                except openai.OpenAIError:
                    continue

                if (
                    response
                    != "I'm having trouble right now. You can try again."
                    and response != "Resources are busy. Please try again later."
                ):
                    main_response = response
                    break

            return main_response



        async def process_completion(completion) -> str:
            output_text = ""
            if completion is None:
                return "Resources are busy. Please try again later."
            async for chunk in completion:
                content = chunk.content or ""
                output_text += content

            return output_text

        await handle_response()

        async def text_generator() -> AsyncGenerator[str, Any]:
            yield main_response

        return text_generator()

    def user_input_keyword_generator(
        self, *, user_input: str, industry_type: str
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "[prose] [Output only Array] "
                    "You are an informative expert bot and your task is to generate meaningful keywords"
                    f"based on the user's input related to the {industry_type} industry and do not go outside the scope of industry type"
                    "You should generate keywords that are relevant to the user's input. "
                    "Provide me an array of maximum 5 keywords but you can adjust according to the user input or message,"
                    "and if not able to find, then provide me an empty array. "
                    "It is a must to provide the keywords in the input itself and then check for other relevant keywords."
                    f"Here is the current user input: {user_input}."
                    "Avoid whitespace, line breaks, or indentation between fields. No codefence block."
                ),
            },
        ]
        return messages

    async def filter_user_input(
        self, user_input: str, industry_type: str
    ) -> AsyncGenerator[str, None]:
        """Gets chat-based GPT response"""
        prompt = self.user_input_keyword_generator(
            user_input=user_input, industry_type=industry_type
        )

        return await self.gpt_response_without_stream(prompt)
    
    async def gpt_response_with_stream_2(self, chat_messages) -> str:
        model = "gpt-4"
        chat_completion_kwargs = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": 8192,
        }

        text = ""
        for _ in range(5):
            try:
                chat_completion_kwargs.update(
                    {
                        "model": "gpt-4o-2024-05-13",
                        "max_tokens": 4096,
                        "stream": True,  # Enable streaming
                    }
                )

                # Start the completion request
                response = await self.client.chat.completions.create(**chat_completion_kwargs)

                # Initialize a variable to store the full response text
                response_text = ""
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        response_text += content

                # Clean up the response text to ensure it is only valid questions
                response_text = response_text.replace('["', '').replace('"]', '').replace('",', '').strip()

                # Now split the response into questions
                questions = [question.strip() for question in response_text.split('?') if question.strip()]
                return questions  # Return the list of questions

            except openai.OpenAIError:
                # Fallback code, retry with other models
                pass


    async def gpt_response_with_stream(self, chat_messages) -> str:
        model = "gpt-4"
        chat_completion_kwargs = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": 8192,
        }

        text = ""
        for _ in range(5):
            try:
                chat_completion_kwargs.update(
                    {
                        "model": "gpt-4o-2024-05-13",
                        "max_tokens": 4096,
                        "stream": True,  # Enable streaming
                    }
                )

                # Start the completion request
                response = await self.client.chat.completions.create(
                    **chat_completion_kwargs
                )
                # Initialize a variable to store the full response text
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                        # print(content)

            except openai.OpenAIError:
                chat_completion_kwargs.update(
                    {
                        "model": "gpt-3.5-turbo",
                        "max_tokens": 4096,
                        "stream": True,  # Enable streaming
                    }
                )
                try:
                    completion = await self.client.chat.completions.create(
                        **chat_completion_kwargs
                    )
                    async for chunk in completion:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                except openai.OpenAIError:
                    engine = "gpt-3.5-turbo-instruct"
                    try:
                        completion = await self.client.chat.completions.create(
                            model=engine,
                            messages=chat_messages,
                            temperature=0.1,
                            max_tokens=2049,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0.6,
                            stop=["Bot:"],
                            stream=True,  # Enable streaming
                        )
                    except openai.OpenAIError:
                        continue
                    else:
                        async for chunk in completion:
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content
                        break
                else:
                    break
            else:
                break


    # async def gpt_response_without_stream(self, chat_messages) -> str:
    #     models = [
    #         {"model": "gpt-3.5-turbo", "max_tokens": 4096},  # Prioritize faster and cheaper models
    #         {"model": "gpt-4", "max_tokens": 8192},          # Fallback to GPT-4
    #         {"model": "gpt-4-0125-preview", "max_tokens": 4096},  # Experimental model
    #     ]
    #     fallback_model = {
    #         "model": "gpt-3.5-turbo-instruct",
    #         "temperature": 0.1,
    #         "max_tokens": 2049,
    #         "top_p": 1,
    #         "frequency_penalty": 0,
    #         "presence_penalty": 0.6,
    #         "stop": ["Bot:"],
    #     }

    #     retries = 5
    #     text = ""
    #     for attempt in range(retries):
    #         for model_config in models:
    #             try:
    #                 completion = await self.client.chat.completions.create(
    #                     messages=chat_messages,
    #                     **model_config
    #                 )
    #                 text = completion.choices[0].message.content
    #                 return text  # Successfully received a response, exit immediately
    #             except openai.OpenAIError as e:
    #                 print(f"Error with model {model_config['model']}: {e}")
    #                 await asyncio.sleep(1)  # Short delay before retrying

    #         # Final fallback to `gpt-3.5-turbo-instruct`
    #         try:
    #             completion = await self.client.chat.completions.create(
    #                 messages=chat_messages,
    #                 **fallback_model
    #             )
    #             text = completion.choices[0].message.content
    #             return text
    #         except openai.OpenAIError as e:
    #             print(f"Error with fallback model: {e}")
    #             await asyncio.sleep(1)  # Delay before retrying

    #     # If all retries fail
    #     print("All attempts to fetch GPT response have failed.")
    #     return text

    async def gpt_response_without_stream(self, chat_messages) -> str:
        models = [
            # {"model": "gpt-4o-mini", "max_tokens": 4096},  # Primary model for accuracy
            {"model": "gpt-3.5-turbo", "max_tokens": 4096}, 
            # {"modal":"gpt-4-32k","max_tokens": 10000}
        ]

        fallback_model = {
            "model": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0.6,
            "stop": None,  # Adjust if needed
        }

        retries = 3  # Maximum retry attempts
        for attempt in range(retries):
            delay = 0.5 * (2 ** attempt)  # Exponential backoff: 0.5s, 1s, 2s

            for model_config in models:
                try:
                    completion = await self.client.chat.completions.create(
                        messages=chat_messages,
                        **model_config
                    )
                    return completion.choices[0].message.content
                except openai.OpenAIError as e:
                    print(f"[Attempt {attempt+1}] Model {model_config['model']} failed: {e}")
                    await asyncio.sleep(delay)  # Wait before retrying

        # Fallback to last-resort GPT-4o-mini if all attempts fail
        try:
            print("All models failed, falling back to GPT-4o-mini.")
            completion = await self.client.chat.completions.create(
                messages=chat_messages,
                **fallback_model
            )
            return completion.choices[0].message.content
        except openai.OpenAIError as e:
            print(f"Final fallback failed: {e}")
            return "Error: Unable to generate response."
     
     
     
    async def get_chunk_response_from_gpt(
                self,
                user_query: str,
                chunk_info: list,  # Ensure this is a list of objects with document & context_data
                industry_type: str,
                language_code: str,
                is_outside_from_document: bool,
                is_outside_from_industry: bool,
                chat_id: int | None = None,
            ) -> AsyncGenerator[str, Any]:
                
            print("inside get_response_from_langchain=========>", chat_id, industry_type, language_code, is_outside_from_document, is_outside_from_industry)
            
            try:
                chat = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=4096,
                    api_key=self.openai_key,
                )

                chat_history = ChatMessageHistory()
                db: Session = next(get_db())

                query = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.chat_id == chat_id)
                    .order_by(ChatMessage.id.desc())
                    .limit(2)
                ) 

                results = query.all()

                chat_history_text = ""
                for msg in results:
                    msg_data = msg.to_dict()
                    if not msg_data["is_bot"]:
                        chat_history_text += f"User: {msg_data['message']}\n"
                    else:
                        chat_history_text += f"AI: {msg_data['message']}\n"
                            
                chat_history_text += f"User: {user_query}\n"  

                chat_history.add_user_message(user_query)

                # Ensure chunk_info is a list and format properly

                context_data = "\n".join([
                    f"Document: {ast.literal_eval(ctx)['document_type']}\nContext: {ast.literal_eval(ctx)['context_data']}"
                    if isinstance(ctx, str) else f"Document: {ctx['document_type']}\nContext: {ctx['context_data']}"
                    for ctx in chunk_info
                ])

                prompt_text = f"""
                You are an intelligent AI assistant that answers user queries based on provided dynamic context data.

                User Query: {user_query}

                ### Context Data:
                {context_data}

                Instructions:
                - Find the **best matching** context chunk using similarity scoring.
                - Select **one document_type** based on the highest relevance.
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

                # Correctly set up the ChatPromptTemplate with the "messages" placeholder
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an intelligent AI assistant that answers user queries based on provided dynamic context data."),
                    MessagesPlaceholder(variable_name="messages"),
                    ("human", prompt_text)
                ])

                print("==========LANCHAIN INSIDE ELSE PROMPT=============", prompt)

                chain = prompt | chat

                result = chain.stream({"messages": chat_history.messages})
                print("result>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",result)

                return result

            except Exception as e:
                print(e, "error in get_response_from_langchain")
                capture_exception(e)
                return None
     
     
     
     

    async def get_response_from_langchain(
        self,
        user_query: str,
        chunk_info: str,
        industry_type: str,
        language_code: str,
        is_outside_from_document: bool,
        is_outside_from_industry: bool,
        chat_id: int | None = None,
    ) -> AsyncGenerator[str, Any]:
        
        print("inside get_response_from_langchain=========>",chat_id)
        try:
            chat = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=4096,
                api_key=self.openai_key,
            )
            chat_history = ChatMessageHistory()
            db = next(get_db())
            query = (
                db.query(ChatMessage)
                .filter(ChatMessage.chat_id ==chat_id)
                .order_by(ChatMessage.id.desc())
                .limit(2)
            ) 

            results = query.all()
 
            chat_history_text = ""
            for msg in results:
                if not msg.to_dict()["is_bot"]:
                    chat_history_text += f"User: {msg.to_dict()['message']}\n"
                else:
                    chat_history_text += f"AI: {msg.to_dict()['message']}\n"
                    
            chat_history_text += f"User: {user_query}\n"  

            chat_history.add_user_message(user_query)
            

            # context_info = [
            #     f"🔹 **User Query:** {user_query}",
            #     f"🔹 **Chat History (if relevant):** {chat_history_text if chat_history_text else 'No relevant history available.'}"
            # ]

            # if chunk_info:
            #     context_info.append(f"🔹 **Reference Material:** {chunk_info}\n")
                
            # context_info.append(f"🔹 **Industry Type:** {industry_type if not is_outside_from_industry else 'General Query (Not Industry-Specific)'}")

            # industry_guidelines = (
            #     f"- **Strictly adhere to the {industry_type} industry context** and **base responses solely on provided reference data**.\n"
            #     "- **DO NOT** use external knowledge for industry-related queries.\n"
            #     '- If an industry-related query **lacks sufficient reference data**, respond with:\n'
            #     '  *"I currently do not have sufficient information to answer this. Please check with your organization or relevant sources."*'
            # ) if not is_outside_from_industry else (
            #     "- **For queries outside the specified industry, generate responses using general knowledge.**\n"
            #     "- **Use `chunk_info` and `keywords_list` if relevant**, but also include general knowledge where needed.\n"
            #     "- **Ensure the response remains structured, clear, and fact-based.**"
            # )

            prompt = ChatPromptTemplate.from_messages(
                [
                    f"""
                    ## **Agent AI: Intelligent & Context-Aware Assistant**
                    - You are **Agent AI**, providing **clear, structured, and context-aware responses**.
                    - **Do not make assumptions**. Respond only with **relevant information** based on **user input** and **available context**.
                    - Avoid referencing external data unless explicitly allowed.
                    - Keep responses **concise**, **polite**, and **relevant**.

                    ## **Context Handling**
                    🔹 **User Query:** {user_query}  
                    🔹 **Chat History (only use if relevant):** {chat_history if chat_history else 'None'}  
                    🔹 **Reference Material:** {chunk_info if chunk_info else 'None'}  

                    ## **Response Guidelines**
                    - If the query involves personal or contact information (e.g., an email ID), and this information is available in the **Reference Material** or **Context**, respond with the relevant data.
                    - If **data is insufficient**, provide a message indicating that further information or clarification is needed.
                    - Always respect privacy and only provide information if it is explicitly available and appropriate.

                    ## **Focus on the `Reference Material` (if available) and `User Query` for accurate, context-driven responses.**
                    """
                    
                    
                ]
            )



            print('promtttt', f"""
                    ## **Agent AI: Intelligent & Context-Aware Assistant**
                    - You are **Agent AI**, providing **clear, structured, and context-aware responses**.
                    - **Do not make assumptions**. Respond only with **relevant information** based on **user input** and **available context**.
                    - Avoid referencing external data unless explicitly allowed.
                    - Keep responses **concise**, **polite**, and **relevant**.

                    ## **Context Handling**
                    🔹 **User Query:** {user_query}  
                    🔹 **Chat History (only use if relevant):** {chat_history if chat_history else 'None'}  
                    🔹 **Reference Material:** {chunk_info if chunk_info else 'None'}  

                    ## **Response Guidelines**
                    - If the query involves personal or contact information (e.g., an email ID), and this information is available in the **Reference Material** or **Context**, respond with the relevant data.
                    - If **data is insufficient**, provide a message indicating that further information or clarification is needed.
                    - Always respect privacy and only provide information if it is explicitly available and appropriate.

                    ## **Focus on the `Reference Material` (if available) and `User Query` for accurate, context-driven responses.**
                    """)
            print("==========LANCHAIN INSIDE ELSE PROMPT=============")
            MessagesPlaceholder(variable_name="messages")

            chain = prompt | chat

            result = chain.stream({"messages": chat_history.messages})

            return result

        except Exception as e:
            print(e, "error in get_response_from_langchain")
            capture_exception(e)
            return None

    async def get_response_from_langchain_if_answer_is_present_in_chat(
        self, db: Session, chat_id: int, user_query: str
    ):
        try:
            results = []

            chat = ChatOpenAI(
                model="gpt-4-0125-preview",
                temperature=0.2,
                max_tokens=1000,
                api_key=self.openai_key,
            )

            query = db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id)

            results = query.all()

            chat_history = ChatMessageHistory()

            for msg in results:
                if msg.to_dict()["is_bot"]:
                    chat_history.add_ai_message(msg.to_dict()["message"])
                else:
                    chat_history.add_user_message(msg.to_dict()["message"])

            chat_history.add_user_message(user_query)

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "[prose][Output only JSON]"
                        "You are an informative expert bot. Your task is to find the answer to the user's question only from the previous conversation. Please generate a response in JSON format following this specific schema: The JSON object should have two keys. The first key is 'answer' which should be associated with a string value. The second key is 'is_answer_present' which should be associated with a boolean value. If you are able to find the answer to the user's question from the previous conversation, then 'is_answer_present' should be true, otherwise, it should be false, and 'answer' should be an empty string. Output a JSON object fitting the schema provided {schema}. Do not include any key-value pair or other data based on assumptions. Use only the provided conversation for reference. Ensure proper JSON formatting and escape any double quotes. Avoid whitespace, line breaks, or indentation between fields.No codefence block. Respond in the same language as the input received.",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            chain = prompt | chat

            result = await chain.ainvoke(
                {
                    "messages": chat_history.messages,
                    "schema": "{answer: string, is_answer_present: boolean'}",
                }
            )

            response = result.content

            return response

        except Exception as e:
            print(e, "get_response_from_langchain_if_answer_is_present_in_chat")
            capture_exception(e)
            return None

    def followup_question_generation(self, *, user_input: str) -> list[dict[str, str]]:

        messages = [
            {
                "role": "system",
                "content": (
                    "[prose]"
                    "[Output only Array]"
                    "You are an expert bot that generates an array of similar questions based on the given data. "
                    "Ensure the meaning of the original question is preserved, and all variations are rephrased accordingly. "
                    "Provide an array of all possible questions derived from the given data. "
                    "Do not include any code fence block. Return only the data in array format. "
                    "Ensure that you respond in the same language as the input."
                    f"Here is the user data: {user_input}\n"
                ),
            },
        ]

        return messages

    async def generate_questions_from_chatgpt(
        self, user_input: str
    ) -> AsyncGenerator[str, None]:
        """Gets chat-based GPT response"""
        prompt = self.followup_question_generation(user_input=user_input)

        return self.gpt_response_with_stream_2(prompt)

    async def get_followup_question(
        self,
        chat_id: str,
        websocket: WebSocket,
        connection_manager,
        user_question: str,
        bot_response: str,
    ) -> AsyncGenerator[str, None]:
        from app.features.bot.websocket_response import WebSocketResponse as WR

        """Gets chat-based GPT response"""

        prompt = self.generate_followup_question(
            user_question=user_question, bot_response=bot_response
        )

        res = self.gpt_response_with_stream(prompt)
        await WR(
            chat_id=chat_id,
            websocket=websocket,
            connection_manager=connection_manager,
        ).create_bot_response(
            res,
            None,
            False,
            None,
            msg_type="followup_msg",
        )

    def generate_followup_question(
        self, *, user_question: str, bot_response: str
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "[prose]"
                    "[Output only Array]"
                    "You are informative expert bot and your task is to generate meaningful followup question based on the user's input which is the last question and its response. You should strictly generate questions that are relevant to the user's input.\n"
                    "Provide me array of maximum 3 questions but you can adjust according to the user input or message and if not able to find or if user's input is just greetings or thanking message than provide me empty array.\n"
                    f"Here is the current user input:- last question:  {user_question}, answer: {bot_response}\n"
                    "It is must to provide the questions strictly from user's input itself.\n"
                    "Avoid whitespace, line breaks, or indentation between fields. No codefence block."
                    "Respond in the same language as the input received."
                ),
            },
        ]
        return messages
