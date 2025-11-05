from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from app.llm_config import settings
import asyncio
from typing import List, Dict

class AIService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key="fdsfdsfdsfdsf"
            )
        
        # Memory to maintain conversation context
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Conversation chain with memory
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        
        # System prompt to define AI behavior
        self.system_prompt = """You are a helpful AI assistant. You are friendly, knowledgeable, and always try to provide accurate and helpful responses. Keep your responses concise but informative. If you don't know something, admit it rather than making up information."""

    async def get_ai_response(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """
        Get AI response for user message using LangChain
        """
        try:
            # Add system message if this is the first message
            if not conversation_history or len(conversation_history) == 0:
                # Initialize with system prompt
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=user_message)
                ]
                
                # Use the LLM directly for first message
                response = await asyncio.to_thread(
                    self.llm.invoke, messages
                )
                return response.content
            else:
                # Use conversation chain for subsequent messages
                response = await asyncio.to_thread(
                    self.conversation.predict, input=user_message
                )
                return response
                
        except Exception as e:
            print(f"Error getting AI response: {e}")
            return "I'm sorry, I encountered an error while processing your request. Please try again."

    def reset_conversation(self):
        """Reset conversation memory"""
        self.memory.clear()

# Global AI service instance
ai_service = AIService()
