#!/usr/bin/env python3
"""
Example usage of the chat system
"""
import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.features.chat.repository import ChatRepository
from app.features.chat.schemas import ChatCreate, ChatMessageCreate

async def example_usage():
    """Example of how to use the chat system"""
    print("🚀 Chat System Usage Example")
    print("=" * 50)
    
    # Initialize repository
    chat_repo = ChatRepository()
    
    try:
        # 1. Create a new chat
        print("\n1. Creating a new chat...")
        chat_data = ChatCreate(
            user_id=1,
            document_id="507f1f77bcf86cd799439011",  # Example document ID
            title="Discussion about AI and Machine Learning",
            status="active"
        )
        chat_id = await chat_repo.create_chat(chat_data)
        print(f"✅ Chat created with ID: {chat_id}")
        
        # 2. Add messages to the chat
        print("\n2. Adding messages to the chat...")
        
        # User message
        user_message = ChatMessageCreate(
            chat_id=chat_id,
            message="What is artificial intelligence?",
            is_bot=False,
            type="text"
        )
        user_msg_id = await chat_repo.create_chat_message(user_message)
        print(f"✅ User message added: {user_msg_id}")
        
        # Bot response
        bot_message = ChatMessageCreate(
            chat_id=chat_id,
            message="Artificial Intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
            is_bot=True,
            token=45,
            type="text"
        )
        bot_msg_id = await chat_repo.create_chat_message(bot_message)
        print(f"✅ Bot message added: {bot_msg_id}")
        
        # 3. Get chat with messages
        print("\n3. Retrieving chat with messages...")
        chat_with_messages = await chat_repo.get_chat_with_messages(chat_id)
        if chat_with_messages:
            print(f"📋 Chat: {chat_with_messages.chat.title}")
            print(f"   Status: {chat_with_messages.chat.status}")
            print(f"   Total Messages: {chat_with_messages.total_messages}")
            print(f"   Messages:")
            for msg in chat_with_messages.messages:
                sender = "🤖 Bot" if msg.is_bot else "👤 User"
                print(f"     {sender}: {msg.message}")
        
        # 4. Add reaction to a message
        print("\n4. Adding reaction to bot message...")
        success = await chat_repo.add_reaction_to_message(bot_msg_id, "👍")
        if success:
            print("✅ Reaction added successfully")
        
        # 5. Get user's chats
        print("\n5. Getting all chats for user...")
        user_chats = await chat_repo.get_user_chats(user_id=1, page=1, limit=10)
        print(f"📋 User has {user_chats.total} chats:")
        for chat in user_chats.chats:
            print(f"   - {chat.title} (Status: {chat.status})")
        
        print("\n🎉 Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in example: {e}")

if __name__ == "__main__":
    asyncio.run(example_usage())
