#!/usr/bin/env python3
"""
Setup script for chat collections in MongoDB
"""
import sys
import os

# Add the parent directory to the path so we can import from app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from app.database.session import get_db
from app.database.mongo_collections import get_collections

def setup_chat_collections():
    """Initialize chat collections with proper indexes"""
    try:
        print("🔧 Setting up chat collections...")
        
        # Get database connection
        db = next(get_db())
        
        # Initialize collections (this will create indexes)
        documents, chunks, chats, chat_messages = get_collections(db)
        
        print("✅ Chat collections setup completed!")
        print(f"   - chats collection: {chats.name}")
        print(f"   - chat_messages collection: {chat_messages.name}")
        
        # Verify indexes
        print("\n📋 Created indexes:")
        
        chat_indexes = list(chats.list_indexes())
        print(f"   Chat collection indexes:")
        for index in chat_indexes:
            print(f"     - {index['name']}: {index['key']}")
        
        message_indexes = list(chat_messages.list_indexes())
        print(f"   Chat messages collection indexes:")
        for index in message_indexes:
            print(f"     - {index['name']}: {index['key']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up chat collections: {e}")
        return False

if __name__ == "__main__":
    success = setup_chat_collections()
    if success:
        print("\n🎉 Chat collections are ready to use!")
    else:
        print("\n💥 Failed to setup chat collections!")
        sys.exit(1)
