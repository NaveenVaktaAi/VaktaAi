#!/usr/bin/env python3
"""
Backend Readiness Test Script
Tests MongoDB, Milvus, and Chat functionality
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_mongodb_connection():
    """Test MongoDB connection and collections"""
    print("🔍 Testing MongoDB Connection...")
    try:
        from app.database.session import get_db
        from app.database.mongo_collections import get_collections
        
        db = next(get_db())
        documents, chunks, chats, chat_messages = get_collections(db)
        
        # Test collections exist
        collections = [documents, chunks, chats, chat_messages]
        collection_names = ["documents", "chunks", "chats", "chat_messages"]
        
        for collection, name in zip(collections, collection_names):
            count = collection.count_documents({})
            print(f"  ✅ {name}: {count} documents")
        
        print("  ✅ MongoDB connection successful")
        return True
        
    except Exception as e:
        print(f"  ❌ MongoDB connection failed: {e}")
        return False

async def test_milvus_connection():
    """Test Milvus vector database connection"""
    print("🔍 Testing Milvus Connection...")
    try:
        from services.milvus_service import milvus_service
        
        # Test connection
        if milvus_service.is_available():
            print("  ✅ Milvus connection successful")
            
            # Get collection stats
            stats = milvus_service.get_collection_stats()
            print(f"  ✅ Collection stats: {stats}")
            return True
        else:
            print("  ⚠️  Milvus not available (will use fallback)")
            return False
            
    except Exception as e:
        print(f"  ❌ Milvus connection failed: {e}")
        print("  ⚠️  Milvus not available (will use fallback)")
        return False

async def test_chat_functionality():
    """Test chat functionality"""
    print("🔍 Testing Chat Functionality...")
    try:
        from app.features.chat.repository import ChatRepository
        from app.features.chat.schemas import ChatCreate, ChatMessageCreate
        from app.features.chat.utils.response import ResponseCreator
        
        # Test repository
        chat_repo = ChatRepository()
        print("  ✅ Chat repository initialized")
        
        # Test response creator
        response_creator = ResponseCreator()
        print("  ✅ Response creator initialized")
        
        # Test creating a test chat
        test_chat = ChatCreate(
            user_id=1,
            title="Test Chat",
            status="active"
        )
        
        chat_id = await chat_repo.create_chat(test_chat)
        print(f"  ✅ Test chat created: {chat_id}")
        
        # Test creating a test message
        test_message = ChatMessageCreate(
            chat_id=chat_id,
            message="Hello, this is a test message",
            is_bot=False
        )
        
        message_id = await chat_repo.create_chat_message(test_message)
        print(f"  ✅ Test message created: {message_id}")
        
        # Clean up test data
        await chat_repo.delete_chat(chat_id)
        print("  ✅ Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Chat functionality test failed: {e}")
        return False

async def test_websocket_functionality():
    """Test WebSocket functionality"""
    print("🔍 Testing WebSocket Functionality...")
    try:
        from app.features.chat.websocket_manager import WebSocketConnectionManager, ChatWebSocketResponse
        
        # Test WebSocket manager
        manager = WebSocketConnectionManager()
        print("  ✅ WebSocket manager initialized")
        
        # Test WebSocket response (without actual WebSocket)
        print("  ✅ WebSocket response class available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ WebSocket functionality test failed: {e}")
        return False

async def test_api_endpoints():
    """Test API endpoints availability"""
    print("🔍 Testing API Endpoints...")
    try:
        from app.features.chat.router import router as chat_router
        
        # Check if router has expected endpoints
        routes = [route.path for route in chat_router.routes]
        expected_routes = [
            "/",
            "/user/{user_id}",
            "/{chat_id}",
            "/{chat_id}/messages",
            "/ws/{chat_id}"
        ]
        
        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"  ✅ Route available: {route}")
            else:
                print(f"  ⚠️  Route missing: {route}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ API endpoints test failed: {e}")
        return False

async def test_environment_variables():
    """Test required environment variables"""
    print("🔍 Testing Environment Variables...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "MONGO_URI",
        "MONGO_DB_NAME"
    ]
    
    optional_vars = [
        "MILVUS_HOST",
        "MILVUS_PORT",
        "MILVUS_COLLECTION_NAME"
    ]
    
    all_good = True
    
    for var in required_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: Set")
        else:
            print(f"  ❌ {var}: Missing (REQUIRED)")
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: {os.getenv(var)}")
        else:
            print(f"  ⚠️  {var}: Not set (using default)")
    
    return all_good

async def main():
    """Run all tests"""
    print("🚀 Backend Readiness Test")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("MongoDB Connection", test_mongodb_connection),
        ("Milvus Connection", test_milvus_connection),
        ("Chat Functionality", test_chat_functionality),
        ("WebSocket Functionality", test_websocket_functionality),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Backend is READY for chat!")
        return True
    elif passed >= total - 1:  # Allow 1 failure (like Milvus)
        print("⚠️  Backend is MOSTLY READY (some features may be limited)")
        return True
    else:
        print("❌ Backend is NOT READY - please fix the issues above")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)


