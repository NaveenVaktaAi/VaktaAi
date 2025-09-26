#!/usr/bin/env python3
"""
Simple test script to verify the backend can start and basic imports work
"""

def test_imports():
    """Test if all required imports work"""
    try:
        print("Testing imports...")
        
        # Test basic FastAPI imports
        from fastapi import FastAPI, WebSocket
        print("✓ FastAPI imports work")
        
        # Test our app imports
        from app.main import app
        print("✓ App imports work")
        
        # Test parent agent
        from app.parent_agent import parent_agent
        print("✓ Parent agent imports work")
        
        # Test AI service
        from app.ai_service import ai_service
        print("✓ AI service imports work")
        
        print("\n🎉 All imports successful! Backend should start properly.")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        print("\nTesting basic functionality...")
        
        from app.parent_agent import parent_agent
        
        # Test parent agent
        response = parent_agent.process_user_query(
            client_id="test_client",
            query="Hello, how are you?",
            conversation_history=[]
        )
        
        print(f"✓ Parent agent response: {response['response'][:50]}...")
        print("✓ Basic functionality works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing VaktaAi Backend Connection")
    print("=" * 50)
    
    imports_ok = test_imports()
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n✅ All tests passed! Backend is ready to run.")
            print("\nTo start the backend:")
            print("cd VaktaAi")
            print("python -m uvicorn app.main:app --host 0.0.0.0 --port 5000")
        else:
            print("\n❌ Functionality tests failed.")
    else:
        print("\n❌ Import tests failed. Check dependencies.")
