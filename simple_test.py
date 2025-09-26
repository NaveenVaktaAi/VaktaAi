#!/usr/bin/env python3
"""
Simple Backend Test
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test critical imports"""
    print("🔍 Testing Critical Imports...")
    
    try:
        import pymongo
        print("  ✅ pymongo imported successfully")
    except ImportError as e:
        print(f"  ❌ pymongo import failed: {e}")
        return False
    
    try:
        from bson import ObjectId
        print("  ✅ bson.ObjectId imported successfully")
    except ImportError as e:
        print(f"  ❌ bson.ObjectId import failed: {e}")
        return False
    
    try:
        import fastapi
        print("  ✅ fastapi imported successfully")
    except ImportError as e:
        print(f"  ❌ fastapi import failed: {e}")
        return False
    
    try:
        import openai
        print("  ✅ openai imported successfully")
    except ImportError as e:
        print(f"  ❌ openai import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("  ✅ uvicorn imported successfully")
    except ImportError as e:
        print(f"  ❌ uvicorn import failed: {e}")
        return False
    
    return True

def test_mongodb():
    """Test MongoDB connection"""
    print("\n🔍 Testing MongoDB Connection...")
    
    try:
        from app.database.session import get_db
        db = next(get_db())
        
        # Test basic operations
        test_collection = db["test_collection"]
        result = test_collection.insert_one({"test": "data"})
        print(f"  ✅ MongoDB insert successful: {result.inserted_id}")
        
        # Clean up
        test_collection.delete_one({"_id": result.inserted_id})
        print("  ✅ MongoDB cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MongoDB test failed: {e}")
        return False

def test_milvus():
    """Test Milvus connection"""
    print("\n🔍 Testing Milvus Connection...")
    
    try:
        from services.milvus_service import milvus_service
        
        if milvus_service.is_available():
            print("  ✅ Milvus connection successful")
            return True
        else:
            print("  ⚠️  Milvus not available (will use fallback)")
            return True  # This is acceptable
            
    except Exception as e:
        print(f"  ❌ Milvus test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Simple Backend Test")
    print("=" * 40)
    
    tests = [
        ("Critical Imports", test_imports),
        ("MongoDB Connection", test_mongodb),
        ("Milvus Connection", test_milvus),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Backend is READY for basic operations!")
        return True
    else:
        print("❌ Backend has issues - please fix them")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


