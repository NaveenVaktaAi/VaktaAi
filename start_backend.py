#!/usr/bin/env python3
"""
Backend Startup Script
Starts all required services and the FastAPI application
"""

import os
import sys
import subprocess
import time
import asyncio
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is installed")
            return True
        else:
            print("❌ Docker is not installed")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Compose is available")
            return True
        else:
            print("❌ Docker Compose is not available")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose is not available")
        return False

def start_milvus():
    """Start Milvus using Docker Compose"""
    print("🚀 Starting Milvus...")
    
    # Check if docker-compose.yml exists
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml not found")
        return False
    
    try:
        # Start Milvus
        result = subprocess.run(['docker-compose', 'up', '-d'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Milvus started successfully")
            
            # Wait for Milvus to be ready
            print("⏳ Waiting for Milvus to be ready...")
            time.sleep(30)  # Wait 30 seconds
            
            return True
        else:
            print(f"❌ Failed to start Milvus: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error starting Milvus: {e}")
        return False

def check_environment():
    """Check environment variables"""
    print("🔍 Checking environment variables...")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for AI responses",
        "MONGO_URI": "MongoDB connection string",
        "MONGO_DB_NAME": "MongoDB database name"
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
        else:
            print(f"  ✅ {var}: Set")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def start_fastapi():
    """Start the FastAPI application"""
    print("🚀 Starting FastAPI application...")
    
    try:
        # Run the test script first
        print("🔍 Running backend readiness test...")
        result = subprocess.run([sys.executable, 'test_backend_readiness.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Backend readiness test passed")
        else:
            print("⚠️  Backend readiness test had issues:")
            print(result.stdout)
            print(result.stderr)
        
        # Start FastAPI with uvicorn
        print("🚀 Starting FastAPI server...")
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'app.main:app', 
            '--reload', 
            '--host', '0.0.0.0', 
            '--port', '5000',
            '--log-level', 'info'
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting FastAPI: {e}")

def main():
    """Main startup function"""
    print("🚀 Vakta AI Backend Startup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_docker():
        print("❌ Docker is required but not installed")
        print("Please install Docker from: https://docs.docker.com/get-docker/")
        return False
    
    if not check_docker_compose():
        print("❌ Docker Compose is required but not available")
        print("Please install Docker Compose from: https://docs.docker.com/compose/install/")
        return False
    
    if not check_environment():
        print("❌ Please set the required environment variables")
        print("Create a .env file with the required variables")
        return False
    
    # Start Milvus (optional)
    print("\n🔍 Checking if Milvus is needed...")
    if not start_milvus():
        print("⚠️  Milvus not started - some features may be limited")
        print("You can start Milvus manually later with: docker-compose up -d")
    
    # Start FastAPI
    print("\n🚀 Starting FastAPI application...")
    start_fastapi()

if __name__ == "__main__":
    main()


