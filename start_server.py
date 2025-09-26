#!/usr/bin/env python3
"""
Simple script to start the VaktaAi backend server
"""

import uvicorn
import sys
import os

def start_server():
    """Start the FastAPI server"""
    try:
        print("🚀 Starting VaktaAi Backend Server...")
        print("📍 Server will be available at: http://127.0.0.1:5000")
        print("🔌 WebSocket endpoint: ws://127.0.0.1:5000/ws/{client_id}")
        print("💬 Chat WebSocket endpoint: ws://127.0.0.1:5000/api/v1/chat/ws/{chat_id}")
        print("🏥 Health check: http://127.0.0.1:5000/health")
        print("\n" + "="*60)
        
        # Start the server
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=5000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
