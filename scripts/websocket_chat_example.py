#!/usr/bin/env python3
"""
WebSocket Chat Example - demonstrates real-time chat functionality
"""
import asyncio
import websockets
import json
import sys
import os

# Add the parent directory to the path so we can import from app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from app.features.chat.repository import ChatRepository
from app.features.chat.schemas import ChatCreate, ChatMessageCreate

async def websocket_client_example():
    """Example WebSocket client for testing chat functionality"""
    
    # First, create a chat using the API
    print("🚀 WebSocket Chat Example")
    print("=" * 50)
    
    # Create a chat
    chat_repo = ChatRepository()
    chat_data = ChatCreate(
        user_id=1,
        title="WebSocket Test Chat",
        status="active"
    )
    chat_id = await chat_repo.create_chat(chat_data)
    print(f"✅ Created chat with ID: {chat_id}")
    
    # WebSocket URL
    websocket_url = f"ws://localhost:8000/chat/ws/{chat_id}"
    
    try:
        print(f"\n🔌 Connecting to WebSocket: {websocket_url}")
        
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Connected to WebSocket!")
            
            # Send a test message
            message = {
                "mt": "message_upload",
                "message": "Hello! This is a test message from the WebSocket client.",
                "userId": "1",
                "timezone": "UTC",
                "selectedLanguage": "en"
            }
            
            print(f"\n📤 Sending message: {message['message']}")
            await websocket.send(json.dumps(message))
            
            # Listen for responses
            print("\n👂 Listening for responses...")
            response_count = 0
            
            async for response in websocket:
                data = json.loads(response)
                response_count += 1
                
                print(f"\n📥 Response {response_count}:")
                print(f"   Type: {data.get('mt', 'unknown')}")
                print(f"   Message: {data.get('message', 'No message')}")
                print(f"   Is Bot: {data.get('isBot', False)}")
                print(f"   Timestamp: {data.get('timestamp', 'No timestamp')}")
                
                # Stop after receiving a few responses
                if response_count >= 5:
                    break
            
            print(f"\n✅ Received {response_count} responses successfully!")
            
    except websockets.exceptions.ConnectionClosed:
        print("❌ WebSocket connection closed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n🎉 WebSocket chat example completed!")
    print(f"Chat ID: {chat_id}")

def create_html_test_page():
    """Create a simple HTML page for testing WebSocket chat"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Chat Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chat-container { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background-color: #e3f2fd; text-align: right; }
        .bot-message { background-color: #f3e5f5; text-align: left; }
        .input-container { margin-top: 20px; }
        input[type="text"] { width: 70%; padding: 10px; }
        button { padding: 10px 20px; margin-left: 10px; }
        .status { margin: 10px 0; padding: 10px; background-color: #e8f5e8; }
        .error { background-color: #ffebee; color: red; }
    </style>
</head>
<body>
    <h1>WebSocket Chat Test</h1>
    
    <div id="status" class="status">Disconnected</div>
    
    <div class="chat-container" id="chatContainer"></div>
    
    <div class="input-container">
        <input type="text" id="messageInput" placeholder="Type your message..." disabled>
        <button id="sendButton" onclick="sendMessage()" disabled>Send</button>
        <button id="connectButton" onclick="connect()">Connect</button>
        <button id="disconnectButton" onclick="disconnect()" disabled>Disconnect</button>
    </div>
    
    <script>
        let ws = null;
        let chatId = null;
        
        function updateStatus(message, isError = false) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = isError ? 'status error' : 'status';
        }
        
        function addMessage(content, isBot = false, messageType = '') {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = isBot ? 'message bot-message' : 'message user-message';
            
            let displayContent = content;
            if (messageType === 'partial') {
                displayContent = `[Streaming...] ${content}`;
            }
            
            messageDiv.innerHTML = `
                <strong>${isBot ? 'Bot' : 'You'}:</strong> ${displayContent}
                <br><small>${new Date().toLocaleTimeString()}</small>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        function connect() {
            // First create a chat
            fetch('/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: 1,
                    title: 'WebSocket Test Chat',
                    status: 'active'
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    chatId = data.chat_id;
                    connectWebSocket();
                } else {
                    updateStatus('Failed to create chat: ' + data.message, true);
                }
            })
            .catch(error => {
                updateStatus('Error creating chat: ' + error, true);
            });
        }
        
        function connectWebSocket() {
            const wsUrl = `ws://localhost:8000/chat/ws/${chatId}`;
            updateStatus('Connecting...');
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                updateStatus('Connected to chat: ' + chatId);
                document.getElementById('messageInput').disabled = false;
                document.getElementById('sendButton').disabled = false;
                document.getElementById('connectButton').disabled = true;
                document.getElementById('disconnectButton').disabled = false;
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                console.log('Received:', data);
                
                if (data.mt === 'message_upload_confirm') {
                    addMessage(data.message, data.isBot);
                } else if (data.mt === 'chat_message_bot_partial') {
                    if (data.partial) {
                        addMessage(data.partial, true, 'partial');
                    }
                } else if (data.mt === 'error') {
                    addMessage('Error: ' + data.message, true);
                    updateStatus('Error: ' + data.message, true);
                }
            };
            
            ws.onclose = function() {
                updateStatus('Disconnected');
                document.getElementById('messageInput').disabled = true;
                document.getElementById('sendButton').disabled = true;
                document.getElementById('connectButton').disabled = false;
                document.getElementById('disconnectButton').disabled = true;
            };
            
            ws.onerror = function(error) {
                updateStatus('WebSocket error', true);
                console.error('WebSocket error:', error);
            };
        }
        
        function disconnect() {
            if (ws) {
                ws.close();
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                const messageData = {
                    mt: 'message_upload',
                    message: message,
                    userId: '1',
                    timezone: 'UTC',
                    selectedLanguage: 'en'
                };
                
                ws.send(JSON.stringify(messageData));
                input.value = '';
            }
        }
        
        // Allow sending message with Enter key
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
    """
    
    with open("websocket_chat_test.html", "w") as f:
        f.write(html_content)
    
    print("📄 Created websocket_chat_test.html for browser testing")

if __name__ == "__main__":
    print("WebSocket Chat Example")
    print("1. Run the WebSocket client example")
    print("2. Create HTML test page")
    
    # Run the WebSocket example
    asyncio.run(websocket_client_example())
    
    # Create HTML test page
    create_html_test_page()
    
    print("\n📋 Next steps:")
    print("1. Start your FastAPI server: uvicorn app.main:app --reload")
    print("2. Open websocket_chat_test.html in your browser")
    print("3. Connect and test the chat functionality")
