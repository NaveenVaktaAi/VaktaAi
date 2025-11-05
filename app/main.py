from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import List, Dict
import json
import asyncio
from datetime import datetime
import nltk
import uvicorn
from app.avatar_config import ai_settings
from app.features.aiAvatar.aiTutorServices.wsHandler import handle_student_ws
from app.schemas.milvus.client import connect_to_milvus, disconnect_from_milvus
from app.features.docSathi.router import router as docSathi_router
from app.features.chat.router import router as chat_router
from app.features.auth.router import router as auth_router
from app.ai_service import ai_service
from app.parent_agent import parent_agent
from app.features.aiAvatar.router import router as aiAvatar_router
routes = APIRouter()

# run it when you want to connect to milvus

@asynccontextmanager
async def lifespan(app: FastAPI):
    # await get_manifests(["chatbot"])
    try:
        connect_to_milvus()
        print("Connected to Milvus")
    except Exception as e:
        print(f"Warning: Could not connect to Milvus: {e}")
        print("Server will start without Milvus connection")
    
    # Define tasks to close before yielding
    tasks_to_close: List[asyncio.Future] = []
    
    yield  # Start request handling

    # Cleanup tasks
    try:
        tasks_to_close.append(disconnect_from_milvus())
        await asyncio.gather(*tasks_to_close, return_exceptions=True)
    except Exception as e:
        print(f"Error during cleanup: {e}")






app = FastAPI(title="AI Chatbot API", version="1.0.0")

# app = FastAPI(title="AI Chatbot API", version="1.0.0",  lifespan=lifespan,)

# CORS middleware to allow frontend connections (MUST be before static files)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001" ,"http://localhost:3002"],  # Frontend dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio (AFTER CORS middleware)
# Ensure static files directory exists
import os
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS headers for static files explicitly
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Content-Type"] = "audio/mpeg" if request.url.path.endswith(".mp3") else response.headers.get("Content-Type", "application/octet-stream")
    return response

@app.get("/health")
async def health():
    return {"status": "healthy", "model": ai_settings.GROQ_MODEL}


@app.websocket("/tutor")
async def tutor_ws(ws: WebSocket):
    await handle_student_ws(ws)




# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Ensure punkt_tab is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

print("docSathi_router>>>>>>>>>>>>>>>>>>>>>>>>>>>>",docSathi_router)
print("chat_router>>>>>>>>>>>>>>>>>>>>>>>>>>>>",chat_router)

# Include all routers directly in the app with proper prefixes
app.include_router(docSathi_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1", tags=["Authentication"])
app.include_router(aiAvatar_router, prefix="/api/v1")
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_histories: Dict[str, List[Dict]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.conversation_histories[client_id] = []

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.conversation_histories:
            del self.conversation_histories[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    def add_to_history(self, client_id: str, message: Dict):
        if client_id not in self.conversation_histories:
            self.conversation_histories[client_id] = []
        self.conversation_histories[client_id].append(message)

    def get_history(self, client_id: str) -> List[Dict]:
        return self.conversation_histories.get(client_id, [])

# manager = ConnectionManager()

# @app.get("/")
# async def root():
#     return {"message": "AI Chatbot API is running"}

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# @app.post("/api/v1/reset/{client_id}")
# async def reset_conversation(client_id: str):
#     """Reset conversation for a client"""
#     try:
#         # Clear conversation history for the client
#         if client_id in manager.conversation_histories:
#             manager.conversation_histories[client_id] = []
        
#         return {"success": True, "message": "Conversation reset successfully"}
#     except Exception as e:
#         return {"success": False, "message": f"Error resetting conversation: {str(e)}"}

# @app.websocket("/ws/{client_id}")
# async def websocket_endpoint(websocket: WebSocket, client_id: str):
#     await manager.connect(websocket, client_id)
#     try:
#         while True:
#             # Receive message from client
#             data = await websocket.receive_text()
#             message_data = json.loads(data)
            
#             user_message = message_data.get('content', '')
#             timestamp = datetime.now().isoformat()
            
#             # Add user message to history
#             user_msg = {
#                 "content": user_message,
#                 "sender": "user",
#                 "timestamp": timestamp
#             }
#             manager.add_to_history(client_id, user_msg)
            
#             # Get conversation history
#             history = manager.get_history(client_id)
            
#             try:
#                 # Process query through RAG system
#                 rag_response = parent_agent.process_user_query(
#                     client_id=client_id,
#                     query=user_message,
#                     conversation_history=history
#                 )
                
#                 # Create enhanced response with RAG metadata
#                 response = {
#                     "mt": "message_upload_confirm",
#                     "message": rag_response["response"],
#                     "content": rag_response["response"],
#                     "timestamp": datetime.now().isoformat(),
#                     "isBot": True,
#                     "token": f"{datetime.now().timestamp()}_{client_id}",
#                     "metadata": {
#                         "source_documents": rag_response.get("source_documents", []),
#                         "context_chunks_used": rag_response.get("context_chunks_used", 0),
#                         "retrieval_performed": rag_response.get("retrieval_performed", False),
#                         "processing_time": rag_response.get("processing_time", 0),
#                         "suggestion": rag_response.get("suggestion")
#                     }
#                 }
                
#             except Exception as e:
#                 print(f"RAG processing failed, falling back to basic AI: {e}")
#                 # Fallback to basic AI service
#                 ai_response = await ai_service.get_ai_response(user_message, history)
#                 response = {
#                     "mt": "message_upload_confirm",
#                     "message": ai_response,
#                     "content": ai_response,
#                     "timestamp": datetime.now().isoformat(),
#                     "isBot": True,
#                     "token": f"{datetime.now().timestamp()}_{client_id}",
#                     "metadata": {
#                         "fallback_mode": True,
#                         "error": "RAG system unavailable"
#                     }
#                 }
            
#             # Add AI response to history
#             manager.add_to_history(client_id, response)
            
#             # Send response back to client
#             await manager.send_personal_message(json.dumps(response), client_id)
            
#     except WebSocketDisconnect:
#         manager.disconnect(client_id)
#         print(f"Client {client_id} disconnected")
#     except Exception as e:
#         print(f"Error in websocket connection: {e}")
#         manager.disconnect(client_id)

# @app.post("/reset/{client_id}")
# async def reset_conversation(client_id: str):
#     """Reset conversation for a specific client"""
#     if client_id in manager.conversation_histories:
#         manager.conversation_histories[client_id] = []
#     ai_service.reset_conversation()
#     return {"message": f"Conversation reset for client {client_id}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
