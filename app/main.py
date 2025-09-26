from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict
import json
import asyncio
from datetime import datetime
import nltk
import uvicorn
from app.schemas.milvus.client import connect_to_milvus, disconnect_from_milvus
from app.features.docSathi.router import router as docSathi_router
from app.features.chat.router import router as chat_router
from app.ai_service import ai_service
from app.parent_agent import parent_agent
routes = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await get_manifests(["chatbot"])
    connect_to_milvus()
    print("Connected to Milvus")
    yield  # Start request handling

    tasks_to_close: List[asyncio.Future] = [disconnect_from_milvus()]

    await asyncio.gather(*tasks_to_close, return_exceptions=True)






app = FastAPI(title="AI Chatbot API", version="1.0.0",  lifespan=lifespan,)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




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
routes.include_router(router=docSathi_router)
routes.include_router(router=chat_router)
# app.include_router(processing_router)
# app.include_router(vector_router)
# app.include_router(rag_router)
app.include_router(routes, prefix="/api/v1")
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
