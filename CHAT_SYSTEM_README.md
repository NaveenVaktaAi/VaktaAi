# Chat System Documentation

## Overview

This chat system provides a complete MongoDB-based solution for managing conversations between users and AI bots. It includes two main collections: `chats` and `chat_messages`.

## Collections Structure

### 1. `chats` Collection

| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Unique chat identifier |
| `user_id` | int | ID of the user who created the chat |
| `document_id` | ObjectId (optional) | ID of the document this chat is related to |
| `status` | string | Chat status: "active" or "inactive" |
| `title` | string | Title of the chat |
| `created_at` | datetime | When the chat was created |
| `updated_at` | datetime | When the chat was last updated |

### 2. `chat_messages` Collection

| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Unique message identifier |
| `chat_id` | ObjectId | Foreign key reference to chats._id |
| `message` | string | The actual message content |
| `is_bot` | boolean | Whether this message is from bot (true) or user (false) |
| `reaction` | string (optional) | Reaction emoji for the message |
| `token` | int (optional) | Token count for the message |
| `type` | string | Message type: "text", "image", etc. (default: "text") |
| `is_edited` | boolean | Whether the message has been edited (default: false) |
| `created_ts` | datetime | When the message was created |
| `updated_ts` | datetime | When the message was last updated |

## WebSocket Integration

### Real-time Chat with WebSocket

The chat system now includes full WebSocket support for real-time communication, adapted from the bot system to work with MongoDB collections.

#### WebSocket Connection

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/chat/ws/{chat_id}');

// Send message
ws.send(JSON.stringify({
    mt: "message_upload",
    message: "Hello, how are you?",
    userId: "123",
    timezone: "UTC",
    selectedLanguage: "en"
}));

// Listen for messages
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

#### WebSocket Message Types

- `message_upload` - Send user message
- `message_upload_confirm` - Confirmation of message received
- `chat_message_bot_partial` - Streaming bot response (partial)
- `message_upload_confirm` - Complete bot response
- `stop` - Stop bot generation
- `stopped_generation` - Confirmation of stopped generation
- `error` - Error messages
- `typing_indicator` - Show bot is typing
- `stop_typing_indicator` - Hide typing indicator

## API Endpoints

### Chat Management

#### Create Chat
```http
POST /chat/
Content-Type: application/json

{
    "user_id": 1,
    "document_id": "507f1f77bcf86cd799439011",
    "title": "AI Discussion",
    "status": "active"
}
```

#### Get User Chats
```http
GET /chat/user/{user_id}?page=1&limit=20
```

#### Get Specific Chat
```http
GET /chat/{chat_id}
```

#### Update Chat
```http
PUT /chat/{chat_id}
Content-Type: application/json

{
    "title": "Updated Title",
    "status": "inactive"
}
```

#### Delete Chat
```http
DELETE /chat/{chat_id}
```

### Message Management

#### Get Chat Messages
```http
GET /chat/{chat_id}/messages?page=1&limit=50
```

#### Create Message
```http
POST /chat/{chat_id}/messages
Content-Type: application/json

{
    "message": "Hello, how are you?",
    "is_bot": false,
    "type": "text"
}
```

#### Get Specific Message
```http
GET /chat/{chat_id}/messages/{message_id}
```

#### Update Message
```http
PUT /chat/{chat_id}/messages/{message_id}
Content-Type: application/json

{
    "message": "Updated message content",
    "reaction": "👍"
}
```

#### Delete Message
```http
DELETE /chat/{chat_id}/messages/{message_id}
```

### Reactions

#### Add Reaction
```http
POST /chat/{chat_id}/messages/{message_id}/reaction?reaction=👍
```

#### Remove Reaction
```http
DELETE /chat/{chat_id}/messages/{message_id}/reaction
```

### Combined Operations

#### Get Chat with Messages
```http
GET /chat/{chat_id}/full?page=1&limit=50
```

### WebSocket Endpoints

#### WebSocket Connection
```http
WebSocket /chat/ws/{chat_id}
```

#### Get WebSocket Status
```http
GET /chat/ws/{chat_id}/status
```

## Database Indexes

The system automatically creates the following indexes for optimal performance:

### Chats Collection
- `_id_` (default MongoDB index)
- `user_id_1` - For querying user's chats
- `document_id_1` - For querying chats by document
- `status_1` - For filtering by status
- `created_at_1` - For sorting by creation date

### Chat Messages Collection
- `_id_` (default MongoDB index)
- `chat_id_1` - For querying messages by chat
- `is_bot_1` - For filtering bot vs user messages
- `created_ts_1` - For sorting messages chronologically

## Usage Examples

### Python Code Example

```python
from app.features.chat.repository import ChatRepository
from app.features.chat.schemas import ChatCreate, ChatMessageCreate

# Initialize repository
chat_repo = ChatRepository()

# Create a new chat
chat_data = ChatCreate(
    user_id=1,
    document_id="507f1f77bcf86cd799439011",
    title="AI Discussion",
    status="active"
)
chat_id = await chat_repo.create_chat(chat_data)

# Add a user message
user_message = ChatMessageCreate(
    chat_id=chat_id,
    message="What is artificial intelligence?",
    is_bot=False,
    type="text"
)
message_id = await chat_repo.create_chat_message(user_message)

# Add a bot response
bot_message = ChatMessageCreate(
    chat_id=chat_id,
    message="AI is a branch of computer science...",
    is_bot=True,
    token=45,
    type="text"
)
bot_msg_id = await chat_repo.create_chat_message(bot_message)

# Get chat with messages
chat_with_messages = await chat_repo.get_chat_with_messages(chat_id)
```

## Setup Instructions

1. **Initialize Collections**: Run the setup script to create collections and indexes:
   ```bash
   python scripts/02_setup_chat_collections.py
   ```

2. **Import in Main App**: Add the chat router to your main FastAPI app:
   ```python
   from app.features.chat.router import router as chat_router
   app.include_router(chat_router)
   ```

3. **Test the System**: Run the example script:
   ```bash
   python scripts/chat_usage_example.py
   ```

## Features

- ✅ **Complete CRUD Operations** for chats and messages
- ✅ **Real-time WebSocket Communication** adapted from bot system
- ✅ **Streaming Bot Responses** with partial message updates
- ✅ **Pagination Support** for large message histories
- ✅ **Reaction System** for message interactions
- ✅ **Message Editing** with edit tracking
- ✅ **Bot vs User Message** differentiation
- ✅ **Token Counting** for AI responses
- ✅ **Document Association** linking chats to documents
- ✅ **Multi-language Support** (English/Hindi)
- ✅ **Typing Indicators** for better UX
- ✅ **Connection Management** with automatic cleanup
- ✅ **Optimized Indexes** for fast queries
- ✅ **Type Safety** with Pydantic schemas
- ✅ **Error Handling** with proper HTTP status codes

## File Structure

```
app/features/chat/
├── __init__.py
├── schemas.py              # Pydantic models
├── repository.py           # Database operations
├── router.py              # FastAPI endpoints + WebSocket
├── websocket_manager.py   # WebSocket connection management
└── bot_handler.py         # Bot message handling for MongoDB

app/database/
└── mongo_collections.py   # Updated with chat collections

scripts/
├── 02_setup_chat_collections.py  # Setup script
└── chat_usage_example.py         # Usage example
```

## Error Handling

The system includes comprehensive error handling:
- **404 Not Found** for non-existent chats/messages
- **500 Internal Server Error** for database issues
- **Validation Errors** for invalid input data
- **Proper HTTP Status Codes** for all operations

## Performance Considerations

- **Indexes** are optimized for common query patterns
- **Pagination** prevents loading large datasets
- **Efficient Queries** using MongoDB aggregation when needed
- **Connection Pooling** through the existing database session management

This chat system is production-ready and can handle high-volume conversations with proper scaling.
