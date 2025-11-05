# AI Tutor API Documentation

## Overview
Complete API documentation for AI Tutor conversation system. Similar to chat system but without document references - pure AI tutoring conversations.

## Base URL
```
/ai-tutor
```

## API Endpoints

### 1. Create Conversation
```
POST /ai-tutor/conversations
```

**Request Body:**
```json
{
  "user_id": 123,
  "title": "",
  "status": "active",
  "subject": "Mathematics",
  "tags": ["math", "algebra"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Conversation created successfully",
  "data": {
    "conversation_id": "6507f1f77e1a2b3c4d5e6f7a"
  }
}
```

### 2. Get User Conversations
```
GET /ai-tutor/conversations/user/{user_id}?page=1&limit=20&status=active
```

**Response:**
```json
{
  "conversations": [
    {
      "_id": "6507f1f77e1a2b3c4d5e6f7a",
      "user_id": 123,
      "title": "Help with Quadratic Equations",
      "status": "active",
      "subject": "Mathematics",
      "tags": ["math", "algebra"],
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 5,
  "page": 1,
  "limit": 20
}
```

### 3. Get Single Conversation
```
GET /ai-tutor/conversations/{conversation_id}
```

**Response:**
```json
{
  "_id": "6507f1f77e1a2b3c4d5e6f7a",
  "user_id": 123,
  "title": "Help with Quadratic Equations",
  "status": "active",
  "subject": "Mathematics",
  "tags": ["math", "algebra"],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### 4. Update Conversation
```
PUT /ai-tutor/conversations/{conversation_id}
```

**Request Body:**
```json
{
  "title": "Advanced Quadratic Equations",
  "status": "active",
  "subject": "Advanced Mathematics",
  "tags": ["math", "algebra", "advanced"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Conversation updated successfully",
  "data": {
    "conversation_id": "6507f1f77e1a2b3c4d5e6f7a"
  }
}
```

### 5. Delete Conversation
```
DELETE /ai-tutor/conversations/{conversation_id}
```

**Response:**
```json
{
  "success": true,
  "message": "Conversation deleted successfully",
  "data": {
    "conversation_id": "6507f1f77e1a2b3c4d5e6f7a"
  }
}
```

### 6. Create Message
```
POST /ai-tutor/conversations/{conversation_id}/messages
```

**Request Body:**
```json
{
  "conversation_id": "6507f1f77e1a2b3c4d5e6f7a",
  "message": "Can you explain quadratic equations?",
  "is_bot": false,
  "type": "text"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Message created successfully",
  "data": {
    "message_id": "6507f2a88e1a2b3c4d5e6f7b",
    "conversation_id": "6507f1f77e1a2b3c4d5e6f7a"
  }
}
```

### 7. Get Conversation Messages
```
GET /ai-tutor/conversations/{conversation_id}/messages?page=1&limit=50
```

**Response:**
```json
[
  {
    "_id": "6507f2a88e1a2b3c4d5e6f7b",
    "conversation_id": "6507f1f77e1a2b3c4d5e6f7a",
    "message": "Can you explain quadratic equations?",
    "is_bot": false,
    "type": "text",
    "is_edited": false,
    "created_ts": "2024-01-15T10:31:00Z",
    "updated_ts": "2024-01-15T10:31:00Z"
  },
  {
    "_id": "6507f2b99e1a2b3c4d5e6f7c",
    "conversation_id": "6507f1f77e1a2b3c4d5e6f7a",
    "message": "Of course! A quadratic equation is...",
    "is_bot": true,
    "type": "text",
    "is_edited": false,
    "created_ts": "2024-01-15T10:31:05Z",
    "updated_ts": "2024-01-15T10:31:05Z"
  }
]
```

### 8. Get Conversation with Messages
```
GET /ai-tutor/conversations/{conversation_id}/full?page=1&limit=50
```

**Response:**
```json
{
  "conversation": {
    "_id": "6507f1f77e1a2b3c4d5e6f7a",
    "user_id": 123,
    "title": "Help with Quadratic Equations",
    "status": "active",
    "subject": "Mathematics",
    "tags": ["math", "algebra"],
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  },
  "messages": [...],
  "total_messages": 10
}
```

### 9. Update Message
```
PUT /ai-tutor/messages/{message_id}
```

**Request Body:**
```json
{
  "message": "Can you explain quadratic equations in detail?",
  "is_edited": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Message updated successfully",
  "data": {
    "message_id": "6507f2a88e1a2b3c4d5e6f7b"
  }
}
```

### 10. Delete Message
```
DELETE /ai-tutor/messages/{message_id}
```

**Response:**
```json
{
  "success": true,
  "message": "Message deleted successfully",
  "data": {
    "message_id": "6507f2a88e1a2b3c4d5e6f7b"
  }
}
```

### 11. Get User Statistics
```
GET /ai-tutor/users/{user_id}/stats
```

**Response:**
```json
{
  "total_conversations": 10,
  "active_conversations": 8,
  "archived_conversations": 2,
  "total_messages": 150,
  "bot_messages": 75,
  "user_messages": 75
}
```

### 12. Search Conversations
```
GET /ai-tutor/conversations/search/{user_id}?q=math&limit=20
```

**Response:**
```json
[
  {
    "_id": "6507f1f77e1a2b3c4d5e6f7a",
    "user_id": 123,
    "title": "Help with Math - Quadratic Equations",
    "status": "active",
    "subject": "Mathematics",
    "tags": ["math", "algebra"],
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
]
```

## Testing Payloads for Swagger

### Create Conversation (Basic)
```json
{
  "user_id": 123,
  "title": "",
  "status": "active"
}
```

### Create Conversation (With Details)
```json
{
  "user_id": 123,
  "title": "Math Tutoring Session",
  "status": "active",
  "subject": "Mathematics",
  "tags": ["math", "algebra", "grade-10"]
}
```

### Create User Message
```json
{
  "conversation_id": "6507f1f77e1a2b3c4d5e6f7a",
  "message": "I need help understanding quadratic equations. Can you explain what they are?",
  "is_bot": false,
  "type": "text"
}
```

### Create Bot Message
```json
{
  "conversation_id": "6507f1f77e1a2b3c4d5e6f7a",
  "message": "Of course! A quadratic equation is a polynomial equation of degree 2. It has the general form: ax² + bx + c = 0, where a, b, and c are constants and a ≠ 0.",
  "is_bot": true,
  "type": "text"
}
```

### Update Conversation
```json
{
  "title": "Advanced Algebra - Quadratic Equations",
  "subject": "Advanced Mathematics"
}
```

### Update Message
```json
{
  "message": "Can you explain quadratic equations in more detail?",
  "is_edited": true
}
```

## Complete Testing Flow

### 1. Create a New Conversation
```bash
POST /ai-tutor/conversations
{
  "user_id": 123,
  "title": "",
  "status": "active",
  "subject": "Mathematics"
}
```

### 2. Add User Message (Auto-updates title if empty)
```bash
POST /ai-tutor/conversations/{conversation_id}/messages
{
  "conversation_id": "{conversation_id}",
  "message": "Can you help me with algebra?",
  "is_bot": false,
  "type": "text"
}
```

### 3. Add Bot Response
```bash
POST /ai-tutor/conversations/{conversation_id}/messages
{
  "conversation_id": "{conversation_id}",
  "message": "Of course! I'd be happy to help you with algebra. What specific topic would you like to learn about?",
  "is_bot": true,
  "type": "text"
}
```

### 4. Get All Messages
```bash
GET /ai-tutor/conversations/{conversation_id}/messages?page=1&limit=50
```

### 5. Get User's Conversations
```bash
GET /ai-tutor/conversations/user/123?page=1&limit=20
```

## Key Differences from Chat System

### What's Different:
- ❌ No `document_id` field
- ❌ No `training_doc_id` field
- ✅ Added `subject` field
- ✅ Added `tags` field
- ✅ Standalone conversations

### What's Similar:
- ✅ User-based conversations
- ✅ Message structure (is_bot, reactions, etc.)
- ✅ Status management (active/archived)
- ✅ Pagination support
- ✅ CRUD operations

## Auto-Features

### 1. Auto Title Generation
When the first user message is sent to a conversation with an empty title, the system automatically:
- Takes the first 50 characters of the message
- Sets it as the conversation title
- Updates the conversation

Example:
```
Message: "Can you help me understand quadratic equations?"
Auto-generated title: "Can you help me understand quadratic equations?"
```

### 2. Auto Timestamp Updates
- `created_at` - Set automatically when conversation/message is created
- `updated_at` - Updated automatically when conversation/message is modified
- `created_ts` - Message creation timestamp
- `updated_ts` - Message update timestamp

## Integration with Main App

To integrate with your FastAPI app:

```python
from app.features.aiAvatar import ai_tutor_router

app.include_router(ai_tutor_router)
```

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message description"
}
```

Common HTTP Status Codes:
- `200` - Success
- `404` - Conversation/Message not found
- `500` - Internal server error

## Next Steps

1. ✅ Collections created
2. ✅ CRUD operations ready
3. ✅ API endpoints ready
4. ⏳ Add to main app
5. ⏳ Test in Swagger
6. ⏳ WebSocket integration (optional)
7. ⏳ AI integration (optional)

Ready to test! 🚀
