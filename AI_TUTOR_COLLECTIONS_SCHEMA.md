# AI Tutor Collections Schema

## Overview
This document outlines the MongoDB collection structure for AI Tutor conversations in the VaktaAI system. Unlike the chat system, AI Tutor conversations don't have document references - they are standalone conversations between user and AI.

## Collection Structure

### 1. CONVERSATIONS Collection
**Collection Name:** `conversations`
**Purpose:** Store AI Tutor conversation sessions

#### Required Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `_id` | ObjectId | Primary key | Auto-generated |
| `user_id` | Integer | User who owns the conversation | Required, Indexed |
| `title` | String | Conversation title | Required |
| `status` | String | Conversation status | Values: active/archived, Default: active |
| `created_at` | DateTime | Creation timestamp | Auto-generated, Indexed |
| `updated_at` | DateTime | Last update timestamp | Auto-updated |

#### Optional Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `subject` | String | Subject/Topic of conversation | Nullable |
| `tags` | Array[String] | Tags for categorization | Nullable |
| `metadata` | Object | Additional metadata | Nullable |

### 2. CONVERSATION_MESSAGES Collection
**Collection Name:** `conversation_messages`
**Purpose:** Store messages within AI Tutor conversations

#### Required Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `_id` | ObjectId | Primary key | Auto-generated |
| `conversation_id` | ObjectId | Reference to conversations | Required, Indexed |
| `message` | String | Message content | Required |
| `is_bot` | Boolean | Whether message is from AI | Default: false |
| `created_ts` | DateTime | Creation timestamp | Auto-generated, Indexed |
| `updated_ts` | DateTime | Last update timestamp | Auto-updated |

#### Optional Fields
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `reaction` | String | User reaction (like/dislike) | Nullable |
| `token` | String | Message token/identifier | Nullable |
| `type` | String | Message type | Default: "text" |
| `is_edited` | Boolean | Whether message was edited | Default: false |
| `metadata` | Object | Additional metadata | Nullable |

## Database Indexes

### Conversations Collection Indexes
- `user_id` (single field)
- `status` (single field)
- `created_at` (single field)
- `updated_at` (single field)
- `(user_id, created_at)` (compound index, descending on created_at)

### Conversation Messages Collection Indexes
- `conversation_id` (single field)
- `is_bot` (single field)
- `created_ts` (single field)
- `(conversation_id, created_ts)` (compound index, ascending on created_ts)

## CRUD Operations Available

### Conversations Collection Operations

#### Basic CRUD
- `create_conversation(db, conversation_doc)` - Create new conversation
- `get_conversation(db, conversation_id)` - Get conversation by ID
- `get_user_conversations(db, user_id, limit, skip)` - Get user's conversations with pagination
- `get_user_conversations_count(db, user_id)` - Get total conversation count
- `update_conversation(db, conversation_id, fields)` - Update conversation
- `delete_conversation(db, conversation_id)` - Delete conversation and all messages

#### Advanced Queries
- `get_conversations_by_status(db, user_id, status, limit, skip)` - Filter by status
- `get_conversations_with_message_counts(db, user_id, limit, skip)` - Get with message counts
- `search_conversations(db, user_id, search_query, limit)` - Search by title
- `get_recent_active_conversations(db, user_id, days, limit)` - Get recently active

### Conversation Messages Collection Operations

#### Basic CRUD
- `create_conversation_message(db, message_doc)` - Create new message
- `get_conversation_messages(db, conversation_id, limit, skip)` - Get messages with pagination
- `get_conversation_message(db, message_id)` - Get message by ID
- `update_conversation_message(db, message_id, fields)` - Update message
- `delete_conversation_message(db, message_id)` - Delete message

#### Advanced Queries
- `get_conversation_message_count(db, conversation_id)` - Get message count
- `get_last_conversation_messages(db, conversation_id, limit)` - Get recent messages
- `delete_all_conversation_messages(db, conversation_id)` - Clear all messages
- `get_user_messages_count(db, user_id)` - Get total messages across all conversations

### Bulk Operations
- `bulk_create_conversation_messages(db, messages)` - Create multiple messages
- `bulk_delete_conversations(db, conversation_ids)` - Delete multiple conversations

### Statistics
- `get_conversation_stats(db, user_id)` - Get comprehensive statistics

## Usage Examples

### Creating a New Conversation
```python
from datetime import datetime
from app.database.ai_tutor_collections import create_conversation

conversation_doc = {
    "user_id": 123,
    "title": "Math Help - Quadratic Equations",
    "status": "active",
    "subject": "Mathematics",
    "tags": ["math", "algebra", "quadratic"],
    "created_at": datetime.utcnow(),
    "updated_at": datetime.utcnow()
}

conversation_id = create_conversation(db, conversation_doc)
print(f"Created conversation: {conversation_id}")
```

### Adding Messages to Conversation
```python
from datetime import datetime
from app.database.ai_tutor_collections import create_conversation_message

# User message
user_message = {
    "conversation_id": ObjectId(conversation_id),
    "message": "Can you help me understand quadratic equations?",
    "is_bot": False,
    "type": "text",
    "created_ts": datetime.utcnow(),
    "updated_ts": datetime.utcnow()
}

user_msg_id = create_conversation_message(db, user_message)

# Bot response
bot_message = {
    "conversation_id": ObjectId(conversation_id),
    "message": "Of course! A quadratic equation is...",
    "is_bot": True,
    "type": "text",
    "created_ts": datetime.utcnow(),
    "updated_ts": datetime.utcnow()
}

bot_msg_id = create_conversation_message(db, bot_message)
```

### Getting User's Conversations
```python
from app.database.ai_tutor_collections import get_user_conversations

conversations = get_user_conversations(db, user_id=123, limit=20, skip=0)

for conv in conversations:
    print(f"ID: {conv['_id']}, Title: {conv['title']}, Status: {conv['status']}")
```

### Getting Conversation Messages
```python
from app.database.ai_tutor_collections import get_conversation_messages

messages = get_conversation_messages(db, conversation_id, limit=50, skip=0)

for msg in messages:
    sender = "Bot" if msg["is_bot"] else "User"
    print(f"{sender}: {msg['message']}")
```

### Getting Statistics
```python
from app.database.ai_tutor_collections import get_conversation_stats

stats = get_conversation_stats(db, user_id=123)
print(f"Total Conversations: {stats['total_conversations']}")
print(f"Active: {stats['active_conversations']}")
print(f"Total Messages: {stats['total_messages']}")
print(f"Bot Messages: {stats['bot_messages']}")
print(f"User Messages: {stats['user_messages']}")
```

### Searching Conversations
```python
from app.database.ai_tutor_collections import search_conversations

results = search_conversations(db, user_id=123, search_query="math", limit=10)

for conv in results:
    print(f"Found: {conv['title']}")
```

### Deleting a Conversation
```python
from app.database.ai_tutor_collections import delete_conversation

# This will also delete all messages in the conversation
delete_conversation(db, conversation_id)
print("Conversation and all messages deleted")
```

## Differences from Chat System

### What's Removed:
- ❌ No `document_id` field
- ❌ No `training_doc_id` field
- ❌ No document references

### What's Similar:
- ✅ User-based conversations
- ✅ Message structure (is_bot, reactions, etc.)
- ✅ Status management (active/archived)
- ✅ Pagination support
- ✅ Timestamp tracking

## Best Practices

### 1. Always Update `updated_at`
When modifying conversations or messages, the `updated_at` field is automatically updated by the functions.

### 2. Use Pagination
Always use limit and skip parameters for large result sets:
```python
conversations = get_user_conversations(db, user_id, limit=20, skip=0)
```

### 3. Clean Up Old Conversations
Periodically archive or delete old conversations:
```python
# Archive old conversations
from datetime import datetime, timedelta
old_date = datetime.utcnow() - timedelta(days=90)

old_conversations = conversations.find({
    "user_id": user_id,
    "updated_at": {"$lt": old_date}
})

for conv in old_conversations:
    update_conversation(db, str(conv["_id"]), {"status": "archived"})
```

### 4. Use Compound Indexes
The compound indexes are optimized for common queries:
```python
# This query uses the (user_id, created_at) compound index
conversations.find({"user_id": 123}).sort("created_at", -1)
```

### 5. Batch Operations
For bulk operations, use the bulk functions:
```python
messages = [
    {"conversation_id": conv_id, "message": "msg1", "is_bot": False, ...},
    {"conversation_id": conv_id, "message": "msg2", "is_bot": True, ...}
]
bulk_create_conversation_messages(db, messages)
```

## Performance Considerations

1. **Indexes**: All critical fields are indexed for fast queries
2. **Pagination**: Always use limit/skip to avoid loading too much data
3. **Message Cleanup**: Consider implementing message retention policies
4. **Statistics**: Use the built-in stats function instead of manual counting
5. **Bulk Operations**: Use bulk functions when creating/deleting multiple items

## Migration from Chat System

If migrating data from the chat system:

```python
# Example migration
from app.database.mongo_collections import get_user_chats, get_chat_messages

# Get old chats
old_chats = get_user_chats(db, user_id)

for chat in old_chats:
    # Create new conversation (without document references)
    new_conv = {
        "user_id": chat["user_id"],
        "title": chat["title"],
        "status": chat["status"],
        "created_at": chat["created_at"],
        "updated_at": chat["updated_at"]
    }
    new_conv_id = create_conversation(db, new_conv)
    
    # Migrate messages
    old_messages = get_chat_messages(db, str(chat["_id"]))
    for msg in old_messages:
        new_msg = {
            "conversation_id": ObjectId(new_conv_id),
            "message": msg["message"],
            "is_bot": msg["is_bot"],
            "reaction": msg.get("reaction"),
            "token": msg.get("token"),
            "type": msg.get("type", "text"),
            "is_edited": msg.get("is_edited", False),
            "created_ts": msg["created_ts"],
            "updated_ts": msg["updated_ts"]
        }
        create_conversation_message(db, new_msg)
```

## Next Steps

1. **Create Repository Layer**: Build repository class for business logic
2. **Create API Endpoints**: Build FastAPI routes for conversations
3. **Add WebSocket Support**: Implement real-time messaging
4. **Add AI Integration**: Connect to AI service for bot responses
5. **Add Analytics**: Track conversation metrics and patterns
