# AI Tutor WebSocket Integration Guide

## Overview
Complete WebSocket integration for AI Tutor real-time conversations. Similar to chat system but optimized for pure AI tutoring without RAG/document search.

## WebSocket Endpoint

```
ws://localhost:8000/ai-tutor/ws/{conversation_id}
```

## Features

### ✅ What's Included:
- Real-time message streaming
- Typing indicators
- User message confirmations
- Auto-title generation
- Error handling
- Connection management
- Multi-user support per conversation

### ❌ What's NOT Included (Different from Chat):
- No RAG/document search
- No training document references
- No document-based context
- Pure AI tutoring only

## WebSocket Message Types

### 1. Connection Messages

#### Connected
```json
{
  "mt": "connected",
  "conversationId": "6507f1f77e1a2b3c4d5e6f7a",
  "message": "Connected to AI Tutor",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. User Messages

#### Send User Message
```json
{
  "mt": "user_message",
  "message": "Can you explain quadratic equations?",
  "userId": "123",
  "token": "msg_abc123",
  "timezone": "Asia/Kolkata",
  "languageCode": "en"
}
```

#### User Message Confirmation
```json
{
  "mt": "user_message_received",
  "conversationId": "6507f1f77e1a2b3c4d5e6f7a",
  "message": "Can you explain quadratic equations?",
  "userId": "123",
  "token": "msg_abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 3. Bot Response Streaming

#### Stream Start
```json
{
  "mt": "stream_start",
  "conversationId": "6507f1f77e1a2b3c4d5e6f7a",
  "messageId": "ai_tutor_response_1705318200.123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Stream Chunk
```json
{
  "mt": "stream_chunk",
  "conversationId": "6507f1f77e1a2b3c4d5e6f7a",
  "messageId": "ai_tutor_response_1705318200.123",
  "chunk": "Of course! ",
  "timestamp": "2024-01-15T10:30:00.123Z"
}
```

#### Stream End
```json
{
  "mt": "stream_end",
  "conversationId": "6507f1f77e1a2b3c4d5e6f7a",
  "messageId": "ai_tutor_response_1705318200.123",
  "fullText": "Of course! A quadratic equation is...",
  "isFinal": true,
  "timestamp": "2024-01-15T10:30:05Z"
}
```

### 4. Typing Indicators

#### Start Typing
```json
{
  "mt": "typing_indicator",
  "conversationId": "6507f1f77e1a2b3c4d5e6f7a",
  "isBot": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Stop Typing
```json
{
  "mt": "stop_typing_indicator",
  "conversationId": "6507f1f77e1a2b3c4d5e6f7a",
  "isBot": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 5. Error Messages

#### Error Response
```json
{
  "mt": "error",
  "conversationId": "6507f1f77e1a2b3c4d5e6f7a",
  "error": "Error message description",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Client Implementation

### JavaScript/TypeScript Example

```javascript
// Connect to WebSocket
const conversationId = "6507f1f77e1a2b3c4d5e6f7a";
const ws = new WebSocket(`ws://localhost:8000/ai-tutor/ws/${conversationId}`);

// Connection opened
ws.onopen = (event) => {
  console.log("Connected to AI Tutor");
};

// Receive messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.mt) {
    case "connected":
      console.log("Connection confirmed:", data.message);
      break;
      
    case "user_message_received":
      console.log("User message confirmed:", data.message);
      break;
      
    case "stream_start":
      console.log("Bot response starting...");
      // Initialize UI for streaming response
      break;
      
    case "stream_chunk":
      console.log("Chunk:", data.chunk);
      // Append chunk to UI
      appendToResponse(data.chunk);
      break;
      
    case "stream_end":
      console.log("Bot response complete:", data.fullText);
      // Finalize UI display
      break;
      
    case "typing_indicator":
      console.log("Bot is typing...");
      showTypingIndicator();
      break;
      
    case "stop_typing_indicator":
      console.log("Bot stopped typing");
      hideTypingIndicator();
      break;
      
    case "error":
      console.error("Error:", data.error);
      showError(data.error);
      break;
  }
};

// Send user message
function sendMessage(message, userId) {
  const messageData = {
    mt: "user_message",
    message: message,
    userId: userId,
    token: `msg_${Date.now()}`,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    languageCode: "en"
  };
  
  ws.send(JSON.stringify(messageData));
}

// Connection closed
ws.onclose = (event) => {
  console.log("WebSocket closed:", event);
};

// Connection error
ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};
```

### React Hook Example

```typescript
import { useEffect, useState, useRef } from 'react';

export const useAITutor = (conversationId: string) => {
  const [messages, setMessages] = useState<any[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState("");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket(`ws://localhost:8000/ai-tutor/ws/${conversationId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log("Connected to AI Tutor");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch(data.mt) {
        case "stream_start":
          setIsTyping(true);
          setStreamingMessage("");
          break;
          
        case "stream_chunk":
          setStreamingMessage(prev => prev + data.chunk);
          break;
          
        case "stream_end":
          setIsTyping(false);
          setMessages(prev => [...prev, {
            id: data.messageId,
            message: data.fullText,
            isBot: true,
            timestamp: data.timestamp
          }]);
          setStreamingMessage("");
          break;
          
        case "user_message_received":
          setMessages(prev => [...prev, {
            message: data.message,
            isBot: false,
            timestamp: data.timestamp
          }]);
          break;
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log("Disconnected from AI Tutor");
    };

    return () => {
      ws.close();
    };
  }, [conversationId]);

  const sendMessage = (message: string, userId: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        mt: "user_message",
        message,
        userId,
        token: `msg_${Date.now()}`,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        languageCode: "en"
      }));
    }
  };

  return {
    messages,
    isConnected,
    isTyping,
    streamingMessage,
    sendMessage
  };
};
```

## Complete Flow Example

### 1. Create Conversation
```bash
POST /ai-tutor/conversations
{
  "user_id": 123,
  "title": "",
  "status": "active"
}

Response: { "conversation_id": "6507f1f77e1a2b3c4d5e6f7a" }
```

### 2. Connect WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ai-tutor/ws/6507f1f77e1a2b3c4d5e6f7a');
```

### 3. Send User Message
```javascript
ws.send(JSON.stringify({
  mt: "user_message",
  message: "Can you explain photosynthesis?",
  userId: "123",
  token: "msg_1705318200123",
  timezone: "Asia/Kolkata",
  languageCode: "en"
}));
```

### 4. Receive Bot Response (Streaming)
```javascript
// You'll receive multiple messages:
// 1. stream_start
// 2. stream_chunk (multiple times)
// 3. stream_end (with full text)
```

## Testing with Postman/Insomnia

### WebSocket Testing Steps:

1. **Open WebSocket Connection:**
   ```
   ws://localhost:8000/ai-tutor/ws/YOUR_CONVERSATION_ID
   ```

2. **Wait for Connected Message:**
   ```json
   {"mt": "connected", "conversationId": "...", "message": "Connected to AI Tutor"}
   ```

3. **Send Test Message:**
   ```json
   {
     "mt": "user_message",
     "message": "What is photosynthesis?",
     "userId": "123",
     "token": "test123",
     "timezone": "UTC",
     "languageCode": "en"
   }
   ```

4. **Observe Response:**
   - User message confirmation
   - Typing indicator
   - Stream start
   - Multiple stream chunks
   - Stream end
   - Stop typing indicator

## Error Handling

### Common Errors:

1. **Invalid JSON:**
```json
{
  "mt": "error",
  "error": "Invalid JSON format",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

2. **Message Processing Error:**
```json
{
  "mt": "error",
  "error": "Error message description",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Key Differences from Chat WebSocket

### Chat System:
- ✅ RAG/document search
- ✅ Training document context
- ✅ Document-based responses
- Uses `chatId` in messages

### AI Tutor System:
- ✅ Pure AI conversation
- ✅ Web search integration
- ✅ General knowledge tutoring
- Uses `conversationId` in messages
- No document context

## Performance Considerations

1. **Connection Pooling:** WebSocket manager handles multiple connections per conversation
2. **Message Deduplication:** Prevents duplicate messages within 5 seconds
3. **Streaming:** Real-time token streaming for better UX
4. **Auto-cleanup:** Dead connections are automatically removed
5. **Error Recovery:** Graceful error handling doesn't crash the connection

## Production Deployment

### Environment Variables:
```bash
# Already configured in your system
TAVILY_API_KEY=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
```

### WebSocket URL (Production):
```
wss://your-domain.com/ai-tutor/ws/{conversation_id}
```

## Next Steps

1. ✅ Collections created
2. ✅ Repository layer ready
3. ✅ API endpoints ready
4. ✅ WebSocket integration complete
5. ✅ Bot handler without RAG ready
6. ⏳ Test WebSocket connection
7. ⏳ Integrate with main app
8. ⏳ Test complete flow

The AI Tutor system is complete and ready for testing! 🚀
