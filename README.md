# AI Chatbot app

A FastAPI app with WebSocket support for real-time AI chat using OpenAI and LangChain.

## Setup

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Create `.env` file:
\`\`\`bash
cp .env.example .env
\`\`\`

3. Add your OpenAI API key to `.env`:
\`\`\`
OPENAI_API_KEY=your_actual_api_key_here
\`\`\`

4. Run the server:
\`\`\`bash
python main.py
\`\`\`

The server will start on `http://localhost:8000`

## Features

- WebSocket connections for real-time chat
- OpenAI integration via LangChain
- Conversation memory and context
- CORS support for frontend integration
- Health check endpoint
- Conversation reset functionality

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `WebSocket /ws/{client_id}` - WebSocket connection for chat
- `POST /reset/{client_id}` - Reset conversation for client
