from app.avatar_config import ai_settings
from langchain_community.tools import DuckDuckGoSearchRun
import asyncio
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
# Note: ChatGroq wrapper may be available as a separate package; validate import path.
try:
    from langchain_groq import ChatGroq
    has_groq = True
except Exception:
    ChatGroq = None
    has_groq = False


# Create LLM (fall back to small OpenAI/ChatOpenAI if Groq wrapper is unavailable)
if has_groq and ai_settings.GROQ_API_KEY:
    llm = ChatGroq(api_key=ai_settings.GROQ_API_KEY, model_name=ai_settings.GROQ_MODEL, streaming=True)
elif ai_settings.OPENAI_API_KEY:
    # Fallback to OpenAI with API key
    llm = ChatOpenAI(
        temperature=0.5, 
        model_name=ai_settings.OPENAI_MODEL,
        openai_api_key=ai_settings.OPENAI_API_KEY
    )
else:
    # No API keys available - raise an error
    raise ValueError("No API keys configured. Please set either GROQ_API_KEY or OPENAI_API_KEY environment variable.")


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=2000)


# Example tool using DuckDuckGo (langchain-community)
search_tool = DuckDuckGoSearchRun()


def rag_search(query: str) -> str:
    return "Placeholder: run vector search and return summarized result"


tools = [
# Tool(name="knowledge_base", func=rag_search, description="Search local curriculum"),
Tool(name="web_search", func=search_tool.run, description="Web search if needed")
]


agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=False)


async def run_agent_streaming(user_input: str, on_new_token):
    """Run agent in streaming mode and call on_new_token(token) for each text token.
    on_new_token can be a coroutine.
    """
    # This function is intentionally generic: actual streaming depends on LLM wrapper
    # For wrappers that support 'stream' yield, iterate and call callback.
    # For non-streaming LLM, use synchronous run and then chunk the text.


    # Example fallback (non-streaming):
    result = agent.run(user_input)
    # naive chunking
    for i in range(0, len(result), 200):
        token_chunk = result[i:i+200]
        await on_new_token(token_chunk)
        await asyncio.sleep(0)