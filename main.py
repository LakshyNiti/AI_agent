import os
import time
import logging
import asyncio
import math
import requests
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory

# Safety: please pin versions in production

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "change-me")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RETRIES = 2
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "60"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to .env")

# Logging
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# Rate limiting (very small in-memory token bucket for demo)
class TokenBucket:
    def __init__(self, rate_per_minute: int):
        self.capacity = rate_per_minute
        self.tokens = rate_per_minute
        self.fill_rate = rate_per_minute / 60.0
        self.timestamp = time.time()

    def consume(self, amount=1):
        now = time.time()
        elapsed = now - self.timestamp
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

rate_limiter = TokenBucket(REQUESTS_PER_MINUTE)

# Embeddings + Vectorstore (Chroma)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create or load a Chroma vectorstore (persistent in CHROMA_DB_DIR)
def get_or_create_vectorstore(doc_texts: List[str] = None):
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        logger.info("Loading existing Chroma DB from %s", CHROMA_DB_DIR)
        vect = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        return vect

    logger.info("Creating new Chroma DB at %s", CHROMA_DB_DIR)
    # The Chroma client needs to be initialized to create a new collection
    vect = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    if doc_texts:
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for i, t in enumerate(doc_texts):
            splits = text_splitter.split_text(t)
            docs.extend([{"page_content": s, "metadata": {"source": f"initial_doc_{i}"}} for s in splits])
        vect.add_documents(docs)
        vect.persist()
    return vect

# Example: pre-load domain docs if you have any
# For demo, this could be left empty; in prod, load relevant PDF/texts and index them.
vectorstore = get_or_create_vectorstore(doc_texts=None)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the chat model
model = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0, openai_api_key=OPENAI_API_KEY)

# Define tools
# 1) Web search tool using SerpAPI
def web_search_tool(query: str) -> str:
    try:
        wrapper = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
        return wrapper.run(query)
    except Exception as e:
        logger.exception("web_search_tool failed: %s", e)
        return f"[web_search_error] {str(e)}"

# 2) Simple math tool
def calculator_tool(expr: str) -> str:
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        result = eval(expr, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        logger.exception("calculator_tool error: %s", e)
        return f"[calculator_error] {str(e)}"

# 3) RAG retriever tool
def rag_search_tool(query: str, k: int = 3) -> str:
    try:
        if not vectorstore:
            return "[rag_error] No vectorstore available"
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return "[rag] No relevant documents found."
        summary = "\n---\n".join([f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs])
        return summary
    except Exception as e:
        logger.exception("rag_search_tool failed: %s", e)
        return f"[rag_error] {str(e)}"

# 4) Custom API tool example
def custom_api_tool(payload: str) -> str:
    try:
        url = os.getenv("CUSTOM_API_URL")
        if not url:
            return "[custom_api_error] CUSTOM_API_URL not configured"
        r = requests.post(url, json={"query": payload}, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.exception("custom_api_tool failed: %s", e)
        return f"[custom_api_error] {str(e)}"

# Register tools
TOOLS = [web_search_tool, calculator_tool, rag_search_tool, custom_api_tool]

# Create the ReAct agent
system_message = SystemMessage(
    content="You are a helpful, concise assistant. When useful, call the provided tools. Only call tools for factual lookups, math, or retrieval from the knowledge base."
)
agent = create_react_agent(model=model, tools=TOOLS, messages_modifier=system_message)


# Wrap agent usage in resilient helper
async def run_agent(user_input: str, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            # Build the inputs in LangGraph-compatible shape
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # This is the corrected block with proper indentation
            if hasattr(agent, "invoke"):
                result = agent.invoke(inputs)
            else:
                raise RuntimeError("Agent object missing 'invoke' method")

            return {"ok": True, "result": result}
        except Exception as e:
            last_err = e
            logger.exception("Agent run failed on attempt %s: %s", attempt, e)
            # backoff
            await asyncio.sleep(1 + attempt * 2)
            continue
    return {"ok": False, "error": str(last_err)}


# FastAPI server
app = FastAPI(title="LangGraph ReAct Agent", docs_url="/docs")


class QueryIn(BaseModel):
    query: str
    max_retries: int = 2


# Simple auth dependency
async def check_api_key(x_api_key: str = Header(None)):
    if x_api_key != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/v1/agent/run")
async def agent_run(payload: QueryIn, request: Request, x_api_key: str = Header(None)):
    # auth
    await check_api_key(x_api_key)

    # rate limit
    if not rate_limiter.consume():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    q = payload.query
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # Run agent
    out = await run_agent(q, max_retries=payload.max_retries)
    if not out.get("ok"):
        raise HTTPException(status_code=500, detail=out.get("error", "unknown"))
    
    # Extract the final answer from the agent's output
    final_answer = out.get("result", {}).get("messages", [])[-1].content
    return {"answer": final_answer}


# Lightweight healthcheck
@app.get("/health")
async def health():
    return {"status": "ok", "model": OPENAI_MODEL}


# CLI convenience
if __name__ == "__main__":
    import uvicorn
    # In production, run via gunicorn/uvicorn workers behind a reverse proxy
    # Changed "react_agent_production_ready:app" to "main:app" to match your filename
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)