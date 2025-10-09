# AI Agent - LangGraph ReAct Agent API

A production-ready AI agent built with LangGraph and LangChain that provides intelligent query processing through a FastAPI REST interface. The agent leverages multiple tools including web search, mathematical calculations, RAG (Retrieval-Augmented Generation), and custom API integrations to provide comprehensive and accurate responses.

## Features

- **🤖 ReAct Agent Architecture**: Implements the ReAct (Reasoning and Acting) pattern using LangGraph for improved decision-making
- **🔍 Web Search Integration**: Real-time web searches powered by SerpAPI
- **🧮 Mathematical Calculations**: Built-in calculator for complex mathematical expressions
- **📚 RAG Capability**: Vector-based document retrieval using Chroma DB for knowledge base queries
- **🔌 Custom API Integration**: Extensible tool for integrating external APIs
- **🔐 API Key Authentication**: Secure endpoint access with API key validation
- **⚡ Rate Limiting**: Token bucket-based rate limiting to prevent abuse
- **🔄 Resilient Execution**: Automatic retry mechanism with exponential backoff
- **📊 Logging**: Comprehensive logging for monitoring and debugging
- **💾 Persistent Vector Store**: ChromaDB for efficient document storage and retrieval

## Architecture

The application follows a modular architecture:

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│     FastAPI Server          │
│  - Auth & Rate Limiting     │
│  - Request Handling         │
└──────────┬──────────────────┘
           │
           ▼
┌──────────────────────────────┐
│    LangGraph ReAct Agent     │
│  - Reasoning Engine          │
│  - Tool Selection            │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│         Tools Layer          │
│  ┌────────────────────────┐  │
│  │  Web Search (SerpAPI)  │  │
│  ├────────────────────────┤  │
│  │  Calculator            │  │
│  ├────────────────────────┤  │
│  │  RAG Search (Chroma)   │  │
│  ├────────────────────────┤  │
│  │  Custom API            │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- SerpAPI key (for web search functionality)
- (Optional) Custom API endpoint for external integrations

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/IJ-s-lab/AI_agent.git
   cd AI_agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here

# Optional - with defaults
OPENAI_MODEL=gpt-4o-mini
AGENT_API_KEY=change-me
CHROMA_DB_DIR=./chroma_db
REQUESTS_PER_MINUTE=60
CUSTOM_API_URL=https://your-custom-api.com/endpoint
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for the chat model | Yes | - |
| `SERPAPI_API_KEY` | SerpAPI key for web search | Yes | - |
| `OPENAI_MODEL` | OpenAI model to use | No | `gpt-4o-mini` |
| `AGENT_API_KEY` | API key for authenticating requests | No | `change-me` |
| `CHROMA_DB_DIR` | Directory for ChromaDB persistence | No | `./chroma_db` |
| `REQUESTS_PER_MINUTE` | Rate limit for API requests | No | `60` |
| `CUSTOM_API_URL` | URL for custom API tool | No | - |

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8080`

### API Documentation

Once the server is running, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8080/docs

### API Endpoints

#### 1. Agent Query Endpoint

**POST** `/v1/agent/run`

Execute a query using the AI agent.

**Headers:**
```
X-API-Key: your_agent_api_key
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "What is the square root of 144?",
  "max_retries": 2
}
```

**Response:**
```json
{
  "answer": "The square root of 144 is 12."
}
```

**Example using cURL:**
```bash
curl -X POST "http://localhost:8080/v1/agent/run" \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in New York?"}'
```

**Example using Python:**
```python
import requests

url = "http://localhost:8080/v1/agent/run"
headers = {
    "X-API-Key": "change-me",
    "Content-Type": "application/json"
}
data = {
    "query": "Calculate the factorial of 5",
    "max_retries": 2
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

#### 2. Health Check Endpoint

**GET** `/health`

Check the health status of the service.

**Response:**
```json
{
  "status": "ok",
  "model": "gpt-4o-mini"
}
```

## Agent Tools

The agent has access to the following tools:

### 1. Web Search Tool
- **Function**: `web_search_tool(query: str)`
- **Purpose**: Performs real-time web searches using SerpAPI
- **Use Case**: Fetching current information, news, or any web-based data

### 2. Calculator Tool
- **Function**: `calculator_tool(expr: str)`
- **Purpose**: Evaluates mathematical expressions safely
- **Use Case**: Complex calculations, mathematical operations
- **Example**: `"sqrt(144) + 10 * 2"`

### 3. RAG Search Tool
- **Function**: `rag_search_tool(query: str, k: int = 3)`
- **Purpose**: Searches the vector database for relevant documents
- **Use Case**: Retrieving information from your knowledge base
- **Parameters**: 
  - `query`: Search query
  - `k`: Number of documents to retrieve (default: 3)

### 4. Custom API Tool
- **Function**: `custom_api_tool(payload: str)`
- **Purpose**: Integrates with external APIs
- **Use Case**: Extending functionality with custom integrations
- **Configuration**: Set `CUSTOM_API_URL` in `.env`

## Development

### Project Structure

```
AI_agent/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── .gitignore          # Git ignore rules
├── LICENSE             # GPL-3.0 License
├── chroma_db/          # Vector database storage (auto-created)
└── README.md           # This file
```

### Adding Documents to the Vector Store

To add documents to the RAG system, modify the `get_or_create_vectorstore` call in `main.py`:

```python
# Example documents
doc_texts = [
    "Your first document content here...",
    "Your second document content here...",
]

vectorstore = get_or_create_vectorstore(doc_texts=doc_texts)
```

### Extending with New Tools

To add a new tool:

1. Define the tool function:
```python
def my_custom_tool(input_param: str) -> str:
    try:
        # Your tool logic here
        result = process_input(input_param)
        return str(result)
    except Exception as e:
        logger.exception("my_custom_tool failed: %s", e)
        return f"[error] {str(e)}"
```

2. Register the tool:
```python
TOOLS = [web_search_tool, calculator_tool, rag_search_tool, custom_api_tool, my_custom_tool]
```

## Deployment

### Production Deployment

For production deployment, use a production-grade ASGI server:

```bash
# Using Gunicorn with Uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080
```

### Docker Deployment (Example)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:
```bash
docker build -t ai-agent .
docker run -p 8080:8080 --env-file .env ai-agent
```

### Security Considerations

- **Change the default API key**: Update `AGENT_API_KEY` in `.env`
- **Use HTTPS**: Deploy behind a reverse proxy (nginx, Caddy) with SSL/TLS
- **Implement proper authentication**: Consider OAuth2 or JWT for production
- **Rate limiting**: Adjust `REQUESTS_PER_MINUTE` based on your needs
- **Secrets management**: Use proper secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)
- **Input validation**: The agent includes basic input validation; enhance as needed
- **Monitor logs**: Set up proper log aggregation and monitoring

## Error Handling

The application includes comprehensive error handling:

- **Automatic retries**: Failed agent executions are automatically retried (configurable)
- **Exponential backoff**: Retry delays increase exponentially to handle transient failures
- **Graceful degradation**: Tools return error messages instead of crashing
- **Rate limiting**: Returns 429 status when rate limit is exceeded
- **Authentication errors**: Returns 401 for invalid API keys

## Logging

Logs are written to stdout with the format:
```
YYYY-MM-DD HH:MM:SS LEVEL message
```

Log levels:
- `INFO`: General operational messages
- `ERROR`: Error messages with stack traces

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [OpenAI](https://openai.com/)
- Web search via [SerpAPI](https://serpapi.com/)
- Vector storage with [ChromaDB](https://www.trychroma.com/)
- API framework: [FastAPI](https://fastapi.tiangolo.com/)
