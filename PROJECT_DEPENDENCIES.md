# 📦 VaktaAi Backend - Complete Dependencies Documentation

> **Project**: VaktaAi - AI-Powered Document Chat & Learning Platform  
> **Last Updated**: October 3, 2025

---

## 🎯 Project Overview
VaktaAi is an intelligent document processing and chat system that uses RAG (Retrieval-Augmented Generation) to help users interact with their documents, generate summaries, create study notes, and take AI-generated quizzes.

---

## 📚 Table of Contents
1. [Web Framework & Server](#-web-framework--server)
2. [AI & Machine Learning](#-ai--machine-learning)
3. [Database & Storage](#-database--storage)
4. [Document Processing](#-document-processing)
5. [HTTP & Network](#-http--network)
6. [Security & Authentication](#-security--authentication)
7. [Translation & Language](#-translation--language)
8. [Data Processing & Search](#-data-processing--search)
9. [YouTube & Media](#-youtube--media)
10. [Monitoring & Logging](#-monitoring--logging)

---

## 🌐 Web Framework & Server

### **fastapi**
- **Version**: Latest
- **Purpose**: Main web framework for building RESTful APIs
- **Used For**:
  - All API endpoints (`/api/v1/docSathi/...`, `/api/v1/chat/...`)
  - WebSocket connections for real-time chat
  - Request/response validation
  - Dependency injection
- **Key Files**: `VaktaAi/app/main.py`, `VaktaAi/app/features/*/router.py`

### **uvicorn**
- **Purpose**: ASGI web server
- **Used For**:
  - Running FastAPI application
  - Handling async requests
  - WebSocket support
- **Command**: `uvicorn app.main:app --reload`

### **websockets**
- **Purpose**: WebSocket protocol implementation
- **Used For**:
  - Real-time streaming chat responses
  - Live document processing updates
  - Bidirectional client-server communication
- **Key Files**: `VaktaAi/app/features/chat/websocket_manager.py`

### **python-multipart**
- **Purpose**: Parse multipart form data
- **Used For**:
  - File uploads (PDF, DOCX, PPTX)
  - Document upload endpoint
- **Endpoint**: `POST /upload-document`

### **asgiref**
- **Purpose**: ASGI specification utilities
- **Used For**:
  - Async to sync conversion helpers
  - ASGI middleware support

---

## 🤖 AI & Machine Learning

### **openai**
- **Purpose**: OpenAI API client (GPT-3.5-turbo, GPT-4)
- **Used For**:
  - ✅ **Chat responses** - Answering user queries with context
  - ✅ **Document summaries** - Generating document overviews with titles
  - ✅ **Study notes** - Creating structured notes from documents
  - ✅ **Quiz generation** - Creating MCQs and True/False questions
  - ✅ **Translation** - Translating responses to user's language (Hindi, English, etc.)
  - ✅ **Document keypoints** - Extracting main points from content
- **Key Files**: 
  - `VaktaAi/app/utils/openai/openai_client.py`
  - `VaktaAi/app/features/chat/bot_handler.py`
  - `VaktaAi/app/features/docSathi/repository.py`

### **langchain**
- **Purpose**: Framework for building LLM applications
- **Used For**:
  - RAG pipeline orchestration
  - Chain of thought prompting
  - Document retrieval workflows
  - Memory management in conversations
- **Key Components**: Chains, Agents, Memory

### **langchain-core**
- **Purpose**: Core LangChain abstractions
- **Used For**:
  - Base classes for custom chains
  - Document loaders
  - Output parsers

### **langchain_openai**
- **Purpose**: LangChain integration with OpenAI
- **Used For**:
  - OpenAI embeddings generation
  - GPT model wrappers
  - Chat model integration

### **langchain-community**
- **Purpose**: Community-contributed LangChain components
- **Used For**:
  - Additional document loaders
  - Vector store integrations
  - Utility tools

### **langchain-text-splitters**
- **Purpose**: Advanced text splitting strategies
- **Used For**:
  - Splitting documents into chunks for vector storage
  - Recursive character text splitter
  - Token-based splitting for optimal chunk sizes
- **Key Files**: `services/text_chunker.py`

### **sentence-transformers**
- **Purpose**: Generate sentence embeddings
- **Used For**:
  - Converting text chunks to vectors (384-dimensional)
  - Semantic search in Milvus vector database
  - Model: `all-MiniLM-L6-v2`
- **Key Files**: `VaktaAi/app/utils/transformers/embedding.py`

### **transformers**
- **Purpose**: HuggingFace transformers library
- **Used For**:
  - Loading pre-trained models
  - Tokenization
  - Model inference

### **huggingface-hub**
- **Purpose**: HuggingFace model hub client
- **Used For**:
  - Downloading models from HuggingFace
  - Model caching

### **torch**
- **Purpose**: PyTorch deep learning framework
- **Used For**:
  - Running sentence-transformers models
  - Tensor operations for embeddings

### **tokenizers**
- **Purpose**: Fast tokenization library
- **Used For**:
  - Efficient text tokenization
  - HuggingFace tokenizer backend

### **tiktoken**
- **Purpose**: OpenAI's tokenizer
- **Used For**:
  - Counting tokens for OpenAI API calls
  - Ensuring prompts stay within token limits
  - Cost estimation

---

## 🗄️ Database & Storage

### **pymongo**
- **Purpose**: MongoDB Python driver
- **Used For**:
  - Storing user chats (`chats` collection)
  - Storing chat messages (`chat_messages` collection)
  - Storing documents metadata (`docSathi_ai_documents` collection)
  - Storing document chunks (`docSathi_ai_document_chunks` collection)
  - Storing student quizzes (`student_quizs` collection)
  - Storing quiz questions (`question_answers` collection)
- **Key Files**: `VaktaAi/app/database/mongo_collections.py`
- **Connection**: MongoDB Atlas or local MongoDB

### **pymilvus**
- **Purpose**: Milvus vector database client
- **Used For**:
  - Storing document chunk embeddings
  - Semantic search across documents
  - Vector similarity search for RAG
  - Collection: `chunk_msmarcos`
- **Key Files**: 
  - `VaktaAi/app/utils/milvus/milvus_manager.py`
  - `services/milvus_service.py`

### **redis**
- **Purpose**: In-memory data store
- **Used For**:
  - Caching frequently accessed data
  - Session management (if implemented)
  - Rate limiting

### **boto3**
- **Purpose**: AWS SDK for Python
- **Used For**:
  - Uploading documents to S3 bucket
  - Downloading documents from S3
  - Managing S3 objects
- **Bucket**: `xr-technolab-vakta-ai`
- **Key Files**: `VaktaAi/app/aws/s3_service.py`

### **botocore**
- **Purpose**: Low-level AWS functionality
- **Used For**: Core AWS service interactions (boto3 dependency)

### **s3transfer**
- **Purpose**: S3 transfer manager
- **Used For**: Efficient multipart uploads/downloads

### **SQLAlchemy**
- **Purpose**: SQL toolkit and ORM
- **Used For**: Database abstraction layer (dependency for other packages)

---

## 📄 Document Processing

### **pypdf2**
- **Purpose**: PDF file parser
- **Used For**:
  - Extracting text from PDF files
  - Reading PDF metadata
- **Supported**: `.pdf` files

### **PyMuPDF** (fitz)
- **Purpose**: Advanced PDF processing
- **Used For**:
  - Better text extraction than pypdf2
  - Handling complex PDF layouts
  - Primary PDF parser in the system
- **Key Files**: `services/file_processor.py`

### **python-docx**
- **Purpose**: Microsoft Word document parser
- **Used For**:
  - Reading `.docx` files
  - Extracting paragraphs and tables
- **Supported**: `.docx` files

### **docx2python**
- **Purpose**: Alternative DOCX parser
- **Used For**:
  - Backup DOCX parsing method
  - Better handling of complex documents

### **python-pptx**
- **Purpose**: PowerPoint file parser
- **Used For**:
  - Reading `.pptx` files
  - Extracting slide text and notes
- **Supported**: `.pptx` files
- **Key Files**: `services/file_processor.py`

### **XlsxWriter**
- **Purpose**: Excel file writer
- **Used For**: Creating Excel reports (if needed in future)

---

## 🌐 HTTP & Network

### **httpx**
- **Purpose**: Modern async HTTP client
- **Used For**:
  - Making external API calls
  - Web search integration (Tavily Search)
  - Async HTTP requests
- **Key Files**: `VaktaAi/app/features/chat/bot_handler.py`

### **httpcore**
- **Purpose**: Low-level HTTP implementation
- **Used For**: Core HTTP functionality (httpx dependency)

### **httpx-sse**
- **Purpose**: Server-sent events support
- **Used For**: Streaming responses from external APIs

### **requests**
- **Purpose**: Simple HTTP library
- **Used For**:
  - Synchronous HTTP requests
  - YouTube API calls
  - External service integrations

### **requests-toolbelt**
- **Purpose**: Advanced request utilities
- **Used For**: Multipart encoding, streaming uploads

### **urllib3**
- **Purpose**: HTTP client library
- **Used For**: Connection pooling (requests dependency)

### **aiohttp**
- **Purpose**: Async HTTP client/server
- **Used For**:
  - Asynchronous API calls
  - Concurrent requests

### **aiofiles**
- **Purpose**: Async file I/O
- **Used For**:
  - Non-blocking file reads/writes
  - Processing uploaded documents asynchronously
- **Key Files**: File upload handlers

### **aiosignal**
- **Purpose**: Signal handling for async
- **Used For**: aiohttp dependency

### **aiohappyeyeballs**
- **Purpose**: Fast async DNS resolution
- **Used For**: Network performance optimization

---

## 🔐 Security & Authentication

### **python-jose**
- **Purpose**: JWT token implementation
- **Used For**:
  - Creating JWT tokens for user authentication
  - Verifying token signatures
  - Token expiration handling
- **Algorithm**: HS256

### **python-dateutil**
- **Purpose**: Date parsing utilities
- **Used For**:
  - Parsing ISO date formats
  - Token expiry calculations

### **ecdsa**
- **Purpose**: Elliptic curve cryptography
- **Used For**: Digital signatures (jose dependency)

### **rsa**
- **Purpose**: RSA encryption/decryption
- **Used For**: Public key cryptography (jose dependency)

### **pyasn1**
- **Purpose**: ASN.1 parser
- **Used For**: Certificate parsing (rsa dependency)

### **certifi**
- **Purpose**: Root SSL certificates
- **Used For**: HTTPS connection verification

---

## 🌍 Translation & Language

### **deep-translator**
- **Purpose**: Translation library (Google Translate API)
- **Used For**:
  - ✅ **Translating bot responses** to user's language (Hindi, Hinglish, etc.)
  - Chunking long text for translation (2000 chars per chunk)
  - Multi-language support
- **Languages**: English, Hindi, and more
- **Key Files**: `VaktaAi/app/features/chat/bot_handler.py`
- **Why this**: Replaced `translate` library due to character limits

### **indic-transliteration**
- **Purpose**: Transliteration for Indian languages
- **Used For**:
  - Converting between Devanagari and Latin scripts
  - Hinglish text processing
- **Key Files**: `VaktaAi/app/youtubeService/*_processor.py`

### **nltk**
- **Purpose**: Natural Language Toolkit
- **Used For**:
  - Text tokenization for BM25 ranking
  - Stopword removal
  - Sentence splitting
- **Key Files**: `VaktaAi/app/features/chat/bot_handler.py`
- **Required Downloads**: `punkt`, `stopwords`

---

## 📊 Data Processing & Search

### **numpy**
- **Purpose**: Numerical computing library
- **Used For**:
  - Vector operations for embeddings
  - Array manipulations
  - Mathematical operations

### **pandas**
- **Purpose**: Data analysis library
- **Used For**:
  - Data manipulation (if used)
  - CSV processing

### **scipy**
- **Purpose**: Scientific computing
- **Used For**:
  - Statistical operations
  - Advanced mathematical functions

### **scikit-learn**
- **Purpose**: Machine learning library
- **Used For**:
  - Preprocessing utilities
  - Similarity calculations
  - Model utilities (dependency for sentence-transformers)

### **rank-bm25**
- **Purpose**: BM25 ranking algorithm
- **Used For**:
  - ✅ **Keyword-based document ranking** alongside semantic search
  - Hybrid search (semantic + keyword)
  - Improving search relevance
- **Key Files**: `VaktaAi/app/features/chat/bot_handler.py`

### **joblib**
- **Purpose**: Parallel processing utilities
- **Used For**: Caching, parallel computations

### **networkx**
- **Purpose**: Graph algorithms
- **Used For**: Network analysis (dependency)

---

## 📹 YouTube & Media

### **youtube-transcript-api**
- **Purpose**: YouTube transcript downloader
- **Used For**:
  - ✅ **Downloading YouTube video captions**
  - Extracting subtitles in multiple languages
  - Processing video content for RAG
- **Key Files**: `VaktaAi/app/youtubeService/*.py`

### **yt-dlp**
- **Purpose**: YouTube video/audio downloader
- **Used For**:
  - Downloading YouTube videos
  - Extracting metadata
  - Backup method for transcript extraction
- **Key Files**: `VaktaAi/app/youtubeService/powerful_transcript_extractor.py`

---

## 📝 Data Validation & Serialization

### **pydantic**
- **Purpose**: Data validation using Python type hints
- **Used For**:
  - ✅ **Request/response schemas** for all API endpoints
  - Automatic validation
  - JSON serialization
- **Key Files**: 
  - `VaktaAi/app/features/chat/schemas.py`
  - `VaktaAi/app/features/docSathi/schema.py`
  - `VaktaAi/app/schemas/schemas.py`

### **pydantic-core**
- **Purpose**: Core validation logic
- **Used For**: Fast validation engine (pydantic dependency)

### **pydantic-settings**
- **Purpose**: Settings management
- **Used For**:
  - Loading configuration from environment variables
  - `.env` file parsing
- **Key Files**: `VaktaAi/app/config/settings.py`

### **marshmallow**
- **Purpose**: Object serialization/deserialization
- **Used For**: Alternative serialization (dependency)

### **dataclasses-json**
- **Purpose**: JSON support for dataclasses
- **Used For**: Dataclass to JSON conversion

### **jsonpatch** / **jsonpointer**
- **Purpose**: JSON operations
- **Used For**: JSON path navigation and patching

### **ujson** / **orjson**
- **Purpose**: Fast JSON parsing
- **Used For**: High-performance JSON operations

---

## 🔧 Configuration & Environment

### **python-dotenv** / **dotenv**
- **Purpose**: Load environment variables
- **Used For**:
  - Reading `.env` file
  - Configuration management
  - API keys, database URLs
- **Key Files**: `VaktaAi/app/config/settings.py`

---

## 🛠️ CLI & Terminal

### **typer**
- **Purpose**: CLI application framework
- **Used For**:
  - Creating command-line scripts
  - Setup scripts
- **Key Files**: `VaktaAi/scripts/*.py`

### **click**
- **Purpose**: CLI framework
- **Used For**: Command-line interface (typer dependency)

### **rich**
- **Purpose**: Rich terminal formatting
- **Used For**:
  - Pretty console output
  - Progress bars
  - Colored logs

### **tqdm**
- **Purpose**: Progress bars
- **Used For**:
  - Download progress
  - Processing progress

### **colorama**
- **Purpose**: Cross-platform colored terminal
- **Used For**: Windows terminal color support

---

## 🕸️ Web Scraping & Parsing

### **beautifulsoup4**
- **Purpose**: HTML/XML parser
- **Used For**:
  - Web scraping
  - Parsing HTML content
  - YouTube page parsing

### **lxml**
- **Purpose**: Fast XML/HTML parser
- **Used For**: BeautifulSoup backend

### **cssselect** / **soupsieve**
- **Purpose**: CSS selector support
- **Used For**: Selecting HTML elements

### **defusedxml**
- **Purpose**: Secure XML parsing
- **Used For**: Preventing XML attacks

---

## 📊 Monitoring & Logging

### **sentry-sdk**
- **Purpose**: Error tracking and monitoring
- **Used For**:
  - ✅ **Production error tracking**
  - Performance monitoring
  - Exception capture
  - Real-time error alerts
- **Key Files**: `VaktaAi/app/main.py`
- **Integration**: Captures exceptions across the application

### **langsmith**
- **Purpose**: LangChain tracing and monitoring
- **Used For**:
  - Monitoring LLM calls
  - Debugging chains
  - Performance tracking

---

## 🧮 Math & Symbolic Computing

### **sympy**
- **Purpose**: Symbolic mathematics
- **Used For**:
  - Mathematical expression processing
  - Symbolic computation

### **mpmath**
- **Purpose**: Arbitrary-precision arithmetic
- **Used For**: High-precision calculations

---

## 🔄 Async & Concurrency

### **anyio**
- **Purpose**: Cross-platform async framework
- **Used For**: Async abstraction layer

### **greenlet**
- **Purpose**: Lightweight concurrent programming
- **Used For**: Coroutine support (SQLAlchemy dependency)

### **threadpoolctl**
- **Purpose**: Thread pool management
- **Used For**: Controlling thread pools in numpy/scipy

---

## 📦 Core Python Utilities

### **typing-extensions**
- **Purpose**: Extended type hints
- **Used For**: Python 3.8+ type annotations

### **setuptools**
- **Purpose**: Python package tools
- **Used For**: Package installation and management

### **packaging**
- **Purpose**: Version handling
- **Used For**: Parsing package versions

### **toml**
- **Purpose**: TOML file parser
- **Used For**: Configuration files

---

## 🌐 Protocol & Network

### **grpcio**
- **Purpose**: gRPC protocol
- **Used For**:
  - Milvus database communication
  - High-performance RPC

### **protobuf**
- **Purpose**: Protocol buffers
- **Used For**: gRPC data serialization

### **h11**
- **Purpose**: HTTP/1.1 protocol
- **Used For**: Low-level HTTP (httpcore dependency)

### **dnspython**
- **Purpose**: DNS toolkit
- **Used For**: Domain name resolution

---

## 🗜️ Compression & Encoding

### **zstandard**
- **Purpose**: Fast compression algorithm
- **Used For**: Data compression (Milvus dependency)

### **charset_normalizer**
- **Purpose**: Character encoding detection
- **Used For**: Detecting file encodings

---

## 🔐 Security & Data Safety

### **safetensors**
- **Purpose**: Safe tensor serialization
- **Used For**: Secure model weight loading (transformers dependency)

---

## 📚 Miscellaneous Dependencies

### **Pillow**
- **Purpose**: Image processing library
- **Used For**: Image manipulation (if needed)

### **paragraphs**
- **Purpose**: Text paragraph handling
- **Used For**: Text formatting

### **roman**
- **Purpose**: Roman numeral conversion
- **Used For**: Numeral processing

### **regex**
- **Purpose**: Advanced regular expressions
- **Used For**: Pattern matching

### **jinja2**
- **Purpose**: Template engine
- **Used For**: Prompt templating (LangChain dependency)

### **MarkupSafe**
- **Purpose**: Safe string handling
- **Used For**: Jinja2 dependency

### **tenacity**
- **Purpose**: Retry mechanism
- **Used For**: Automatic retries on failures

---

## 📊 Dependency Statistics

| **Category** | **Direct** | **Total with Dependencies** |
|--------------|------------|----------------------------|
| Web & Server | 3 | 8 |
| AI & ML | 9 | 20 |
| Database | 4 | 10 |
| Document Processing | 5 | 8 |
| HTTP & Network | 4 | 15 |
| Security | 2 | 8 |
| Translation | 3 | 5 |
| Data Processing | 2 | 12 |
| Monitoring | 2 | 5 |
| **Total Direct** | **33** | **~130** |

---

## 🎯 Key Technology Decisions

### Why OpenAI over other LLMs?
- Better Hindi/Hinglish support
- Consistent API
- High-quality responses
- Good cost-performance ratio

### Why Milvus over other vector DBs?
- High performance for semantic search
- Scalable
- Open-source
- Good Python support

### Why MongoDB over SQL?
- Flexible schema for chat messages
- Better for document storage
- Easy to scale
- Native JSON support

### Why deep-translator over translate?
- No character limits
- Better API
- More reliable
- Supports chunking

### Why sentence-transformers?
- Fast inference
- Good quality embeddings
- Lightweight models
- Easy to use

---

## 🚀 Installation Commands

```bash
# Install all dependencies
pip install -r requirements.txt

# NLTK data downloads (run in Python)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 📝 Notes

- All dependencies are specified in `VaktaAi/requirements.txt`
- Some packages are automatically installed as sub-dependencies
- Python 3.8+ required
- CUDA optional for faster transformer inference

---

**Generated for VaktaAi Backend**  
*Last updated: October 3, 2025*

