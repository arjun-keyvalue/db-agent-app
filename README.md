# Database Agent App

A sophisticated AI-powered database analysis tool that allows users to interact with databases using natural language queries. Built with Python, Dash, and multiple AI query engines.

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for PostgreSQL setup)
- OpenAI API key (or use free local embeddings)

## Quick Start

### Automated Setup (Recommended)

```bash
git clone <repository-url>
cd db-agent-app
./setup.sh
```

The setup script will:

- Create your `.env` file
- Let you choose between `uv` or `pip`
- Set up the virtual environment
- Install all dependencies

### Manual Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd db-agent-app
cp .env.copy .env
```

**Important**: Edit `.env` and replace `your_actual_openai_api_key_here` with your real OpenAI API key, or configure free local embeddings (see Embedding Configuration below).

### 2. Set Up Python Environment

Choose one of the following methods:

#### Option A: Using uv (Recommended - Faster)

```bash
# Install uv if you haven't already
# Visit: https://docs.astral.sh/uv/getting-started/installation/

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

#### Option B: Using pip (Traditional)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option C: Using pyproject.toml (Development)

```bash
# With uv
uv pip install -e .

# With pip
pip install -e .
```

### 3. Database Setup

Choose one of the following database options:

#### Option A: PostgreSQL with Docker (Recommended)

The app includes a sample library database with books, users, and borrowing records.

```bash
# Start PostgreSQL database
docker-compose up -d

# Wait a few seconds for the database to initialize
```

The database will be available at:

- **Host**: localhost
- **Port**: 5432 (or check your .env file)
- **Database**: library
- **Username**: postgres
- **Password**: postgres

#### Option B: SQLite3 (No Docker required)

```bash
# Create database directory
mkdir -p database/sqllite3

# Initialize SQLite database with sample data
sqlite3 database/sqllite3/library.db < database/sqllite3/sqllite3_seed.sql
```

For SQLite3, update your `.env` file:

```
DB_TYPE=sqlite3
```

### 4. Start the Application

```bash
python app.py
```

The app will be available at `http://localhost:8050`

## Usage

### 1. Connect to Database

1. Open the app in your browser at `http://localhost:8050`
2. Click "Connect to Database"
3. Use these connection details:

   **For PostgreSQL:**

   - **Database Type**: PostgreSQL
   - **Host**: localhost
   - **Port**: 5432 (or check your .env file)
   - **Database Name**: library
   - **Username**: postgres
   - **Password**: postgres

   **For SQLite3:**

   - **Database Type**: SQLite3
   - Leave other fields empty (uses local file)

### 2. Choose Query Strategy

Select from multiple AI-powered query approaches:

- **Schema-Based**: Direct SQL generation from database schema
- **RAG**: Retrieval-Augmented Generation (basic)
- **RAG (Self-Correction and Validation)**: Advanced RAG with 4-layer validation system
- **Multi-Table Join**: Complex relationship queries
- **Visualize**: Generate interactive charts and graphs

### 3. Ask Questions

Once connected, you can ask questions like:

**Basic Queries:**

- "Show me all users"
- "What books are available?"
- "Who has borrowed books?"
- "Show me overdue books"
- "What are the book ratings?"

**Visualization Queries:**

- "Visualize book ratings by genre"
- "Show me a chart of books by publication year"
- "Plot the distribution of user ages"
- "Create a pie chart of book genres"

**Advanced Queries:**

- "Which users have the most overdue books?"
- "Show me books with ratings above 4 stars"
- "Find users who haven't returned books yet"

## Sample Database Schema

The app comes with a pre-populated library database:

### Tables

- **`users`**: Library members with names, emails, and phone numbers
- **`books`**: Book catalog with titles, authors, genres, and copy counts
- **`book_loans`**: Tracks who borrowed what and when it's due
- **`book_reviews`**: User ratings and reviews for books

## Configuration

### LLM & Embedding Configuration

The app supports multiple LLM providers and embedding models:

#### LLM Providers

```bash
# OpenAI (Default)
RAG_PROVIDER=openai
RAG_MODEL=gpt-4o-mini
RAG_API_KEY=your_openai_api_key_here

# Groq (Fast & Free)
RAG_PROVIDER=groq
RAG_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your_groq_api_key_here

# Local Ollama
RAG_PROVIDER=ollama
RAG_MODEL=llama3.1
# No API key needed

# Anthropic Claude
RAG_PROVIDER=anthropic
RAG_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini
RAG_PROVIDER=gemini
RAG_MODEL=gemini-1.5-pro
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Embedding Models

**OpenAI Embeddings** (High Quality)

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
RAG_API_KEY=your_openai_api_key_here
```

**Local Ollama Embeddings** (Recommended - Fast & Free)

```bash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
# Requires: ollama pull nomic-embed-text
```

**Local HuggingFace Embeddings** (Free)

```bash
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Requires: pip install sentence-transformers
```

#### Automatic Schema Indexing

```bash
AUTO_INDEX_SCHEMA=true  # Automatically index database schema to LanceDB
LANCEDB_PATH=./lancedb_rag  # Where to store vector embeddings
```

#### Embedding Troubleshooting

**If you see "LanceDB not initialized" but embeddings work fine:**

- This is normal! The recent fixes ensure the RAG agent works properly
- LanceDB connection and embeddings are now fully compatible
- Schema indexing will work when you connect to a database

**If embeddings fail to initialize:**

- Check that your embedding provider is properly configured
- For Ollama: Ensure `ollama pull nomic-embed-text` was run
- For HuggingFace: Install with `pip install sentence-transformers`
- For OpenAI: Verify your API key is valid
- Schema indexing will be disabled, but the RAG agent will still work

## Package Management

This project supports multiple Python package managers:

### uv (Recommended)

- **Faster installation**: 10-100x faster than pip
- **Better dependency resolution**: More reliable than pip
- **Installation**: Visit [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

### pip (Traditional)

- **Widely supported**: Works everywhere Python works
- **Familiar**: Standard Python package manager
- **Reliable**: Battle-tested and stable

### Switching Between Package Managers

Both `requirements.txt` and `pyproject.toml` are kept in sync. You can use either:

```bash
# Using uv
uv pip install -r requirements.txt

# Using pip
pip install -r requirements.txt

# Development install (either)
uv pip install -e .
pip install -e .
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: If port 5432 is in use, update `DB_PORT` in `.env`
2. **API key issues**: Ensure your API keys are valid and have sufficient credits
3. **Database connection**: Wait a few seconds after `docker-compose up` before connecting
4. **Package conflicts**: Use a fresh virtual environment if you encounter dependency issues
5. **Embedding errors**: Run `python test_embedding_system.py` to verify embedding setup
6. **LanceDB issues**: Check that `LANCEDB_PATH` directory is writable

### Testing Your Setup

```bash
# Test all dependencies
python test_dependencies.py

# Test embedding system
python test_embedding_system.py

# Test RAG agent initialization
python test_rag_init.py

# Test model switching
python test_model_switching.py
```

### Environment Variables

Make sure your `.env` file is properly configured:

```bash
# Database settings
DB_TYPE=postgresql  # or sqlite3
DB_HOST=localhost
DB_PORT=5432
DB_NAME=library
DB_USER=postgres
DB_PASSWORD=postgres

# LLM Configuration
RAG_PROVIDER=openai  # openai, groq, ollama, anthropic, gemini
RAG_MODEL=gpt-4o-mini
RAG_API_KEY=sk-your-actual-key-here

# Embedding Configuration
EMBEDDING_PROVIDER=openai  # openai or huggingface
EMBEDDING_MODEL=text-embedding-3-small
AUTO_INDEX_SCHEMA=true

# Alternative API Keys
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## Development

### Project Structure

```
db-agent-app/
├── app.py                 # Main Dash application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Database setup
├── .env                  # Environment variables
├── database/
│   ├── __init__.py
│   ├── connection.py     # Database connection management
│   ├── query_engine.py  # AI-powered query generation
│   └── seed.sql         # Sample database data
└── README.md
```

### Key Components

- **`SchemaBasedQueryEngine`**: Generates SQL using OpenAI and database schema
- **`RAGAgent`**: Advanced agent with self-correction and 4-layer validation
- **`DatabaseConnection`**: Manages database connections and queries with auto-indexing
- **`SecurityGuardrail`**: Validates SQL queries for security
- **`SchemaIndexer`**: Automatically indexes database schema to LanceDB on connection
- **`LanceDB Integration`**: Embedded vector database for RAG context retrieval
- **`ModelSwitcher`**: Easy switching between different LLM providers and models
- **Conversation Context**: Maintains chat history for better AI responses

### Advanced RAG Agent Features

The RAG Agent with Self-Correction implements a sophisticated 4-layer validation system:

1. **Layer 1: Syntactic Validation** - SQLFluff and SQLGlot for syntax checking
2. **Layer 2: Semantic Validation** - Schema verification to prevent AI hallucinations
3. **Layer 3: AI-Powered Self-Correction** - Execution feedback loop with iterative debugging
4. **Layer 4: Performance Guardrails** - Automatic LIMIT clauses and safety measures

## Technology Stack

- **Frontend**: Dash + Bootstrap (dark theme)
- **Backend**: Python, SQLAlchemy, Pandas
- **AI**: Multi-provider LLM support via LiteLLM
- **LLM Providers**: OpenAI, Groq, Ollama, Anthropic, Gemini
- **Embeddings**: OpenAI, HuggingFace (local/free)
- **Databases**: PostgreSQL, MySQL, SQLite3
- **Visualization**: Plotly
- **Vector DB**: LanceDB (embedded, no Docker required)
- **SQL Validation**: SQLFluff, SQLGlot
- **Workflow**: LangGraph for agent orchestration

## RAG Agent Architecture

The advanced RAG agent implements a sophisticated workflow:

```
User Query → Schema Retrieval → Context Retrieval → Query Generation
     ↓
Syntactic Validation → Semantic Validation → Performance Guards
     ↓
Query Execution → Self-Correction Loop → Formatted Output
```

### Embedding & Vector Storage

**Automatic Schema Indexing**: When you connect to a database, the system automatically:

1. Extracts database schema (tables, columns, relationships)
2. Creates intelligent text chunks with sample data
3. Generates embeddings using your chosen model
4. Stores embeddings in LanceDB for fast similarity search
5. Enables intelligent query generation with proper table/column names

**LanceDB Benefits**:

- **No Docker Required**: Embedded database that runs in-process
- **High Performance**: Optimized for similarity search and retrieval
- **Persistent Storage**: Data persists between application restarts
- **Easy Setup**: No external services or configuration needed
- **Automatic Indexing**: Schema changes are automatically re-indexed

**Embedding Options**:

- **OpenAI**: High quality, ~$0.02 for typical database
- **HuggingFace**: Free, runs locally, good quality

This makes the application easier to deploy and maintain while providing powerful RAG capabilities with intelligent schema understanding.

## Quick Setup Examples

### Free Setup (No API Keys)

```bash
# Use local models only
RAG_PROVIDER=ollama
RAG_MODEL=llama3.1
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Production Setup (Best Quality)

```bash
# Use OpenAI for both LLM and embeddings
RAG_PROVIDER=openai
RAG_MODEL=gpt-4o
RAG_API_KEY=your_openai_api_key_here
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
```

### Fast & Cheap Setup

```bash
# Use Groq for LLM, local embeddings
RAG_PROVIDER=groq
RAG_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

For detailed embedding configuration, see [EMBEDDING_SYSTEM.md](EMBEDDING_SYSTEM.md).
