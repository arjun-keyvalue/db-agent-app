# Database Agent App

A sophisticated AI-powered database analysis tool that allows users to interact with databases using natural language queries. Built with Python, Dash, and multiple AI query engines.

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for PostgreSQL setup)
- OpenAI API key

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

**Important**: Edit `.env` and replace `your_actual_openai_api_key_here` with your real OpenAI API key.

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

### OpenAI Settings

- **Model**: gpt-3.5-turbo
- **Max Tokens**: 1000
- **Temperature**: 0 (for consistent SQL generation)

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
2. **OpenAI API key**: Ensure your API key is valid and has sufficient credits
3. **Database connection**: Wait a few seconds after `docker-compose up` before connecting
4. **Package conflicts**: Use a fresh virtual environment if you encounter dependency issues

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

# AI settings
OPENAI_API_KEY=sk-your-actual-key-here
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
- **`DatabaseConnection`**: Manages database connections and queries
- **`SecurityGuardrail`**: Validates SQL queries for security
- **`LanceDB Integration`**: Embedded vector database for RAG context retrieval
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
- **AI**: OpenAI GPT-3.5/GPT-4, LangChain, LangGraph
- **LLM Integration**: LiteLLM (multi-model support)
- **Databases**: PostgreSQL, MySQL, SQLite3
- **Visualization**: Plotly
- **Vector DB**: LanceDB (embedded, no Docker required)
- **SQL Validation**: SQLFluff, SQLGlot

## RAG Agent Architecture

The advanced RAG agent implements a sophisticated workflow:

```
User Query → Schema Retrieval → Context Retrieval → Query Generation
     ↓
Syntactic Validation → Semantic Validation → Performance Guards
     ↓
Query Execution → Self-Correction Loop → Formatted Output
```

### Why LanceDB?

LanceDB is used as the vector database because:
- **No Docker Required**: Embedded database that runs in-process
- **High Performance**: Optimized for similarity search and retrieval
- **Persistent Storage**: Data persists between application restarts
- **Easy Setup**: No external services or configuration needed

This makes the application easier to deploy and maintain while providing powerful RAG capabilities.