# Database Agent App

- Python 3.8+
- Docker and Docker Compose
- OpenAI API key

## Installation

### Clone the Repo

```bash
git clone
cd db-agent-app
cp .env.copy .env
```

**Important**: Replace `your_actual_openai_api_key_here` with your real OpenAI API key.

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Start the Database

The app includes a sample library database with books, users, and borrowing records.

```bash
docker-compose up
```

The database will be available at:

- **Host**: localhost
- **Port**: 5432
- **Database**: library
- **Username**: postgres
- **Password**: postgres

### Start the Application

```bash
python app.py
```

The app will be available at `http://localhost:8050`

### 2. Connect to Database

1. Open the app in your browser
2. Click "Connect to Database"
3. Use these connection details:
   - **Database Type**: PostgreSQL
   - **Host**: localhost
   - **Port**: 5432
   - **Database Name**: library
   - **Username**: postgres
   - **Password**: postgres

### 3. Ask Questions

Once connected, you can ask questions like:

- "Show me all users"
- "What books are available?"
- "Who has borrowed books?"
- "Show me overdue books"
- "What are the book ratings?"

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
- **`DatabaseConnection`**: Manages database connections and queries
- **`SecurityGuardrail`**: Validates SQL queries for security
- **Conversation Context**: Maintains chat history for better AI responses
