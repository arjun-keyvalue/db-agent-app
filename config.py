import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application"""
    
    # Database Configuration
    DB_TYPE = os.getenv('DB_TYPE', 'postgresql')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'your_database')
    DB_USERNAME = os.getenv('DB_USERNAME', 'your_username')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_password')
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # RAG Agent Configuration
    RAG_MODEL = os.getenv('RAG_MODEL', 'gpt-3.5-turbo')
    RAG_API_KEY = os.getenv('RAG_API_KEY', os.getenv('OPENAI_API_KEY'))  # Fallback to OPENAI_API_KEY
    RAG_PROVIDER = os.getenv('RAG_PROVIDER', 'openai')
    LANCEDB_PATH = os.getenv('LANCEDB_PATH', './lancedb_rag')
    MAX_CORRECTIONS = int(os.getenv('MAX_CORRECTIONS', '3'))
    QUERY_TIMEOUT = int(os.getenv('QUERY_TIMEOUT', '30'))
    
    # Additional API Keys for different providers
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    @classmethod
    def get_rag_api_key(cls):
        """Get the appropriate API key based on RAG provider"""
        provider = cls.RAG_PROVIDER.lower()
        
        if provider == 'groq':
            return cls.GROQ_API_KEY or cls.RAG_API_KEY
        elif provider == 'google' or provider == 'gemini':
            return cls.GEMINI_API_KEY or cls.RAG_API_KEY
        elif provider == 'anthropic':
            return cls.ANTHROPIC_API_KEY or cls.RAG_API_KEY
        elif provider == 'openai':
            return cls.RAG_API_KEY or cls.OPENAI_API_KEY
        else:
            return cls.RAG_API_KEY
    
    # App Configuration
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '8050'))
    
    @classmethod
    def get_database_url(cls):
        """Generate database connection URL"""
        if cls.DB_TYPE == 'sqlite':
            return f"sqlite:///{cls.DB_NAME}.db"
        elif cls.DB_TYPE == 'postgresql':
            return f"postgresql://{cls.DB_USERNAME}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        elif cls.DB_TYPE == 'mysql':
            return f"mysql+pymysql://{cls.DB_USERNAME}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        elif cls.DB_TYPE == 'mssql':
            return f"mssql+pyodbc://{cls.DB_USERNAME}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
        else:
            raise ValueError(f"Unsupported database type: {cls.DB_TYPE}")
