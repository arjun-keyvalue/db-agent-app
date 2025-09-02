import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from typing import Dict, Any, Tuple
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Handles database connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.connection = None
        self.connection_info = {}
    
    def connect(self, db_type: str, host: str, port: str, db_name: str, username: str, password: str) -> Tuple[bool, str]:
        """
        Establish database connection
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Build connection URL
            if db_type == 'postgresql':
                connection_url = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
            elif db_type == 'mysql':
                connection_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
            elif db_type == 'sqlite3':
                # For SQLite3, use the db_name parameter directly as it contains the full path
                if db_name and os.path.exists(db_name):
                    connection_url = f"sqlite:///{db_name}"
                else:
                    # Fallback: construct absolute path to the database file
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    db_path = os.path.join(base_dir, 'sqllite3', 'library.db')
                    connection_url = f"sqlite:///{db_path}"
            
            else:
                return False, f"Unsupported database type: {db_type}"
            
            # Create engine
            logger.info(f"Creating database engine with URL: {connection_url}")
            self.engine = create_engine(
                connection_url,
                echo=False,  # Set to True for debugging SQL queries
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Test connection
            logger.info(f"Testing database connection...")
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Database connection test successful")
            
            # Store connection info
            self.connection_info = {
                'type': db_type,
                'host': host,
                'port': port,
                'database': db_name,
                'username': username
            }
            
            logger.info(f"Successfully connected to {db_type} database: {db_name}")
            return True, f"Successfully connected to {db_type} database: {db_name}"
            
        except SQLAlchemyError as e:
            error_msg = f"Database connection failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def disconnect(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.connection_info = {}
            logger.info("Database connection closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False
    
    def get_tables(self) -> Tuple[bool, Any]:
        """Get list of tables in the database"""
        if not self.is_connected():
            return False, "Not connected to database"
        
        try:
            if self.connection_info['type'] == 'postgresql':
                query = """
                    SELECT table_name, table_type 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """
            elif self.connection_info['type'] == 'mysql':
                query = f"""
                    SELECT table_name, table_type 
                    FROM information_schema.tables 
                    WHERE table_schema = '{self.connection_info['database']}'
                    ORDER BY table_name
                """
            elif self.connection_info['type'] == 'sqlite3':
                query = """
                    SELECT 
                        name as table_name,
                        type as table_type
                    FROM sqlite_master
                    WHERE type='table'
                    ORDER BY name
                """
            else:
                return False, f"Unsupported database type: {self.connection_info['type']}"
            
            df = pd.read_sql(query, self.engine)
            return True, df
            
        except Exception as e:
            error_msg = f"Failed to get tables: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_table_schema(self, table_name: str) -> Tuple[bool, Any]:
        """Get schema information for a specific table"""
        if not self.is_connected():
            return False, "Not connected to database"
        
        logger.info(f"Getting schema for table: {table_name}")
        logger.info(f"Engine exists: {self.engine is not None}")
        logger.info(f"Connection info: {self.connection_info}")
        
        try:
            if self.connection_info['type'] == 'postgresql':
                query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = :table_name AND table_schema = 'public'
                ORDER BY ordinal_position
            """
                # Use SQLAlchemy text() for proper parameter binding
                from sqlalchemy import text
                with self.engine.connect() as conn:
                    result = conn.execute(text(query), {"table_name": table_name})
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
            elif self.connection_info['type'] == 'mysql':
                query = f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = :table_name AND table_schema = '{self.connection_info['database']}'
                    ORDER BY ordinal_position
                """
                df = pd.read_sql(query, self.engine, params={'table_name': table_name})
            elif self.connection_info['type'] == 'sqlite3':
                # For SQLite, we use the pragma_table_info function
                query = f"""
                    SELECT 
                        name as column_name,
                        type as data_type,
                        CASE WHEN "notnull" = 0 THEN 'YES' ELSE 'NO' END as is_nullable,
                        dflt_value as column_default
                    FROM pragma_table_info('{table_name}')
                """
                df = pd.read_sql(query, self.engine)
            else:
                return False, f"Unsupported database type: {self.connection_info['type']}"
                
            return True, df
            
        except Exception as e:
            error_msg = f"Failed to get table schema: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def execute_query(self, query: str) -> Tuple[bool, Any]:
        """Execute a custom SQL query"""
        if not self.is_connected():
            return False, "Not connected to database"
        
        try:
            df = pd.read_sql(query, self.engine)
            return True, df
            
        except Exception as e:
            error_msg = f"Query execution failed: {str(e) if 'syntax' not in str(e).lower() else 'sorry'}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information"""
        return self.connection_info.copy()

# Global database connection instance
db_connection = DatabaseConnection()
