from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import openai
from openai import OpenAI
import logging
from datetime import datetime
from database.connection import db_connection
from sqlalchemy import text
from langchain_community.utilities import SQLDatabase
from .sqllite3.schema_embeddings import SchemaEmbeddings
from .sqllite3.sql_generator import SQLGenerator


from services.visualization.service import VisualizationService

logger = logging.getLogger(__name__)

class QueryEngine(ABC):
    """Base interface for query engines"""
    
    @abstractmethod
    def generate_query(self, user_query: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Generate SQL query from user input"""
        pass
    
    @abstractmethod
    def execute_query(self, sql_query: str) -> Tuple[bool, Any]:
        """Execute the generated SQL query"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the query engine"""
        pass

class SecurityGuardrail(ABC):
    """Base interface for security guardrails"""
    
    @abstractmethod
    def validate_query(self, sql_query: str, context: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Validate SQL query for security concerns. Returns (success, message, modified_sql)"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the security guardrail"""
        pass

class SchemaBasedQueryEngine(QueryEngine):
    """Schema-based query generation using OpenAI"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.conversation_context = []  # Store conversation history
        self.current_table_context = None  # Store current table being discussed
        logger.info(f"Initialized SchemaBasedQueryEngine with empty conversation context")
    
    def get_name(self) -> str:
        return "Schema-Based Querying"
    
    def _update_conversation_context(self, user_query: str, sql_query: str, table_name: str = None):
        """Update conversation context with current interaction"""
        logger.info(f"Adding to conversation context: User='{user_query}', SQL='{sql_query}', Table='{table_name}'")
        
        self.conversation_context.append({
            'user_query': user_query,
            'sql_query': sql_query,
            'table_name': table_name,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 5 interactions to avoid context bloat
        if len(self.conversation_context) > 5:
            self.conversation_context = self.conversation_context[-5:]
        
        # Update current table context if we're working with a specific table
        if table_name:
            self.current_table_context = table_name
            logger.info(f"Updated current table context to: {table_name}")
        
        logger.info(f"Conversation context now has {len(self.conversation_context)} interactions")
    
    def generate_query(self, user_query: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Generate SQL query using OpenAI based on database schema"""
        logger.info(f"Starting generate_query with user_query: '{user_query}'")
        logger.info(f"Current conversation context has {len(self.conversation_context)} interactions")
        
        try:
            # Get database schema information
            if not db_connection.is_connected():
                return False, "Not connected to database"
            
            # Get tables and their schemas
            tables_success, tables_result = db_connection.get_tables()
            if not tables_success:
                return False, f"Failed to get tables: {tables_result}"
            
            # Debug logging
            logger.info(f"Tables found: {tables_result}")
            
            # Check if we have any tables
            if tables_result.empty:
                return False, "No tables found in the database"
            
            # Build schema context
            schema_context = self._build_schema_context(tables_result)
            
            # Debug logging
            logger.info(f"Schema context: {schema_context}")
            
            # Validate that we have actual schema data
            if "WARNING: No tables found" in schema_context or "Available tables in this database:" not in schema_context:
                return False, "Failed to extract database schema properly"
            
            # Create prompt for OpenAI
            prompt = self._create_prompt(user_query, schema_context, context)
            
            # Debug logging - show exactly what's being sent to OpenAI
            logger.info(f"=== OPENAI PROMPT ===")
            logger.info(f"User Query: {user_query}")
            logger.info(f"Schema Context: {schema_context}")
            logger.info(f"Full Prompt: {prompt}")
            logger.info(f"=== END PROMPT ===")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. You are STRICTLY FORBIDDEN from using any tables, columns, or relationships that are not explicitly listed in the provided schema. You must verify every element exists before generating SQL. If anything is missing, explain what IS available instead of guessing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.0
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response (remove markdown if present)
            if sql_query.startswith("```sql"):
                sql_query = sql_query[7:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            sql_query = sql_query.strip()
            
            # Add public schema prefix to all table references for PostgreSQL
            if context.get('db_type') == 'postgresql':
                import re
                # Simple pattern matching to add public. prefix to table names
                # Match FROM table, JOIN table, etc.
                sql_query = re.sub(r'\bFROM\s+(\w+)\b', r'FROM public.\1', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\bJOIN\s+(\w+)\b', r'JOIN public.\1', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\bUPDATE\s+(\w+)\b', r'UPDATE public.\1', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\bINSERT\s+INTO\s+(\w+)\b', r'INSERT INTO public.\1', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\bDELETE\s+FROM\s+(\w+)\b', r'DELETE FROM public.\1', sql_query, flags=re.IGNORECASE)
                logger.info(f"Added schema prefix to query: {sql_query}")
            
            # Final validation: ensure the generated SQL only uses existing tables and columns
            if not self._validate_sql_against_schema(sql_query, tables_result):
                return False, "Generated SQL references non-existent tables or columns. Please try rephrasing your question."
            
            logger.info(f"Generated SQL query: {sql_query}")
            
            # Update conversation context with the successful SQL generation
            table_name = self._extract_table_name_from_sql(sql_query)
            logger.info(f"Updating conversation context with table: {table_name}")
            self._update_conversation_context(user_query, sql_query, table_name)
            logger.info(f"Conversation context updated. Total interactions: {len(self.conversation_context)}")
            
            logger.info(f"Ending generate_query successfully. Final context has {len(self.conversation_context)} interactions")
            return True, sql_query
            
        except Exception as e:
            error_msg = f"Failed to generate SQL query: {str(e)}"
            logger.error(error_msg)
            
            # Update conversation context with the error
            self._update_conversation_context(user_query, f"ERROR: {error_msg}", None)
            
            logger.info(f"Ending generate_query with error. Final context has {len(self.conversation_context)} interactions")
            return False, error_msg
    
    def execute_query(self, sql_query: str) -> Tuple[bool, Any]:
        """Execute the generated SQL query"""
        try:
            if not db_connection.is_connected():
                return False, "Not connected to database"
            
            success, result = db_connection.execute_query(sql_query)
            return success, result
            
        except Exception as e:
            error_msg = f"Failed to execute SQL query: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _build_schema_context(self, tables_df) -> str:
        """Build comprehensive schema context for OpenAI"""
        schema_context = "AVAILABLE DATABASE SCHEMA:\n\n"
        
        available_tables = []
        
        for _, table_row in tables_df.iterrows():
            table_name = table_row['table_name']
            table_type = table_row['table_type']
            
            if table_type == 'BASE TABLE':  # Only include actual tables, not views
                available_tables.append(table_name)
                # Get table schema
                schema_success, schema_result = db_connection.get_table_schema(table_name)
                if schema_success:
                    logger.info(f"Building schema for table {table_name}: {schema_result}")
                    schema_context += f"TABLE: {table_name}\n"
                    schema_context += "COLUMNS:\n"
                    
                    for _, col_row in schema_result.iterrows():
                        col_name = col_row['column_name']
                        col_type = col_row['data_type']
                        nullable = col_row['is_nullable']
                        default_val = col_row['column_default']
                        
                        schema_context += f"  - {col_name}: {col_type}"
                        if nullable == 'NO':
                            schema_context += " (NOT NULL)"
                        if default_val:
                            schema_context += f" DEFAULT {default_val}"
                        schema_context += "\n"
                    
                    schema_context += "\n"
        
        # Add summary of available tables
        if available_tables:
            schema_context += f"SUMMARY: Available tables in this database: {', '.join(available_tables)}\n\n"
            schema_context += "IMPORTANT: You can ONLY use these tables and their columns. Do NOT reference any other tables.\n\n"
        else:
            schema_context += "WARNING: No tables found in this database.\n\n"
        
        return schema_context
    
    def _create_prompt(self, user_query: str, schema_context: str, context: Dict[str, Any]) -> str:
        """Create the prompt for OpenAI"""
        db_type = context.get('db_type', 'postgresql')
        
        prompt = f"""
You are a SQL expert. Generate a SQL query based on the EXACT database schema provided below.

Database Type: {db_type}

{schema_context}

CURRENT USER REQUEST: {user_query}

CONVERSATION HISTORY (for context only - focus on current request above):
{self._build_conversation_context()}

CRITICAL REQUIREMENTS - READ CAREFULLY:
1. ONLY use tables and columns that are EXPLICITLY listed in the schema above
2. Do NOT assume any column names exist - use ONLY what is shown
3. Do NOT assume any relationships between tables - use ONLY what is shown
4. If joining tables, use ONLY columns that actually exist in both tables
5. If the user asks about something not in the schema, respond with available tables
6. Generate valid SQL for {db_type}
7. Add LIMIT if the query might return many rows
8. Return ONLY the SQL query - NO explanatory text, NO markdown, NO comments

COLUMN NAME MAPPING - CRITICAL:
- When user asks for something (like "name", "phone", "email"), look through the schema to find the most appropriate column
- Use ONLY the exact column names that exist in the schema
- If multiple columns could match the user's intent, choose the most relevant one
- NEVER invent or assume column names that don't exist in the schema
- Do NOT prefix column names with table names (e.g., use "full_name" not "user.full_name")
- You can use AS aliases to rename columns if needed (e.g., "full_name AS name")

SCHEMA-BASED COLUMN SELECTION:
- Look at the ACTUAL column names listed in the schema above
- Use ONLY those exact column names when generating SQL
- Do not invent, assume, or guess column names
- The schema shows you exactly what columns exist - use them as-is

SCHEMA COMPLIANCE CHECK:
- Before generating SQL, verify every table and column exists in the schema above
- If any table or column is missing, DO NOT generate SQL
- Instead, list what IS available and suggest alternatives

OUTPUT FORMAT:
- Return ONLY the SQL query
- Do NOT include any explanatory text, comments, or markdown formatting
- Do NOT include phrases like "Here's a SQL query:" or "This query selects..."
- Just return the raw SQL query

If the user's request cannot be satisfied with the available schema, respond with:
"Schema Analysis: The requested information is not available in the current database schema. Available tables are: [list of table names]"

Otherwise, generate the SQL query using ONLY the exact schema provided:
"""
        return prompt
    
    def _build_conversation_context(self) -> str:
        """Build conversation context for the AI model"""
        logger.info(f"Building conversation context. Total interactions: {len(self.conversation_context)}")
        logger.info(f"Conversation context content: {self.conversation_context}")
        
        if not self.conversation_context:
            return "This is a new conversation. No previous context available."
        
        context = "Previous interactions in this session:\n"
        for i, interaction in enumerate(self.conversation_context[-3:], 1):  # Show last 3 interactions
            context += f"{i}. User asked: '{interaction['user_query']}'\n"
            context += f"   Generated SQL: {interaction['sql_query']}\n"
            if interaction['table_name']:
                context += f"   Table discussed: {interaction['table_name']}\n"
            context += "\n"
        
        if self.current_table_context:
            context += f"Context: You were previously working with the '{self.current_table_context}' table.\n"
            context += "Use this as background context, but prioritize the CURRENT USER REQUEST above.\n"
        
        logger.info(f"Built conversation context: {context}")
        return context
    
    def _extract_table_name_from_sql(self, sql_query: str) -> str:
        """Extract the main table name from SQL query"""
        import re
        
        # Look for FROM clause
        from_match = re.search(r'FROM\s+(?:\w+\.)?(\w+)', sql_query, re.IGNORECASE)
        if from_match:
            return from_match.group(1)
        
        return None
    
    def _validate_sql_against_schema(self, sql_query: str, tables_df) -> bool:
        """Validate that the generated SQL only uses existing tables and columns"""
        try:
            # Extract table names from SQL (simple regex approach)
            import re
            
            # Get all available table names
            available_tables = set(tables_df[tables_df['table_type'] == 'BASE TABLE']['table_name'].tolist())
            
            # Extract table names from SQL (FROM, JOIN clauses)
            # Handle both schema.table and table formats
            from_pattern = r'FROM\s+(?:\w+\.)?(\w+)'
            join_pattern = r'JOIN\s+(?:\w+\.)?(\w+)'
            
            from_tables = re.findall(from_pattern, sql_query, re.IGNORECASE)
            join_tables = re.findall(join_pattern, sql_query, re.IGNORECASE)
            
            all_referenced_tables = set(from_tables + join_tables)
            
            # Check if all referenced tables exist
            for table in all_referenced_tables:
                if table.lower() not in [t.lower() for t in available_tables]:
                    logger.warning(f"SQL references non-existent table: {table}")
                    return False
            
            # Now validate column names for each referenced table
            for table in all_referenced_tables:
                # Get the actual table name (remove schema prefix if present)
                actual_table = table
                if '.' in table:
                    actual_table = table.split('.')[-1]
                
                # Get table schema to validate columns
                schema_success, schema_result = db_connection.get_table_schema(actual_table)
                if not schema_success:
                    logger.warning(f"Could not get schema for table: {actual_table}")
                    continue
                
                available_columns = set(schema_result['column_name'].tolist())
                
                # Extract column names from SELECT clause (simple approach)
                select_pattern = r'SELECT\s+(.*?)\s+FROM'
                select_match = re.search(select_pattern, sql_query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1).strip()
                    # Handle * case
                    if select_clause == '*':
                        continue
                    
                    # Extract individual column names, handling aliases and table prefixes
                    columns = []
                    for col in select_clause.split(','):
                        col = col.strip()
                        if col == '*':
                            continue
                        
                        # Remove table prefix if present (e.g., "user.full_name" -> "full_name")
                        if '.' in col:
                            col = col.split('.')[-1]
                        
                        # Remove AS alias if present (e.g., "full_name AS name" -> "full_name")
                        if ' AS ' in col.upper():
                            col = col.split(' AS ')[0].strip()
                        
                        columns.append(col)
                    
                    for column in columns:
                        if column.lower() not in [col.lower() for col in available_columns]:
                            logger.warning(f"SQL references non-existent column '{column}' in table '{actual_table}'")
                            return False
            
            logger.info(f"SQL validation passed for tables: {all_referenced_tables}")
            return True
            
        except Exception as e:
            logger.error(f"Error during SQL validation: {str(e)}")
            return False
        

class MultitablejoinQueryEngine(QueryEngine):
    """Multitablejoin-based query generation"""
    
    def __init__(self, db_uri: str, openai_api_key: str, groq_api_key: str, lancedb_uri: str = "./lancedb"):
        
        if not db_connection.is_connected():
            raise ValueError("Not connected to database")
        # self.db = SQLDatabase.from_uri(db_uri)
        # self.engine = self.db._engine
        self.schema_embeddings = SchemaEmbeddings(openai_api_key, lancedb_uri)
        self.sql_generator = SQLGenerator(groq_api_key)
        
        # Initialize schema embeddings
        self._initialize_schema_embeddings()
    
    def get_name(self) -> str:
        return "Multitablejoin Querying"
    
    def _initialize_schema_embeddings(self) -> None:
        """Initialize schema embeddings by extracting and storing schema information"""
        schema_data = []
        
        with db_connection.engine.connect() as conn:
            # Get tables first
            tables_success, tables_result = db_connection.get_tables()
            if not tables_success:
                raise ValueError(f"Failed to get tables: {tables_result}")
            
            # Iterate over the DataFrame rows
            for _, table_row in tables_result.iterrows():
                table_name = table_row['table_name']
                table_data = {}
                table_data['table'] = table_name
                
                # Get schema
                schema = conn.execute(text(f"PRAGMA table_info({table_name});")).fetchall()
                schema_str = ", ".join([f"{col[1]} {col[2]}" for col in schema])
                table_data['schema'] = schema_str
                
                # Get foreign keys
                fkeys = conn.execute(text(f"PRAGMA foreign_key_list({table_name});")).fetchall()
                if fkeys:
                    fk_str = "; ".join([f"{table_name}.{fk[3]} → {fk[2]}.{fk[4]}" for fk in fkeys])
                    table_data['relationships'] = fk_str
                
                # Get sample rows
                rows = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3;")).fetchall()
                table_data['sample_rows'] = rows
                
                schema_data.append(table_data)
        
        # Create and store embeddings
        chunks = self.schema_embeddings.create_schema_chunks(schema_data)
        self.schema_embeddings.store_embeddings(chunks)
    
    def generate_query(self, user_query: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Generate SQL query using multitablejoin approach"""
        try:
            # Get relevant schema chunks
            schema_chunks = self.schema_embeddings.get_relevant_chunks(user_query)
            
            # Generate SQL query
            sql_query = self.sql_generator.generate_sql(user_query, schema_chunks)
            
            return True, sql_query
            
        except Exception as e:
            return False, f"Failed to generate query: {str(e)}"
    
    def execute_query(self, sql_query: str) -> Tuple[bool, Any]:
        """Execute the generated SQL query"""
        try:
            result = db_connection.execute_query(sql_query)
            
            if not result[0]:
                return False, f"Failed to execute query: {result[1]}"
            
            return True, result[1]
            
        except Exception as e:
            return False, f"Query execution failed: {str(e)}"













class RAGQueryEngine(QueryEngine):
    """RAG-based query generation (placeholder)"""
    
    def get_name(self) -> str:
        return "RAG Querying"
    
    def generate_query(self, user_query: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Placeholder for RAG implementation"""
        return False, "RAG querying not yet implemented"
    
    def execute_query(self, sql_query: str) -> Tuple[bool, Any]:
        """Placeholder for RAG implementation"""
        return False, "RAG querying not yet implemented"

class VisualizationQueryEngine(QueryEngine):
    """Visualization-based query generation using LangChain and Plotly"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.visualization_service = None
        logger.info("Initialized VisualizationQueryEngine")
    
    def get_name(self) -> str:
        return "Data Visualization"
    
    def generate_query(self, user_query: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Generate visualization from user query.
        Note: This method returns visualization metadata instead of SQL.
        """
        try:
            # Lazy import to avoid circular dependencies
            if self.visualization_service is None:
                self.visualization_service = VisualizationService(db_connection, self.groq_api_key)
            
            # Validate if the query is suitable for visualization
            is_valid, validation_message = self.visualization_service.validate_visualization_request(user_query)
            if not is_valid:
                return False, validation_message
            
            # Process the visualization request
            success, result = self.visualization_service.process_visualization_request(user_query, context)
            
            if success:
                # Store the visualization result for execute_query to return
                self._last_result = result
                # Return the SQL query that was generated for transparency
                return True, result['sql_query']
            else:
                return False, result
                
        except Exception as e:
            error_msg = f"Visualization generation failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def execute_query(self, sql_query: str) -> Tuple[bool, Any]:
        """
        Return the visualization result instead of executing SQL.
        The actual SQL execution is handled by the visualization service.
        """
        try:
            if hasattr(self, '_last_result'):
                result = self._last_result
                delattr(self, '_last_result')  # Clean up
                return True, result
            else:
                return False, "No visualization result available"
        except Exception as e:
            error_msg = f"Failed to return visualization: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

class BasicSecurityGuardrail(SecurityGuardrail):
    """Basic SQL injection and destructive operation protection"""
    
    def get_name(self) -> str:
        return "Basic Security Guardrails"
    
    def validate_query(self, sql_query: str, context: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Validate query and input for unsafe patterns, optionally with LLM classification."""
        try:
            user_text = (context.get("user_input") or "").lower()
            combined_text = f" {user_text} \n {sql_query.lower()} "

            unsafe_markers = [
                " drop ",
                " delete ",
                " truncate ",
                " alter ",
                " update ",
                " insert ",
                " create ",
                " grant ",
                " revoke ",
                " vacuum ",
                " analyze ",
                ";--",
                " or 1=1",
                " xp_cmdshell",
                # intent synonyms
                " destroy ",
                " remove table",
                " erase table",
                " wipe ",
                " purge ",
                " drop table",
            ]

            # Additional sensitive data exfiltration markers (critical)
            sensitive_markers = [
                " username ",
                " user name ",
                " usernames ",
                " password ",
                " passwords ",
                " passwd ",
                " password hash",
                " password hashes",
                " email ",
                " emails ",
                " phone ",
                " phone number",
                " phone numbers",
                " date of birth",
                " date_of_birth",
                " dob ",
                " all users",
                " all user",
                " every user",
                " user table",
                " users table",
            ]

            # Block if any static destructive marker appears (always enabled for basic safety)
            if any(marker in combined_text for marker in unsafe_markers):
                return False, "Blocked by guardrails: potentially destructive or unsafe database operation detected.", sql_query

            # Sensitive data protection (only when security is enabled)
            if context.get("security_enabled", True):
                # Inspect SELECT columns for sensitive identifiers and modify queries
                import re
                
                # Define sensitive tables that contain PII or credentials
                sensitive_tables = ['users', 'user', 'accounts', 'account']
                
                # Define safe columns for sensitive tables (columns that can be safely exposed)
                safe_columns = {
                    'users': ['id', 'username', 'full_name', 'is_active', 'created_at', 'updated_at'],
                    'user': ['id', 'username', 'full_name', 'is_active', 'created_at', 'updated_at'],
                    'accounts': ['id', 'username', 'is_active', 'created_at', 'updated_at'],
                    'account': ['id', 'username', 'is_active', 'created_at', 'updated_at']
                }
                
                # Check user intent first - if they're explicitly asking for sensitive data, block the query
                user_input = context.get("user_input", "").lower()
                sensitive_intent_markers = [
                    "email", "emails", "e-mail", "e-mails",
                    "phone", "phone number", "phone numbers", "telephone", "telephone number",
                    "password", "passwords", "passwd", "pwd",
                    "date of birth", "birth date", "dob", "birthday",
                    "social security", "ssn", "ss number",
                    "credit card", "cc number", "card number",
                    "address", "home address", "mailing address",
                    "personal information", "pii", "private data"
                ]
                
                # If user is explicitly asking for sensitive data, block the query
                if any(marker in user_input for marker in sensitive_intent_markers):
                    return False, "Blocked by guardrails: request for sensitive personal information is not allowed.", sql_query
                
                # Check for SELECT queries on sensitive tables
                select_match = re.search(r"SELECT\s+(.*?)\s+FROM\s+(\w+)", sql_query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1).strip()
                    table_name = select_match.group(2).strip().lower()
                    
                    # If querying a sensitive table, modify the SELECT clause to only include safe columns
                    if table_name in sensitive_tables:
                        if select_clause == '*':
                            # Replace SELECT * with only safe columns
                            safe_cols = safe_columns.get(table_name, [])
                            if safe_cols:
                                modified_select = ', '.join(safe_cols)
                                sql_query = re.sub(
                                    r"SELECT\s+\*\s+FROM\s+" + re.escape(table_name),
                                    f"SELECT {modified_select} FROM {table_name}",
                                    sql_query,
                                    flags=re.IGNORECASE
                                )
                                logger.info(f"Modified SELECT * query on sensitive table '{table_name}' to only include safe columns: {safe_cols}")
                            else:
                                return False, f"Blocked by guardrails: no safe columns defined for sensitive table '{table_name}'", sql_query
                        else:
                            # Check if any sensitive columns are explicitly selected
                            columns = [c.strip() for c in select_clause.split(',')]
                            sensitive_cols_found = []
                            safe_cols_to_keep = []
                            
                            for col in columns:
                                # Normalize column name
                                normalized_col = col.lower()
                                # Remove AS alias
                                if ' as ' in normalized_col:
                                    normalized_col = normalized_col.split(' as ')[0].strip()
                                # Remove table prefix
                                if '.' in normalized_col:
                                    normalized_col = normalized_col.split('.')[-1].strip()
                                
                                # Check if it's a sensitive column
                                if re.search(r"\b(user_?name|username|password|passwd|password_?hash|pwd_hash|email|phone_?number|date_?of_?birth)\b", normalized_col):
                                    sensitive_cols_found.append(col.strip())
                                else:
                                    safe_cols_to_keep.append(col.strip())
                            
                            # If sensitive columns were found, remove them from the query
                            if sensitive_cols_found:
                                if safe_cols_to_keep:
                                    # Replace the SELECT clause with only safe columns
                                    modified_select = ', '.join(safe_cols_to_keep)
                                    sql_query = re.sub(
                                        r"SELECT\s+" + re.escape(select_clause) + r"\s+FROM\s+" + re.escape(table_name),
                                        f"SELECT {modified_select} FROM {table_name}",
                                        sql_query,
                                        flags=re.IGNORECASE
                                    )
                                    logger.info(f"Removed sensitive columns from query on '{table_name}': {sensitive_cols_found}")
                                else:
                                    # If no safe columns remain, use default safe columns
                                    safe_cols = safe_columns.get(table_name, [])
                                    if safe_cols:
                                        modified_select = ', '.join(safe_cols)
                                        sql_query = re.sub(
                                            r"SELECT\s+" + re.escape(select_clause) + r"\s+FROM\s+" + re.escape(table_name),
                                            f"SELECT {modified_select} FROM {table_name}",
                                            sql_query,
                                            flags=re.IGNORECASE
                                        )
                                        logger.info(f"Replaced query with safe columns for '{table_name}': {safe_cols}")
                                    else:
                                        return False, f"Blocked by guardrails: no safe columns available for sensitive table '{table_name}'", sql_query

            # Optional: LLM-based secondary classification (only if security is enabled)
            if context.get("security_enabled", True):
                from config import Config
                if Config.GROQ_API_KEY:
                    try:
                        from langchain_groq import ChatGroq
                        llm = ChatGroq(
                            groq_api_key=Config.GROQ_API_KEY,
                            model_name="llama-3.1-8b-instant",
                            temperature=0.0,
                            max_tokens=64,
                        )
                        prompt = (
                            "Classify the following user request for database safety. "
                            "If it could cause destructive effects (e.g., DROP/TRUNCATE/ALTER/DELETE/INSERT/CREATE/GRANT/REVOKE, privilege escalation, exfiltration), label it UNSAFE. "
                            "If it looks like a harmless read/query request, label it SAFE. "
                            "Note: Queries on sensitive tables (like users) will be automatically modified to exclude sensitive data, so they should be labeled SAFE. "
                            "Respond with SAFE or UNSAFE only.\n\nRequest: " + (context.get("user_input") or "")
                        )
                        res = llm.invoke([
                            ("system", "You are a strict DB safety classifier."),
                            ("user", prompt),
                        ])
                        verdict = (res.content if hasattr(res, "content") else str(res)).strip().upper()
                        if "UNSAFE" in verdict and "SAFE" not in verdict:
                            return False, "Blocked by guardrails: the request was classified as potentially harmful to the database.", sql_query
                    except Exception:
                        # Non-fatal: skip if classifier unavailable
                        pass

            return True, "Security validation passed", sql_query
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return False, "Blocked by guardrails due to internal validation error", sql_query

class GroqSchemaBasedQueryEngine(SchemaBasedQueryEngine):
    """Schema-based query generation using Groq (LangChain ChatGroq)"""
    
    def __init__(self, groq_api_key: str):
        # Avoid OpenAI client setup; just initialize shared state
        self.groq_api_key = groq_api_key
        self.conversation_context = []
        self.current_table_context = None
        logger.info("Initialized GroqSchemaBasedQueryEngine with empty conversation context")
    
    def get_name(self) -> str:
        return "Schema-Based Querying (Groq)"
    
    def generate_query(self, user_query: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        logger.info(f"Starting generate_query (Groq) with user_query: '{user_query}'")
        logger.info(f"Current conversation context has {len(self.conversation_context)} interactions")
        
        try:
            # Ensure DB connection
            if not db_connection.is_connected():
                return False, "Not connected to database"
            
            # Get schema
            tables_success, tables_result = db_connection.get_tables()
            if not tables_success:
                return False, f"Failed to get tables: {tables_result}"
            if tables_result.empty:
                return False, "No tables found in the database"
            
            schema_context = self._build_schema_context(tables_result)
            if "WARNING: No tables found" in schema_context or "Available tables in this database:" not in schema_context:
                return False, "Failed to extract database schema properly"
            
            prompt = self._create_prompt(user_query, schema_context, context)
            logger.info("=== GROQ PROMPT ===")
            logger.info(f"User Query: {user_query}")
            logger.info(f"Schema Context: {schema_context}")
            logger.info(f"Full Prompt: {prompt}")
            logger.info("=== END PROMPT ===")
            
            # Lazy import to avoid hard dependency unless used
            try:
                from langchain_groq import ChatGroq
            except Exception as import_error:
                return False, f"Groq dependencies not available: {import_error}"
            
            llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=1000,
            )
            res = llm.invoke([
                ("system", "You are a SQL expert. Generate ONLY valid SQL queries. Do NOT include any explanatory text, comments, or markdown formatting. Return ONLY the SQL query itself."),
                ("user", prompt),
            ])
            sql_query = (res.content if hasattr(res, "content") else str(res)).strip()
            
            # Debug: Log the raw LLM response
            logger.info(f"Raw LLM response: '{sql_query}'")
            
            # Extract SQL from the response (in case LLM includes explanatory text)
            import re
            
            # First, try to extract from code blocks
            sql_match = re.search(r'```sql\s*(.*?)\s*```', sql_query, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql_query = sql_match.group(1).strip()
            else:
                # If no code blocks, try to find SQL in the text
                # Look for complete SELECT statements including LIMIT
                sql_match = re.search(r'(SELECT\s+.*?FROM\s+.*?)(?:\s+LIMIT\s+\d+)?(?:\s*;)?', sql_query, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    sql_query = sql_match.group(1).strip()
                    # Add LIMIT if not present and it's a SELECT query
                    if re.match(r'^\s*SELECT', sql_query, re.IGNORECASE) and 'LIMIT' not in sql_query.upper():
                        sql_query += ' LIMIT 10'
                else:
                    # If still no match, check if the entire response looks like SQL
                    if re.match(r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)', sql_query, re.IGNORECASE):
                        # It's already SQL, just clean it up
                        sql_query = sql_query.strip()
                        # Add LIMIT if it's a SELECT query without LIMIT
                        if re.match(r'^\s*SELECT', sql_query, re.IGNORECASE) and 'LIMIT' not in sql_query.upper():
                            sql_query += ' LIMIT 10'
                    else:
                        # If we can't find SQL, log the original response for debugging
                        logger.warning(f"Could not extract SQL from LLM response: {sql_query}")
                        return False, "Failed to generate valid SQL query from LLM response"
            
            # Clean up any trailing semicolon and whitespace
            sql_query = sql_query.rstrip(';').strip()
            
            # Validate that we have a complete SQL query
            if not sql_query or len(sql_query.strip()) < 10:
                logger.error(f"Generated SQL query is too short or empty: '{sql_query}'")
                # Try a simple fallback query for basic requests
                if "books" in user_query.lower():
                    sql_query = "SELECT id, isbn, title, author, genre, publication_year, total_copies, available_copies FROM books LIMIT 10"
                    logger.info(f"Using fallback query for books: {sql_query}")
                elif "users" in user_query.lower():
                    sql_query = "SELECT id, username, full_name, is_active, created_at FROM users LIMIT 10"
                    logger.info(f"Using fallback query for users: {sql_query}")
                else:
                    return False, "Generated SQL query is invalid or empty"
            
            # Additional validation: ensure the query is complete
            if sql_query.upper().endswith('FROM') or sql_query.upper().endswith('WHERE') or sql_query.upper().endswith('AND') or sql_query.upper().endswith('OR'):
                logger.error(f"Generated SQL query is incomplete: '{sql_query}'")
                # Use fallback for incomplete queries
                if "books" in user_query.lower():
                    sql_query = "SELECT id, isbn, title, author, genre, publication_year, total_copies, available_copies FROM books LIMIT 10"
                    logger.info(f"Using fallback query for incomplete books query: {sql_query}")
                elif "users" in user_query.lower():
                    sql_query = "SELECT id, username, full_name, is_active, created_at FROM users LIMIT 10"
                    logger.info(f"Using fallback query for incomplete users query: {sql_query}")
                else:
                    return False, "Generated SQL query is incomplete"
            
            # Postgres schema prefixing - only for actual table names
            if context.get('db_type') == 'postgresql':
                # Define known table names to prefix
                known_tables = ['users', 'books', 'book_loans', 'book_reviews']
                
                # Only prefix known table names
                for table in known_tables:
                    # Use word boundaries to ensure we match the whole table name
                    sql_query = re.sub(r'\bFROM\s+' + re.escape(table) + r'\b', r'FROM public.' + table, sql_query, flags=re.IGNORECASE)
                    sql_query = re.sub(r'\bJOIN\s+' + re.escape(table) + r'\b', r'JOIN public.' + table, sql_query, flags=re.IGNORECASE)
                    sql_query = re.sub(r'\bUPDATE\s+' + re.escape(table) + r'\b', r'UPDATE public.' + table, sql_query, flags=re.IGNORECASE)
                    sql_query = re.sub(r'\bINSERT\s+INTO\s+' + re.escape(table) + r'\b', r'INSERT INTO public.' + table, sql_query, flags=re.IGNORECASE)
                    sql_query = re.sub(r'\bDELETE\s+FROM\s+' + re.escape(table) + r'\b', r'DELETE FROM public.' + table, sql_query, flags=re.IGNORECASE)
                
                logger.info(f"Added schema prefix to query: {sql_query}")
            
            # Validate against schema
            if not self._validate_sql_against_schema(sql_query, tables_result):
                return False, "Generated SQL references non-existent tables or columns. Please try rephrasing your question."
            
            logger.info(f"Generated SQL query (Groq): {sql_query}")
            
            # Update conversation context
            table_name = self._extract_table_name_from_sql(sql_query)
            logger.info(f"Updating conversation context with table: {table_name}")
            self._update_conversation_context(user_query, sql_query, table_name)
            logger.info(f"Conversation context updated. Total interactions: {len(self.conversation_context)}")
            
            logger.info(f"Ending generate_query (Groq) successfully. Final context has {len(self.conversation_context)} interactions")
            return True, sql_query
        
        except Exception as e:
            error_msg = f"Failed to generate SQL query (Groq): {str(e)}"
            logger.error(error_msg)
            self._update_conversation_context(user_query, f"ERROR: {error_msg}", None)
            return False, error_msg

class QueryEngineFactory:
    """Factory for creating query engines"""
    
    @staticmethod
    def create_query_engine(engine_type: str, config: Dict[str, Any]) -> QueryEngine:
        """Create a query engine instance"""
        if engine_type == "schema":
            groq_key = config.get("groq_api_key") or config.get("GROQ_API_KEY")
            if groq_key:
                return GroqSchemaBasedQueryEngine(groq_key)
            api_key = config.get("openai_api_key")
            if not api_key:
                raise ValueError("OpenAI API key required for schema-based querying")
            return SchemaBasedQueryEngine(api_key)
        elif engine_type == "rag":
            return RAGQueryEngine()
        elif engine_type == "multitablejoin":
            return MultitablejoinQueryEngine(config.get('db_uri'), config.get('openai_api_key'), config.get('groq_api_key'))
        elif engine_type == "visualize":
            api_key = config.get('groq_api_key')
            if not api_key:
                raise ValueError("Groq API key required for visualization")
            return VisualizationQueryEngine(api_key)
        else:
            raise ValueError(f"Unknown query engine type: {engine_type}")
    
    @staticmethod
    def create_security_guardrail(guardrail_type: str, config: Dict[str, Any]) -> SecurityGuardrail:
        """Create a security guardrail instance"""
        if guardrail_type == "basic":
            return BasicSecurityGuardrail()
        else:
            raise ValueError(f"Unknown security guardrail type: {guardrail_type}")

# Global instances
query_engine_factory = QueryEngineFactory()
