from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from openai import OpenAI
import logging
from datetime import datetime
from database.connection import db_connection


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
    def validate_query(self, sql_query: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate SQL query for security concerns"""
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
            logger.error(f"SQL validation error: {str(e)}")
            return False

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
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
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
                self.visualization_service = VisualizationService(db_connection, self.openai_api_key)
            
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
    """Basic SQL injection protection (placeholder)"""
    
    def get_name(self) -> str:
        return "Basic Security Guardrails"
    
    def validate_query(self, sql_query: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Placeholder for security validation"""
        return True, "Security validation passed"

class QueryEngineFactory:
    """Factory for creating query engines"""
    
    @staticmethod
    def create_query_engine(engine_type: str, config: Dict[str, Any]) -> QueryEngine:
        """Create a query engine instance"""
        if engine_type == "schema":
            api_key = config.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key required for schema-based querying")
            return SchemaBasedQueryEngine(api_key)
        elif engine_type == "rag":
            return RAGQueryEngine()
        elif engine_type == "visualize":
            api_key = config.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key required for visualization")
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
