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
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from services.visualization.service import VisualizationService

logger = logging.getLogger(__name__)


class QueryEngine(ABC):
    """Base interface for query engines"""

    @abstractmethod
    def generate_query(
        self, user_query: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
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


class SchemaBasedQueryEngine(QueryEngine):
    """Schema-based query generation using Groq (LangChain ChatGroq)"""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.conversation_context = []
        self.current_table_context = None

        # Initialize LangChain components
        self.llm = ChatGroq(
            api_key=groq_api_key, model="openai/gpt-oss-120b", temperature=0.0
        )

        # Create the SQL generation chain
        self.sql_chain = self._create_sql_chain()

        logger.debug("Initialized SchemaBasedQueryEngine")

    def get_name(self) -> str:
        return "Schema-Based Querying"

    def _create_sql_chain(self) -> LLMChain:
        """Create a LangChain for generating SQL queries from natural language."""

        sql_template = """
        You are a SQL expert specializing in generating queries for PostgreSQL databases.
        Given a user's natural language request, generate an appropriate SQL query.

        Database Schema:
        {schema_context}

        User Request: {user_query}

        IMPORTANT GUIDELINES:
        1. Generate SQL that directly answers the user's question
        2. Use appropriate JOINs when multiple tables are needed (ONLY WHEN REQUESTED)
        3. Include WHERE clauses for filtering when appropriate
        4. Use GROUP BY and aggregations (COUNT, SUM, AVG, etc.) when needed
        5. Include ORDER BY for better result presentation
        6. Limit results to reasonable numbers when appropriate
        7. Only use tables and columns that exist in the schema above
        8. Use proper PostgreSQL syntax
        9. For table names, use the exact names from the schema (without schema prefix unless specified)

        Generate ONLY the SQL query, no explanations, no markdown:
        """

        prompt = PromptTemplate(
            input_variables=["schema_context", "user_query"], template=sql_template
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def _build_schema_context(self) -> str:
        """Build schema context for the LLM"""
        try:
            if not db_connection.is_connected():
                return ""

            # Get tables and their schemas
            tables_success, tables_result = db_connection.get_tables()
            if not tables_success:
                return ""

            schema_context = "AVAILABLE DATABASE SCHEMA:\n\n"

            for _, table_row in tables_result.iterrows():
                table_name = table_row["table_name"]
                table_type = table_row["table_type"]

                if table_type == "BASE TABLE":
                    # Get table schema
                    schema_success, schema_result = db_connection.get_table_schema(
                        table_name
                    )
                    if schema_success:
                        schema_context += f"TABLE: {table_name}\n"
                        schema_context += "COLUMNS:\n"

                        for _, col_row in schema_result.iterrows():
                            col_name = col_row["column_name"]
                            col_type = col_row["data_type"]
                            nullable = col_row["is_nullable"]

                            schema_context += f"  - {col_name}: {col_type}"
                            if nullable == "NO":
                                schema_context += " (NOT NULL)"
                            schema_context += "\n"

                        schema_context += "\n"

            return schema_context

        except Exception as e:
            logger.error(f"Failed to build schema context: {str(e)}")
            return ""

    def generate_query(
        self, user_query: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Generate SQL query from natural language using LangChain."""
        try:
            # Get database schema context
            schema_context = self._build_schema_context()
            if not schema_context:
                return False, "Unable to access database schema"

            # Use the LangChain to generate SQL
            result = self.sql_chain.run(
                schema_context=schema_context, user_query=user_query
            )

            # Clean up the response
            sql_query = result.strip()

            # Remove markdown formatting if present
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]

            sql_query = sql_query.strip()

            # Add schema prefix for PostgreSQL if needed
            if db_connection.connection_info.get("type") == "postgresql":
                import re

                # Only add schema prefix if not already present
                sql_query = re.sub(
                    r"\bFROM\s+(\w+)\b(?!\.)",
                    r"FROM public.\1",
                    sql_query,
                    flags=re.IGNORECASE,
                )
                sql_query = re.sub(
                    r"\bJOIN\s+(\w+)\b(?!\.)",
                    r"JOIN public.\1",
                    sql_query,
                    flags=re.IGNORECASE,
                )

            logger.info(f"Generated SQL: {sql_query}")
            return True, sql_query

        except Exception as e:
            error_msg = f"SQL generation failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def execute_query(self, sql_query: str) -> Tuple[bool, Any]:
        """Execute the generated SQL query"""
        try:
            success, result = db_connection.execute_query(sql_query)
            if success:
                return True, result
            else:
                return False, result
        except Exception as e:
            return False, str(e)


class SecurityGuardrail(ABC):
    """Base interface for security guardrails"""

    @abstractmethod
    def validate_query(
        self, sql_query: str, context: Dict[str, Any]
    ) -> Tuple[bool, str, str]:
        """Validate SQL query for security concerns. Returns (success, message, modified_sql)"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the security guardrail"""
        pass


class MultitablejoinQueryEngine(QueryEngine):
    """Multitablejoin-based query generation"""

    def __init__(
        self,
        db_uri: str,
        openai_api_key: str,
        groq_api_key: str,
        lancedb_uri: str = "./lancedb",
    ):

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
                table_name = table_row["table_name"]
                table_data = {}
                table_data["table"] = table_name

                # Get schema
                schema = conn.execute(
                    text(f"PRAGMA table_info({table_name});")
                ).fetchall()
                schema_str = ", ".join([f"{col[1]} {col[2]}" for col in schema])
                table_data["schema"] = schema_str

                # Get foreign keys
                fkeys = conn.execute(
                    text(f"PRAGMA foreign_key_list({table_name});")
                ).fetchall()
                if fkeys:
                    fk_str = "; ".join(
                        [f"{table_name}.{fk[3]} → {fk[2]}.{fk[4]}" for fk in fkeys]
                    )
                    table_data["relationships"] = fk_str

                # Get sample rows
                rows = conn.execute(
                    text(f"SELECT * FROM {table_name} LIMIT 3;")
                ).fetchall()
                table_data["sample_rows"] = rows

                schema_data.append(table_data)

        # Create and store embeddings
        chunks = self.schema_embeddings.create_schema_chunks(schema_data)
        self.schema_embeddings.store_embeddings(chunks)

    def generate_query(
        self, user_query: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
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
            return (
                False,
                f"Query execution failed: {str(e)} if 'syntax' not in str(e).lower() else 'sorry'",
            )


class AdvancedRAGQueryEngine(QueryEngine):
    """Advanced RAG-based query generation with self-correction and validation"""

    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.model = model
        self.rag_agent = None
        self._last_result = None
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the RAG agent"""
        try:
            from agents.rag_agent import RAGAgent
            from config import Config

            # Use the provided API key or fall back to config
            api_key = self.openai_api_key or Config.get_rag_api_key()
            if not api_key:
                logger.warning(
                    "No API key provided for RAG agent, some features may not work"
                )

            self.rag_agent = RAGAgent(
                db_connection=db_connection,
                openai_api_key=api_key,
                model=Config.RAG_MODEL,
                lancedb_path=Config.LANCEDB_PATH,
            )
            logger.debug("Advanced RAG Agent initialized successfully")
            logger.debug(f"Using embedding provider: {Config.EMBEDDING_PROVIDER}")
            logger.debug(f"Using embedding model: {Config.EMBEDDING_MODEL}")

        except Exception as e:
            logger.error(f"Failed to initialize Advanced RAG Agent: {str(e)}")
            import traceback

            traceback.print_exc()
            self.rag_agent = None

    def get_name(self) -> str:
        return "RAG (Self-Correction and Validation)"

    def generate_query(
        self, user_query: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Generate query using RAG agent with self-correction"""
        if not self.rag_agent:
            return (
                False,
                "Advanced RAG Agent not initialized. Check dependencies: pip install langgraph litellm lancedb sqlfluff sqlglot",
            )

        try:
            from config import Config

            database_type = context.get("db_type", "postgresql")

            result = self.rag_agent.process_query(
                user_query=user_query,
                database_type=database_type,
                max_corrections=Config.MAX_CORRECTIONS,
                security_enabled=context.get("security_enabled", True),
            )

            # Store the full result for execute_query
            self._last_result = result

            if result["success"]:
                return True, result["sql_query"]
            else:
                return False, result["error_message"]

        except Exception as e:
            error_msg = f"Advanced RAG query generation failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def execute_query(self, sql_query: str) -> Tuple[bool, Any]:
        """Execute the provided SQL query directly to respect security guardrail modifications"""
        try:
            # Always execute the provided SQL query directly to ensure security guardrail modifications are respected
            # This is important because the security guardrail may have modified the query to exclude sensitive columns
            success, result = db_connection.execute_query(sql_query)
            
            if success:
                return True, result
            else:
                return False, result
                
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


class VisualizationQueryEngine(QueryEngine):
    """Visualization-based query generation using LangChain and Plotly"""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.visualization_service = None
        logger.info("Initialized VisualizationQueryEngine")

    def get_name(self) -> str:
        return "Data Visualization"

    def generate_query(
        self, user_query: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Generate visualization from user query.
        Note: This method returns visualization metadata instead of SQL.
        """
        try:
            # Lazy import to avoid circular dependencies
            if self.visualization_service is None:
                self.visualization_service = VisualizationService(
                    db_connection, self.groq_api_key
                )

            # Validate if the query is suitable for visualization
            is_valid, validation_message = (
                self.visualization_service.validate_visualization_request(user_query)
            )
            if not is_valid:
                return False, validation_message

            # Process the visualization request
            success, result = self.visualization_service.process_visualization_request(
                user_query, context
            )

            if success:
                # Store the visualization result for execute_query to return
                self._last_result = result
                # Return the SQL query that was generated for transparency
                return True, result["sql_query"]
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
            if hasattr(self, "_last_result"):
                result = self._last_result
                delattr(self, "_last_result")  # Clean up
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

    def validate_query(
        self, sql_query: str, context: Dict[str, Any]
    ) -> Tuple[bool, str, str]:
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
                return (
                    False,
                    "Blocked by guardrails: potentially destructive or unsafe database operation detected.",
                    sql_query,
                )

            # Sensitive data protection (only when security is enabled)
            if context.get("security_enabled", True):
                # Inspect SELECT columns for sensitive identifiers and modify queries
                import re

                # Define sensitive tables that contain PII or credentials
                sensitive_tables = [
                    "users", "user", "accounts", "account", 
                    "employees", "employee", "employee_management"
                ]

                # Define safe columns for sensitive tables (columns that can be safely exposed)
                safe_columns = {
                    "users": [
                        "id",
                        "username",
                        "full_name",
                        "is_active",
                        "created_at",
                        "updated_at",
                    ],
                    "user": [
                        "id",
                        "username",
                        "full_name",
                        "is_active",
                        "created_at",
                        "updated_at",
                    ],
                    "accounts": [
                        "id",
                        "username",
                        "is_active",
                        "created_at",
                        "updated_at",
                    ],
                    "account": [
                        "id",
                        "username",
                        "is_active",
                        "created_at",
                        "updated_at",
                    ],
                    "employees": [
                        "employee_id",
                        "first_name",
                        "last_name",
                        "hire_date",
                        "status",
                        "role_id",
                        "manager_id",
                        "team_id",
                    ],
                    "employee": [
                        "employee_id",
                        "first_name",
                        "last_name",
                        "hire_date",
                        "status",
                        "role_id",
                        "manager_id",
                        "team_id",
                    ],
                    "employee_management": [
                        "employee_id",
                        "first_name",
                        "last_name",
                        "hire_date",
                        "status",
                        "role_id",
                        "manager_id",
                        "team_id",
                    ],
                }

                # Check user intent first - if they're explicitly asking for sensitive data, block the query
                user_input = context.get("user_input", "").lower()
                sensitive_intent_markers = [
                    "email",
                    "emails",
                    "e-mail",
                    "e-mails",
                    "phone",
                    "phone number",
                    "phone numbers",
                    "telephone",
                    "telephone number",
                    "password",
                    "passwords",
                    "passwd",
                    "pwd",
                    "date of birth",
                    "birth date",
                    "dob",
                    "birthday",
                    "social security",
                    "ssn",
                    "ss number",
                    "credit card",
                    "cc number",
                    "card number",
                    "address",
                    "home address",
                    "mailing address",
                    "personal information",
                    "pii",
                    "private data",
                ]

                # If user is explicitly asking for sensitive data, block the query
                if any(marker in user_input for marker in sensitive_intent_markers):
                    return (
                        False,
                        "Blocked by guardrails: request for sensitive personal information is not allowed.",
                        sql_query,
                    )

                # Check for SELECT queries on sensitive tables
                select_match = re.search(
                    r"SELECT\s+(.*?)\s+FROM\s+([\w.]+)",
                    sql_query,
                    re.IGNORECASE | re.DOTALL,
                )
                if select_match:
                    select_clause = select_match.group(1).strip()
                    full_table_name = select_match.group(2).strip()
                    # Extract table name without schema prefix
                    table_name = full_table_name.split('.')[-1].lower()

                    # If querying a sensitive table, modify the SELECT clause to only include safe columns
                    if table_name in sensitive_tables:
                        if select_clause == "*":
                            # Replace SELECT * with only safe columns
                            safe_cols = safe_columns.get(table_name, [])
                            if safe_cols:
                                modified_select = ", ".join(safe_cols)
                                sql_query = re.sub(
                                    r"SELECT\s+\*\s+FROM\s+" + re.escape(full_table_name),
                                    f"SELECT {modified_select} FROM {full_table_name}",
                                    sql_query,
                                    flags=re.IGNORECASE,
                                )
                                logger.info(
                                    f"Modified SELECT * query on sensitive table '{table_name}' to only include safe columns: {safe_cols}"
                                )
                            else:
                                return (
                                    False,
                                    f"Blocked by guardrails: no safe columns defined for sensitive table '{table_name}'",
                                    sql_query,
                                )
                        else:
                            # Check if any sensitive columns are explicitly selected
                            columns = [c.strip() for c in select_clause.split(",")]
                            sensitive_cols_found = []
                            safe_cols_to_keep = []

                            for col in columns:
                                # Normalize column name
                                normalized_col = col.lower()
                                # Remove AS alias
                                if " as " in normalized_col:
                                    normalized_col = normalized_col.split(" as ")[
                                        0
                                    ].strip()
                                # Remove table prefix
                                if "." in normalized_col:
                                    normalized_col = normalized_col.split(".")[
                                        -1
                                    ].strip()

                                # Check if it's a sensitive column
                                if re.search(
                                    r"\b(user_?name|username|password|passwd|password_?hash|pwd_hash|email|phone|phone_?number|address|date_?of_?birth|dob|ssn|social_?security|credit_?card|card_?number)\b",
                                    normalized_col,
                                ):
                                    sensitive_cols_found.append(col.strip())
                                else:
                                    safe_cols_to_keep.append(col.strip())

                            # If sensitive columns were found, remove them from the query
                            if sensitive_cols_found:
                                if safe_cols_to_keep:
                                    # Replace the SELECT clause with only safe columns
                                    modified_select = ", ".join(safe_cols_to_keep)
                                    sql_query = re.sub(
                                        r"SELECT\s+"
                                        + re.escape(select_clause)
                                        + r"\s+FROM\s+"
                                        + re.escape(full_table_name),
                                        f"SELECT {modified_select} FROM {full_table_name}",
                                        sql_query,
                                        flags=re.IGNORECASE,
                                    )
                                    logger.info(
                                        f"Removed sensitive columns from query on '{table_name}': {sensitive_cols_found}"
                                    )
                                else:
                                    # If no safe columns remain, use default safe columns
                                    safe_cols = safe_columns.get(table_name, [])
                                    if safe_cols:
                                        modified_select = ", ".join(safe_cols)
                                        sql_query = re.sub(
                                            r"SELECT\s+"
                                            + re.escape(select_clause)
                                            + r"\s+FROM\s+"
                                            + re.escape(full_table_name),
                                            f"SELECT {modified_select} FROM {full_table_name}",
                                            sql_query,
                                            flags=re.IGNORECASE,
                                        )
                                        logger.info(
                                            f"Replaced query with safe columns for '{table_name}': {safe_cols}"
                                        )
                                    else:
                                        return (
                                            False,
                                            f"Blocked by guardrails: no safe columns available for sensitive table '{table_name}'",
                                            sql_query,
                                        )

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
                            "Note: Queries on sensitive tables (like users, employees) will be automatically modified to exclude sensitive data, so they should be labeled SAFE. "
                            "Respond with SAFE or UNSAFE only.\n\nRequest: "
                            + (context.get("user_input") or "")
                        )
                        res = llm.invoke(
                            [
                                ("system", "You are a strict DB safety classifier."),
                                ("user", prompt),
                            ]
                        )
                        verdict = (
                            (res.content if hasattr(res, "content") else str(res))
                            .strip()
                            .upper()
                        )
                        if "UNSAFE" in verdict and "SAFE" not in verdict:
                            return (
                                False,
                                "Blocked by guardrails: the request was classified as potentially harmful to the database.",
                                sql_query,
                            )
                    except Exception:
                        # Non-fatal: skip if classifier unavailable
                        pass

            return True, "Security validation passed", sql_query
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return (
                False,
                "Blocked by guardrails due to internal validation error",
                sql_query,
            )


class QueryEngineFactory:
    """Factory for creating query engines"""

    @staticmethod
    def create_query_engine(engine_type: str, config: Dict[str, Any]) -> QueryEngine:
        """Create a query engine instance"""
        if engine_type == "schema":
            api_key = config.get("groq_api_key")
            if not api_key:
                raise ValueError("Groq API key required for schema-based querying")
            return SchemaBasedQueryEngine(api_key)
        elif engine_type == "rag":
            api_key = config.get("openai_api_key")
            if not api_key:
                raise ValueError("OpenAI API key required for basic RAG")
            return AdvancedRAGQueryEngine(api_key)
        elif engine_type == "rag_advanced":
            api_key = config.get("openai_api_key")
            if not api_key:
                raise ValueError("OpenAI API key required for RAG with self-correction")
            return AdvancedRAGQueryEngine(api_key)
        elif engine_type == "multitablejoin":
            return MultitablejoinQueryEngine(
                config.get("db_uri"),
                config.get("openai_api_key"),
                config.get("groq_api_key"),
            )
        elif engine_type == "visualize":
            api_key = config.get("groq_api_key")
            if not api_key:
                raise ValueError("Groq API key required for visualization")
            return VisualizationQueryEngine(api_key)
        else:
            raise ValueError(f"Unknown query engine type: {engine_type}")

    @staticmethod
    def create_security_guardrail(
        guardrail_type: str, config: Dict[str, Any]
    ) -> SecurityGuardrail:
        """Create a security guardrail instance"""
        if guardrail_type == "basic":
            return BasicSecurityGuardrail()
        else:
            raise ValueError(f"Unknown security guardrail type: {guardrail_type}")


# Global instances
query_engine_factory = QueryEngineFactory()
