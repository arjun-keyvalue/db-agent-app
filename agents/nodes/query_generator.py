"""
Query generation node using LiteLLM
"""

import logging
from typing import Dict, Any
from ..llm_client import SmartLLMClient
from ..states import AgentState
from ..step_logger import AgentStepLogger

logger = logging.getLogger(__name__)


class QueryGeneratorNode:
    """Generate SQL queries using LiteLLM with RAG context"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    def __call__(self, state: AgentState) -> AgentState:
        """Generate SQL query using LLM with context"""

        user_query = state.get("user_query", "")
        database_schema = state.get("database_schema", "")
        relevant_context = state.get("relevant_context", [])
        database_type = state.get("database_type", "postgresql")
        user_intent = state.get("user_intent", {})

        if not user_query:
            state["error_message"] = "No user query provided"
            state["next_action"] = "output_formatter"
            return state

        try:
            # Create comprehensive prompt
            prompt = self._create_generation_prompt(
                user_query=user_query,
                database_schema=database_schema,
                relevant_context=relevant_context,
                database_type=database_type,
                user_intent=user_intent,
            )

            # Generate SQL using SmartLLMClient
            from ..llm_client import SmartLLMClient

            llm_client = SmartLLMClient()

            # Try LLM completion with rate limit handling
            try:
                response = llm_client.completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert SQL developer. Generate accurate, efficient SQL queries based on the provided schema and context.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,  # Low temperature for consistent SQL generation
                    max_tokens=1000,
                )
            except Exception as llm_error:
                # If rate limit or other LLM error, try basic SQL generation
                if "rate limit" in str(llm_error).lower():
                    logger.warning("LLM rate limited, using basic SQL generation")
                    sql_query = self._generate_basic_sql(user_query, database_schema)
                else:
                    raise llm_error
            else:
                # LLM succeeded, extract the SQL
                sql_query = response.choices[0].message.content.strip()

            # Clean up the response
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]

            sql_query = sql_query.strip()

            # Update state
            state["sql_query"] = sql_query
            state["next_action"] = "syntactic_validation"

            AgentStepLogger.log_query_generation(sql_query)

        except Exception as e:
            error_msg = f"Query generation failed: {str(e)}"
            logger.error(error_msg)
            state["error_message"] = error_msg
            state["next_action"] = "output_formatter"
            state["success"] = False

        return state

    def _create_generation_prompt(
        self,
        user_query: str,
        database_schema: str,
        relevant_context: list,
        database_type: str,
        user_intent: dict,
    ) -> str:
        """Create a comprehensive prompt for SQL generation"""

        prompt_parts = [
            f"Generate a SQL query for {database_type} database.",
            "",
            "USER REQUEST:",
            user_query,
            "",
            "DATABASE SCHEMA:",
            database_schema,
        ]

        # Add relevant context if available
        if relevant_context:
            prompt_parts.extend(
                [
                    "",
                    "RELEVANT CONTEXT:",
                    "\n".join(relevant_context),
                ]
            )

        # Add intent information if available
        if user_intent:
            intent_info = []
            if user_intent.get("primary_intent"):
                intent_info.append(f"Primary Intent: {user_intent['primary_intent']}")
            if user_intent.get("complexity"):
                intent_info.append(f"Complexity: {user_intent['complexity']}")
            if user_intent.get("requires_joins"):
                intent_info.append("Requires table joins")
            if user_intent.get("requires_aggregation"):
                intent_info.append("Requires aggregation functions")
            if user_intent.get("requires_filtering"):
                intent_info.append("Requires WHERE conditions")
            if user_intent.get("requires_sorting"):
                intent_info.append("Requires ORDER BY clause")

            if intent_info:
                prompt_parts.extend(
                    [
                        "",
                        "DETECTED INTENT:",
                        "\n".join(f"- {info}" for info in intent_info),
                    ]
                )

        prompt_parts.extend(
            [
                "",
                "REQUIREMENTS:",
                f"1. Generate valid {database_type} SQL syntax",
                "2. Only use tables and columns that exist in the schema above",
                "3. Do not invent or assume any table or column names",
                "4. Include appropriate JOINs if multiple tables are needed",
                "5. Add LIMIT clause for queries that might return many rows",
                "6. Use proper data types and functions for the database type",
                "7. Follow SQL best practices and optimization guidelines",
                "",
                "IMPORTANT:",
                "- If the request cannot be fulfilled with available schema, explain what's missing",
                "- Prefer explicit column names over SELECT *",
                "- Use table aliases for better readability in complex queries",
                "- Consider performance implications of your query design",
                "",
                "Return ONLY the SQL query, no explanations or formatting:",
            ]
        )

        return "\n".join(prompt_parts)
    
    def _generate_basic_sql(self, user_query: str, database_schema: str) -> str:
        """Generate basic SQL when LLM is unavailable (rate limited)"""
        query_lower = user_query.lower()
        
        # Extract table names from schema
        table_names = []
        if database_schema:
            lines = database_schema.split('\n')
            for line in lines:
                if line.strip().startswith('TABLE:') or 'table' in line.lower():
                    # Extract table name
                    if ':' in line:
                        table_name = line.split(':')[1].strip()
                        if table_name and not table_name.startswith('('):
                            table_names.append(table_name)
        
        # Default to a common table name if none found
        if not table_names:
            table_names = ['data', 'records', 'table1']
        
        # Try to match query to appropriate table
        target_table = table_names[0]  # default
        for table in table_names:
            if table.lower() in query_lower:
                target_table = table
                break
        
        # Generate basic SQL based on query patterns
        if any(word in query_lower for word in ['show', 'list', 'all', 'get']):
            return f"SELECT * FROM {target_table} LIMIT 10;"
        elif any(word in query_lower for word in ['count', 'how many']):
            return f"SELECT COUNT(*) FROM {target_table};"
        elif 'sum' in query_lower or 'total' in query_lower:
            return f"SELECT SUM(*) FROM {target_table};"
        else:
            return f"SELECT * FROM {target_table} LIMIT 10;"
