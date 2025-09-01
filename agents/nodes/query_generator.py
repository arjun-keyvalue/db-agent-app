"""
Query generation node using LiteLLM
"""

import logging
from typing import Dict, Any
from ..llm_client import SmartLLMClient
from ..states import AgentState

logger = logging.getLogger(__name__)


class QueryGeneratorNode:
    """Generate SQL queries using LiteLLM with RAG context"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
    
    def __call__(self, state: AgentState) -> AgentState:
        """Generate SQL query using LLM with context"""
        
        user_query = state.get('user_query', '')
        database_schema = state.get('database_schema', '')
        relevant_context = state.get('relevant_context', [])
        database_type = state.get('database_type', 'postgresql')
        user_intent = state.get('user_intent', {})
        
        if not user_query:
            state['error_message'] = "No user query provided"
            state['next_action'] = 'output_formatter'
            return state
        
        try:
            # Create comprehensive prompt
            prompt = self._create_generation_prompt(
                user_query=user_query,
                database_schema=database_schema,
                relevant_context=relevant_context,
                database_type=database_type,
                user_intent=user_intent
            )
            
            # Generate SQL using LiteLLM
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL developer. Generate accurate, efficient SQL queries based on the provided schema and context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_tokens=1000
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            sql_query = sql_query.strip()
            
            # Update state
            state['sql_query'] = sql_query
            state['next_action'] = 'syntactic_validation'
            
            logger.info("SQL query generated successfully")
            
        except Exception as e:
            error_msg = f"Query generation failed: {str(e)}"
            logger.error(error_msg)
            state['error_message'] = error_msg
            state['next_action'] = 'output_formatter'
            state['success'] = False
        
        return state
    
    def _create_generation_prompt(self, user_query: str, database_schema: str, 
                                relevant_context: list, database_type: str, user_intent: dict) -> str:
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
            prompt_parts.extend([
                "",
                "RELEVANT CONTEXT:",
                "\n".join(relevant_context),
            ])
        
        # Add intent information if available
        if user_intent:
            intent_info = []
            if user_intent.get('primary_intent'):
                intent_info.append(f"Primary Intent: {user_intent['primary_intent']}")
            if user_intent.get('complexity'):
                intent_info.append(f"Complexity: {user_intent['complexity']}")
            if user_intent.get('requires_joins'):
                intent_info.append("Requires table joins")
            if user_intent.get('requires_aggregation'):
                intent_info.append("Requires aggregation functions")
            if user_intent.get('requires_filtering'):
                intent_info.append("Requires WHERE conditions")
            if user_intent.get('requires_sorting'):
                intent_info.append("Requires ORDER BY clause")
            
            if intent_info:
                prompt_parts.extend([
                    "",
                    "DETECTED INTENT:",
                    "\n".join(f"- {info}" for info in intent_info),
                ])
        
        prompt_parts.extend([
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
            "Return ONLY the SQL query, no explanations or formatting:"
        ])
        
        return "\n".join(prompt_parts)