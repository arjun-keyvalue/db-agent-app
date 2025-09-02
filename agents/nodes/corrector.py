"""
Self-correction node - Layer 3: AI-Powered Self-Correction with execution feedback loop
"""

import logging
from typing import Dict, Any
from ..llm_client import SmartLLMClient
from ..states import AgentState

logger = logging.getLogger(__name__)


class SelfCorrectionNode:
    """Layer 3: AI-powered self-correction with execution feedback loop"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.llm_client = SmartLLMClient()

    def __call__(self, state: AgentState) -> AgentState:
        """Perform self-correction based on validation errors or execution feedback"""

        correction_attempts = state.get("correction_attempts", 0)
        max_corrections = state.get("max_corrections", 3)

        if correction_attempts >= max_corrections:
            state["error_message"] = (
                f"Maximum correction attempts ({max_corrections}) exceeded"
            )
            state["next_action"] = "output_formatter"
            state["success"] = False
            return state

        # Increment correction attempts
        state["correction_attempts"] = correction_attempts + 1

        # Gather all available feedback
        feedback_parts = []

        # Validation errors
        validation_errors = state.get("validation_errors", [])
        if validation_errors:
            feedback_parts.append("Validation Errors:")
            for error in validation_errors:
                feedback_parts.append(f"- {error}")

        # Execution errors
        execution_error = state.get("execution_error", "")
        if execution_error:
            feedback_parts.append(f"Execution Error: {execution_error}")

        # Performance issues
        if not state.get("has_limit_clause", True):
            feedback_parts.append(
                "Performance Issue: Query lacks LIMIT clause for potentially large result sets"
            )

        feedback = "\n".join(feedback_parts)

        # Create correction prompt
        correction_prompt = self._create_correction_prompt(
            user_query=state.get("user_query", ""),
            current_sql=state.get("sql_query", ""),
            database_schema=state.get("database_schema", ""),
            feedback=feedback,
            database_type=state.get("database_type", "postgresql"),
            attempt_number=state["correction_attempts"],
        )

        try:
            # Use SmartLLMClient for model flexibility
            response = self.llm_client.completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL developer with strong debugging skills. Your job is to fix SQL queries based on error feedback.",
                    },
                    {"role": "user", "content": correction_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent corrections
                max_tokens=1000,
            )

            corrected_sql = response.choices[0].message.content.strip()

            # Clean up the response
            if corrected_sql.startswith("```sql"):
                corrected_sql = corrected_sql[6:]
            if corrected_sql.endswith("```"):
                corrected_sql = corrected_sql[:-3]

            corrected_sql = corrected_sql.strip()

            # Update state with corrected query
            state["sql_query"] = corrected_sql
            state["correction_feedback"] = feedback
            state["validation_errors"] = []  # Reset validation errors
            state["execution_error"] = ""  # Reset execution error

            # Reset validation flags to re-validate
            state["syntax_valid"] = False
            state["semantic_valid"] = False
            state["query_successful"] = False

            # Next action is to re-validate
            state["next_action"] = "syntactic_validation"

            logger.debug(
                f"Self-correction attempt {state['correction_attempts']} completed"
            )

        except Exception as e:
            error_msg = f"Self-correction failed: {str(e)}"
            logger.error(error_msg)
            state["error_message"] = error_msg
            state["next_action"] = "output_formatter"
            state["success"] = False

        return state

    def _create_correction_prompt(
        self,
        user_query: str,
        current_sql: str,
        database_schema: str,
        feedback: str,
        database_type: str,
        attempt_number: int,
    ) -> str:
        """Create a detailed correction prompt for the LLM"""

        prompt = f"""
You are debugging a SQL query that has errors. This is correction attempt #{attempt_number}.

ORIGINAL USER REQUEST:
{user_query}

CURRENT SQL QUERY (with errors):
{current_sql}

DATABASE TYPE: {database_type}

DATABASE SCHEMA:
{database_schema}

ERROR FEEDBACK:
{feedback}

INSTRUCTIONS:
1. Carefully analyze the error feedback above
2. Identify the specific issues in the current SQL query
3. Generate a corrected SQL query that addresses ALL the issues
4. Ensure the corrected query:
   - Has correct syntax for {database_type}
   - Only references tables and columns that exist in the schema
   - Includes appropriate LIMIT clauses for large result sets
   - Follows SQL best practices
   - Answers the original user request

IMPORTANT RULES:
- Only use tables and columns that are explicitly listed in the database schema
- Do not invent or assume any table or column names
- If the user request cannot be fulfilled with the available schema, explain what's missing
- Always include a LIMIT clause unless the query is an aggregation (COUNT, SUM, etc.)
- Use proper {database_type} syntax and functions

Return ONLY the corrected SQL query, no explanations or markdown formatting:
"""

        return prompt
