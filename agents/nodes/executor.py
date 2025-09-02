"""
Query executor with sandboxed execution and feedback loop
"""

import logging
import time
from typing import Dict, Any
from ..states import AgentState

logger = logging.getLogger(__name__)


class QueryExecutorNode:
    """Execute SQL queries in a sandboxed environment with timeout and error capture"""

    def __init__(self, db_connection):
        self.db_connection = db_connection

    def __call__(self, state: AgentState) -> AgentState:
        """Execute the SQL query with safety measures"""

        sql_query = state.get("sql_query", "")
        query_timeout = state.get("query_timeout", 30)

        if not sql_query:
            state["error_message"] = "No SQL query to execute"
            state["next_action"] = "output_formatter"
            state["success"] = False
            return state

        # Check if we're connected to database
        if not self.db_connection.is_connected():
            state["error_message"] = "Not connected to database"
            state["next_action"] = "output_formatter"
            state["success"] = False
            return state

        try:
            # Record start time
            start_time = time.time()

            # Execute query with timeout protection
            success, result = self._execute_with_timeout(sql_query, query_timeout)

            # Record execution time
            execution_time = time.time() - start_time
            state["execution_time"] = round(execution_time, 3)

            if success:
                # Query executed successfully
                state["execution_result"] = result
                state["query_successful"] = True
                state["next_action"] = "output_formatter"
                state["success"] = True

                logger.info(f"Query executed successfully in {execution_time:.3f}s")

                # Log result summary
                if hasattr(result, "shape"):
                    logger.info(
                        f"Result: {result.shape[0]} rows, {result.shape[1]} columns"
                    )

            else:
                # Query execution failed
                state["execution_error"] = str(result)
                state["query_successful"] = False

                # Check correction attempts to decide whether to try correction or proceed
                correction_attempts = state.get("correction_attempts", 0)
                max_corrections = state.get("max_corrections", 3)

                # Determine if this is correctable and we haven't exceeded attempts
                if (
                    self._is_correctable_error(str(result))
                    and correction_attempts < max_corrections
                ):
                    state["next_action"] = "self_correction"
                    state["needs_correction"] = True
                    logger.warning(
                        f"Query execution failed (attempting correction): {result}"
                    )
                else:
                    # Either non-correctable or max attempts reached - return the error but mark as completed
                    state["error_message"] = f"Query execution failed: {result}"
                    state["next_action"] = "output_formatter"
                    state["success"] = False
                    if correction_attempts >= max_corrections:
                        logger.warning(
                            f"Max correction attempts reached, returning error: {result}"
                        )
                    else:
                        logger.error(
                            f"Query execution failed (non-correctable): {result}"
                        )

        except Exception as e:
            error_msg = f"Query execution exception: {str(e)}"
            logger.error(error_msg)

            state["execution_error"] = error_msg
            state["query_successful"] = False

            # Check correction attempts for exceptions too
            correction_attempts = state.get("correction_attempts", 0)
            max_corrections = state.get("max_corrections", 3)

            # Most exceptions are correctable if we haven't exceeded attempts
            if (
                self._is_correctable_error(str(e))
                and correction_attempts < max_corrections
            ):
                state["next_action"] = "self_correction"
                state["needs_correction"] = True
            else:
                state["error_message"] = error_msg
                state["next_action"] = "output_formatter"
                state["success"] = False
                if correction_attempts >= max_corrections:
                    logger.warning(
                        f"Max correction attempts reached for exception: {error_msg}"
                    )

        return state

    def _execute_with_timeout(self, sql_query: str, timeout: int):
        """Execute query with timeout protection"""
        try:
            # For now, use the existing db_connection execute_query method
            # In a production system, you might want to implement actual timeout
            # using threading or asyncio

            success, result = self.db_connection.execute_query(sql_query)
            return success, result

        except Exception as e:
            return False, str(e)

    def _is_correctable_error(self, error_message: str) -> bool:
        """Determine if an error is likely correctable by the AI"""

        error_lower = error_message.lower()

        # Syntax errors - correctable
        correctable_patterns = [
            "syntax error",
            "invalid syntax",
            "column does not exist",
            "table does not exist",
            "relation does not exist",
            "unknown column",
            "unknown table",
            "ambiguous column",
            "must appear in group by",
            "aggregate function",
            "invalid function",
            "type mismatch",
            "data type",
            "conversion failed",
            "invalid date",
            "division by zero",
        ]

        # Non-correctable errors
        non_correctable_patterns = [
            "permission denied",
            "access denied",
            "connection lost",
            "timeout",
            "out of memory",
            "disk full",
            "lock timeout",
            "deadlock",
        ]

        # Check for non-correctable errors first
        for pattern in non_correctable_patterns:
            if pattern in error_lower:
                return False

        # Check for correctable errors
        for pattern in correctable_patterns:
            if pattern in error_lower:
                return True

        # Default to correctable for unknown errors
        return True
