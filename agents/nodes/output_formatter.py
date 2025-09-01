"""
Output formatting node for final results
"""

import logging
from typing import Dict, Any
import pandas as pd
from ..states import AgentState

logger = logging.getLogger(__name__)


class OutputFormatterNode:
    """Format final output for the user"""
    
    def __call__(self, state: AgentState) -> AgentState:
        """Format the final output based on execution results"""
        
        success = state.get('success', False)
        query_successful = state.get('query_successful', False)
        execution_result = state.get('execution_result', None)
        error_message = state.get('error_message', '')
        sql_query = state.get('sql_query', '')
        correction_attempts = state.get('correction_attempts', 0)
        execution_time = state.get('execution_time', 0)
        
        if success and query_successful and execution_result is not None:
            # Successful execution
            formatted_output = self._format_success_output(
                execution_result, sql_query, correction_attempts, execution_time
            )
            state['final_result'] = execution_result
            
        else:
            # Failed execution
            formatted_output = self._format_error_output(
                error_message, sql_query, correction_attempts
            )
            state['final_result'] = None
        
        state['formatted_output'] = formatted_output
        state['is_complete'] = True
        
        logger.info("Output formatting completed")
        
        return state
    
    def _format_success_output(self, result, sql_query: str, attempts: int, exec_time: float) -> str:
        """Format successful query results"""
        
        output_parts = []
        
        # Add correction info if applicable
        if attempts > 0:
            output_parts.append(f"✅ Query succeeded after {attempts} correction(s)")
        else:
            output_parts.append("✅ Query executed successfully")
        
        # Add execution time
        output_parts.append(f"⏱️ Execution time: {exec_time:.3f}s")
        
        # Add SQL query
        output_parts.extend([
            "",
            "🔍 Generated SQL Query:",
            f"```sql\n{sql_query}\n```",
            ""
        ])
        
        # Add result summary
        if isinstance(result, pd.DataFrame):
            rows, cols = result.shape
            output_parts.append(f"📊 Results: {rows} rows, {cols} columns")
            
            if rows == 0:
                output_parts.append("No data returned by the query.")
            else:
                # Show first few rows as preview
                preview_rows = min(5, rows)
                output_parts.extend([
                    "",
                    f"Preview (first {preview_rows} rows):",
                    result.head(preview_rows).to_string(index=False)
                ])
                
                if rows > preview_rows:
                    output_parts.append(f"... and {rows - preview_rows} more rows")
        else:
            output_parts.append(f"📊 Result: {str(result)}")
        
        return "\n".join(output_parts)
    
    def _format_error_output(self, error_message: str, sql_query: str, attempts: int) -> str:
        """Format error output"""
        
        output_parts = []
        
        # Add error header
        if attempts > 0:
            output_parts.append(f"❌ Query failed after {attempts} correction attempt(s)")
        else:
            output_parts.append("❌ Query generation/execution failed")
        
        # Add error message
        output_parts.extend([
            "",
            "Error Details:",
            error_message
        ])
        
        # Add SQL query if available
        if sql_query:
            output_parts.extend([
                "",
                "🔍 Last Generated SQL Query:",
                f"```sql\n{sql_query}\n```"
            ])
        
        # Add troubleshooting tips
        output_parts.extend([
            "",
            "💡 Troubleshooting Tips:",
            "- Check if your question relates to the available database tables",
            "- Try rephrasing your question more specifically",
            "- Ensure you're asking for data that exists in the database",
            "- Consider breaking complex requests into simpler parts"
        ])
        
        return "\n".join(output_parts)