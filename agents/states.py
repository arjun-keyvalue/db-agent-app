"""
State definitions for agents
"""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Main agent state for RAG agent with self-correction"""
    
    # Input
    user_query: str
    database_type: str
    
    # Intent detection
    user_intent: Dict[str, Any]
    
    # Schema and context
    database_schema: str
    relevant_context: List[str]
    
    # Query generation and validation
    sql_query: str
    validation_errors: List[str]
    syntax_valid: bool
    semantic_valid: bool
    
    # Execution and results
    execution_result: Any
    execution_error: str
    query_successful: bool
    
    # Self-correction
    correction_attempts: int
    max_corrections: int
    needs_correction: bool
    correction_feedback: str
    
    # Performance and safety
    query_timeout: int
    has_limit_clause: bool
    estimated_cost: float
    
    # Output
    final_result: Any
    formatted_output: str
    error_message: str
    
    # Flow control
    next_action: str
    is_complete: bool
    
    # Metadata
    execution_time: float
    total_attempts: int
    success: bool