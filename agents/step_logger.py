"""
Agent Step Logger - Clean, structured logging for RAG workflow steps
"""

import logging

logger = logging.getLogger(__name__)

class AgentStepLogger:
    """Clean step-by-step logging for RAG agent workflow"""
    
    @staticmethod
    def log_step(step_name: str, status: str, details: str = "", step_number: int = None):
        """
        Log a clean agent step
        
        Args:
            step_name: Name of the step (e.g., "Intent Detection", "Context Retrieval")  
            status: Status icon/text (e.g., "✅", "❌", "🔄")
            details: Optional details about the step
            step_number: Optional step number for sequence
        """
        if step_number:
            prefix = f"STEP {step_number}"
        else:
            prefix = "AGENT"
            
        if details:
            logger.info(f"{prefix} | {status} {step_name} - {details}")
        else:
            logger.info(f"{prefix} | {status} {step_name}")
    
    @staticmethod
    def log_intent_result(query: str, intent: str, rejected: bool):
        """Log intent detection result"""
        if rejected:
            AgentStepLogger.log_step("Intent Detection", "❌ REJECTED", f'"{query}" -> Not database-related', 1)
        else:
            AgentStepLogger.log_step("Intent Detection", "✅ ACCEPTED", f'"{query}" -> {intent}', 1)
    
    @staticmethod
    def log_context_retrieval(chunks_found: int, query: str):
        """Log context retrieval result"""
        if chunks_found > 0:
            AgentStepLogger.log_step("Context Retrieval", "📄 SUCCESS", f"{chunks_found} relevant chunks for '{query}'", 2)
        else:
            AgentStepLogger.log_step("Context Retrieval", "⚠️  WARNING", f"No context found for '{query}'", 2)
    
    @staticmethod
    def log_query_generation(sql_query: str):
        """Log query generation result"""
        preview = sql_query[:50] + "..." if len(sql_query) > 50 else sql_query
        AgentStepLogger.log_step("Query Generation", "⚡ GENERATED", f"SQL: {preview}", 3)
    
    @staticmethod
    def log_validation(layer: str, passed: bool, errors: list = None):
        """Log validation result"""
        step_num = {"syntactic": 4, "semantic": 5, "performance": 6}.get(layer.lower(), 4)
        layer_name = f"{layer.title()} Validation"
        
        if passed:
            AgentStepLogger.log_step(layer_name, "✅ PASSED", "", step_num)
        else:
            error_count = len(errors) if errors else 0
            AgentStepLogger.log_step(layer_name, "❌ FAILED", f"{error_count} errors found", step_num)
    
    @staticmethod
    def log_execution(success: bool, result_count: int = None, error: str = None):
        """Log query execution result"""
        if success:
            detail = f"{result_count} rows returned" if result_count is not None else "Success"
            AgentStepLogger.log_step("Query Execution", "🚀 SUCCESS", detail, 7)
        else:
            detail = f"Error: {error[:50]}..." if error and len(error) > 50 else error or "Failed"
            AgentStepLogger.log_step("Query Execution", "❌ FAILED", detail, 7)
    
    @staticmethod
    def log_correction(attempt: int, max_attempts: int):
        """Log self-correction attempt"""
        AgentStepLogger.log_step("Self-Correction", "🛠️  ATTEMPTING", f"Attempt {attempt}/{max_attempts}", 8)
    
    @staticmethod
    def log_final_result(success: bool, message: str = ""):
        """Log final workflow result"""
        if success:
            AgentStepLogger.log_step("Workflow Complete", "✅ SUCCESS", message, 9)
        else:
            AgentStepLogger.log_step("Workflow Complete", "❌ FAILED", message, 9)