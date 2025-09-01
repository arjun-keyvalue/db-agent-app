"""
RAG Agent with Self-Correction and Validation
Implements 4-layer validation and correction system
"""

import os
import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.types import Checkpointer

from .base_agent import BaseAgent
from .states import AgentState
from .nodes import (
    IntentDetectorNode,
    SchemaRetrieverNode,
    ContextRetrieverNode,
    QueryGeneratorNode,
    SyntacticValidatorNode,
    SemanticValidatorNode,
    PerformanceGuardNode,
    QueryExecutorNode,
    SelfCorrectionNode,
    OutputFormatterNode
)

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    RAG Agent with 4-layer validation and self-correction:
    
    Layer 1: Syntactic Validation (SQLFluff, SQLGlot)
    Layer 2: Semantic Validation (Schema verification)
    Layer 3: AI-Powered Self-Correction (Execution feedback loop)
    Layer 4: Performance Guardrails (Timeout, LIMIT clauses)
    """
    
    def __init__(self, db_connection, openai_api_key: str, 
                 model: str = "gpt-3.5-turbo", 
                 lancedb_path: str = "./lancedb_rag",
                 checkpointer: Optional[Checkpointer] = None):
        
        self.db_connection = db_connection
        self.openai_api_key = openai_api_key
        self.model = model
        self.lancedb_path = lancedb_path
        
        # Initialize schema indexer for embeddings
        from .schema_indexer import get_schema_indexer
        self.schema_indexer = get_schema_indexer()
        
        # Auto-index schema if enabled
        from config import Config
        if Config.AUTO_INDEX_SCHEMA:
            logger.info("Auto-indexing database schema...")
            self.schema_indexer.index_database_schema(db_connection)
        
        # Initialize nodes
        self._initialize_nodes()
        
        # Initialize base agent
        super().__init__(checkpointer)
        
        # Save graph visualization
        self._save_graph_visualization()
        
        logger.info("RAG Agent initialized with self-correction capabilities")
    
    def _initialize_nodes(self):
        """Initialize all workflow nodes"""
        self.intent_detector = IntentDetectorNode(self.model)
        self.schema_retriever = SchemaRetrieverNode(self.db_connection)
        self.context_retriever = ContextRetrieverNode(self.openai_api_key, self.lancedb_path)
        self.query_generator = QueryGeneratorNode(self.model)
        self.syntactic_validator = SyntacticValidatorNode()
        self.semantic_validator = SemanticValidatorNode(self.db_connection)
        self.performance_guard = PerformanceGuardNode()
        self.query_executor = QueryExecutorNode(self.db_connection)
        self.self_corrector = SelfCorrectionNode(self.model)
        self.output_formatter = OutputFormatterNode()
    
    def define_agent_state(self):
        """Define the agent state schema"""
        return AgentState
    
    def define_graph(self):
        """Define the RAG agent workflow graph"""
        
        # Add all nodes
        self.graph.add_node("intent_detector", self.intent_detector)
        self.graph.add_node("schema_retriever", self.schema_retriever)
        self.graph.add_node("context_retriever", self.context_retriever)
        self.graph.add_node("query_generator", self.query_generator)
        self.graph.add_node("syntactic_validation", self.syntactic_validator)
        self.graph.add_node("semantic_validation", self.semantic_validator)
        self.graph.add_node("performance_guard", self.performance_guard)
        self.graph.add_node("query_executor", self.query_executor)
        self.graph.add_node("self_correction", self.self_corrector)
        self.graph.add_node("output_formatter", self.output_formatter)
        
        # Define the workflow edges
        
        # Start -> Intent Detection -> Schema Retrieval
        self.graph.add_edge(START, "intent_detector")
        self.graph.add_edge("intent_detector", "schema_retriever")
        
        # Schema -> Context -> Query Generation
        self.graph.add_edge("schema_retriever", "context_retriever")
        self.graph.add_edge("context_retriever", "query_generator")
        
        # Query Generation -> Syntactic Validation
        self.graph.add_edge("query_generator", "syntactic_validation")
        
        # Syntactic Validation -> [Semantic Validation | Self-Correction]
        self.graph.add_conditional_edges(
            "syntactic_validation",
            lambda state: state.get("next_action", ""),
            {
                "semantic_validation": "semantic_validation",
                "self_correction": "self_correction"
            }
        )
        
        # Semantic Validation -> [Performance Guard | Self-Correction]
        self.graph.add_conditional_edges(
            "semantic_validation",
            lambda state: state.get("next_action", ""),
            {
                "performance_guard": "performance_guard",
                "self_correction": "self_correction"
            }
        )
        
        # Performance Guard -> [Query Executor | Self-Correction]
        self.graph.add_conditional_edges(
            "performance_guard",
            lambda state: state.get("next_action", ""),
            {
                "query_executor": "query_executor",
                "self_correction": "self_correction"
            }
        )
        
        # Query Executor -> [Output Formatter | Self-Correction]
        self.graph.add_conditional_edges(
            "query_executor",
            lambda state: state.get("next_action", ""),
            {
                "output_formatter": "output_formatter",
                "self_correction": "self_correction"
            }
        )
        
        # Self-Correction -> Syntactic Validation (retry loop)
        self.graph.add_conditional_edges(
            "self_correction",
            lambda state: state.get("next_action", ""),
            {
                "syntactic_validation": "syntactic_validation",
                "output_formatter": "output_formatter"
            }
        )
        
        # Output Formatter -> END
        self.graph.add_edge("output_formatter", END)
    
    def process_query(self, user_query: str, database_type: str = "postgresql", 
                     max_corrections: int = 3) -> Dict[str, Any]:
        """
        Process a user query through the RAG agent workflow
        
        Args:
            user_query: Natural language query from user
            database_type: Type of database (postgresql, mysql, sqlite3)
            max_corrections: Maximum number of self-correction attempts
            
        Returns:
            Dict containing results, SQL query, and metadata
        """
        
        # Initialize state
        initial_state = {
            "user_query": user_query,
            "database_type": database_type,
            "user_intent": {},
            "database_schema": "",
            "relevant_context": [],
            "sql_query": "",
            "validation_errors": [],
            "syntax_valid": False,
            "semantic_valid": False,
            "execution_result": None,
            "execution_error": "",
            "query_successful": False,
            "correction_attempts": 0,
            "max_corrections": max_corrections,
            "needs_correction": False,
            "correction_feedback": "",
            "query_timeout": 30,
            "has_limit_clause": False,
            "estimated_cost": 0.0,
            "final_result": None,
            "formatted_output": "",
            "error_message": "",
            "next_action": "",
            "is_complete": False,
            "execution_time": 0.0,
            "total_attempts": 0,
            "success": False
        }
        
        try:
            # Execute the workflow
            final_state = self.invoke(initial_state)
            
            # Extract results
            return {
                "success": final_state.get("success", False),
                "sql_query": final_state.get("sql_query", ""),
                "result": final_state.get("final_result", None),
                "formatted_output": final_state.get("formatted_output", ""),
                "error_message": final_state.get("error_message", ""),
                "execution_time": final_state.get("execution_time", 0.0),
                "correction_attempts": final_state.get("correction_attempts", 0),
                "estimated_cost": final_state.get("estimated_cost", 0.0),
                "metadata": {
                    "syntax_valid": final_state.get("syntax_valid", False),
                    "semantic_valid": final_state.get("semantic_valid", False),
                    "has_limit_clause": final_state.get("has_limit_clause", False),
                    "query_timeout": final_state.get("query_timeout", 30),
                    "total_attempts": final_state.get("total_attempts", 0)
                }
            }
        
        except Exception as e:
            error_msg = f"RAG Agent workflow failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "sql_query": "",
                "result": None,
                "formatted_output": error_msg,
                "error_message": error_msg,
                "execution_time": 0.0,
                "correction_attempts": 0,
                "estimated_cost": 0.0,
                "metadata": {}
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and capabilities"""
        return {
            "agent_type": "RAG Agent with Self-Correction",
            "capabilities": [
                "Syntactic SQL validation (SQLFluff, SQLGlot)",
                "Semantic schema validation",
                "AI-powered self-correction with execution feedback",
                "Performance guardrails and safety measures",
                "Vector-based context retrieval (LanceDB)",
                "Multi-model support via LiteLLM"
            ],
            "validation_layers": [
                "Layer 1: Syntactic Validation",
                "Layer 2: Semantic Validation", 
                "Layer 3: AI-Powered Self-Correction",
                "Layer 4: Performance Guardrails"
            ],
            "database_support": ["PostgreSQL", "MySQL", "SQLite3"],
            "vector_storage": "LanceDB (embedded)",
            "llm_backend": "LiteLLM (multi-model)"
        }
    
    def _save_graph_visualization(self):
        """Save the graph as a PNG visualization"""
        try:
            # Create agents directory if it doesn't exist
            os.makedirs("agents", exist_ok=True)
            
            # Compile the graph first if not already compiled
            if not self.compiled_graph:
                self.compile()
            
            # Generate and save the graph visualization
            graph_image = self.compiled_graph.get_graph().draw_mermaid_png()
            
            with open("agents/rag_workflow_graph.png", "wb") as f:
                f.write(graph_image)
            
            logger.info("RAG workflow graph saved as agents/rag_workflow_graph.png")
            
        except Exception as e:
            logger.warning(f"Could not save graph visualization: {e}")
            # Try alternative method without mermaid
            try:
                if not self.compiled_graph:
                    self.compile()
                graph_ascii = self.compiled_graph.get_graph().draw_ascii()
                with open("agents/rag_workflow_graph.txt", "w") as f:
                    f.write(graph_ascii)
                logger.info("RAG workflow graph saved as agents/rag_workflow_graph.txt")
            except Exception as e2:
                logger.warning(f"Could not save ASCII graph either: {e2}")
                # Create a simple text description as fallback
                try:
                    with open("agents/rag_workflow_description.txt", "w") as f:
                        f.write("RAG Agent Workflow:\n")
                        f.write("1. Intent Detection\n")
                        f.write("2. Schema Retrieval\n")
                        f.write("3. Context Retrieval\n")
                        f.write("4. Query Generation\n")
                        f.write("5. Syntactic Validation\n")
                        f.write("6. Semantic Validation\n")
                        f.write("7. Performance Guard\n")
                        f.write("8. Query Execution\n")
                        f.write("9. Self-Correction (if needed)\n")
                        f.write("10. Output Formatting\n")
                    logger.info("RAG workflow description saved as agents/rag_workflow_description.txt")
                except Exception as e3:
                    logger.warning(f"Could not save workflow description: {e3}")