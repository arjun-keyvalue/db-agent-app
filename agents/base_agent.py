"""
Base agent class for all database agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer


class BaseAgent(ABC):
    """Base class for all database agents"""
    
    def __init__(self, checkpointer: Optional[Checkpointer] = None):
        self.state = self.define_agent_state()
        self.graph = StateGraph(self.state)
        self.checkpointer = checkpointer
        self.define_graph()
        self.compiled_graph = None
    
    @abstractmethod
    def define_agent_state(self):
        """Define the state schema for the agent"""
        pass
    
    @abstractmethod
    def define_graph(self):
        """Define the graph structure and nodes"""
        pass
    
    def compile(self):
        """Compile the graph"""
        if self.checkpointer:
            self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        else:
            self.compiled_graph = self.graph.compile()
        return self.compiled_graph
    
    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the compiled graph with initial state"""
        if not self.compiled_graph:
            self.compile()
        return self.compiled_graph.invoke(initial_state)
    
    def export_graph_png(self, output_path: str = "agents/graph.png"):
        """Export the compiled graph as a PNG using Mermaid rendering."""
        if not self.compiled_graph:
            self.compile()
        self.compiled_graph.get_graph().draw_mermaid_png(output_file_path=output_path)