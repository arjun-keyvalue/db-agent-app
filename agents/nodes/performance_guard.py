"""
Performance guardrails - Layer 4: AI-Aware Performance Guardrails
"""

import logging
import re
from typing import Dict, Any
from ..states import AgentState

logger = logging.getLogger(__name__)


class PerformanceGuardNode:
    """Layer 4: AI-aware performance guardrails for safe query execution"""
    
    def __init__(self, default_timeout: int = 30, max_limit: int = 1000):
        self.default_timeout = default_timeout
        self.max_limit = max_limit
    
    def __call__(self, state: AgentState) -> AgentState:
        """Apply performance guardrails to the validated SQL query"""
        
        sql_query = state.get('sql_query', '')
        
        if not sql_query:
            state['error_message'] = "No SQL query to apply performance guardrails"
            state['next_action'] = 'output_formatter'
            return state
        
        # Apply guardrails
        modified_query = sql_query
        guardrail_applied = False
        
        # 1. Check for LIMIT clause
        has_limit = self._has_limit_clause(sql_query)
        state['has_limit_clause'] = has_limit
        
        # 2. Add LIMIT clause if missing and query might return large results
        if not has_limit and self._needs_limit_clause(sql_query):
            modified_query = self._add_limit_clause(modified_query)
            guardrail_applied = True
            logger.info(f"Added LIMIT {self.max_limit} clause for performance")
        
        # 3. Set query timeout
        state['query_timeout'] = self.default_timeout
        
        # 4. Estimate query cost (basic heuristics)
        estimated_cost = self._estimate_query_cost(sql_query)
        state['estimated_cost'] = estimated_cost
        
        # 5. Check for dangerous operations
        dangerous_operations = self._check_dangerous_operations(sql_query)
        if dangerous_operations:
            state['error_message'] = f"Dangerous operations detected: {', '.join(dangerous_operations)}"
            state['next_action'] = 'self_correction'
            state['validation_errors'] = [f"Dangerous operation: {op}" for op in dangerous_operations]
            return state
        
        # Update query if modified
        if guardrail_applied:
            state['sql_query'] = modified_query
        
        # Proceed to execution
        state['next_action'] = 'query_executor'
        
        logger.info(f"Performance guardrails applied. Estimated cost: {estimated_cost}")
        
        return state
    
    def _has_limit_clause(self, sql_query: str) -> bool:
        """Check if query already has a LIMIT clause"""
        return bool(re.search(r'\bLIMIT\s+\d+', sql_query, re.IGNORECASE))
    
    def _needs_limit_clause(self, sql_query: str) -> bool:
        """Determine if query needs a LIMIT clause"""
        sql_upper = sql_query.upper()
        
        # Don't add LIMIT to aggregation queries
        if any(agg in sql_upper for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
            return False
        
        # Don't add LIMIT if already has GROUP BY (likely aggregation)
        if 'GROUP BY' in sql_upper:
            return False
        
        # Don't add LIMIT to UPDATE, INSERT, DELETE
        if any(op in sql_upper for op in ['UPDATE ', 'INSERT ', 'DELETE ']):
            return False
        
        # Add LIMIT to SELECT queries without aggregation
        if sql_upper.strip().startswith('SELECT'):
            return True
        
        return False
    
    def _add_limit_clause(self, sql_query: str) -> str:
        """Add LIMIT clause to query"""
        # Remove trailing semicolon if present
        query = sql_query.rstrip(';').strip()
        
        # Add LIMIT clause
        return f"{query} LIMIT {self.max_limit}"
    
    def _estimate_query_cost(self, sql_query: str) -> float:
        """Estimate query cost using simple heuristics"""
        cost = 1.0
        sql_upper = sql_query.upper()
        
        # JOIN operations increase cost
        join_count = len(re.findall(r'\bJOIN\b', sql_upper))
        cost += join_count * 2.0
        
        # Subqueries increase cost
        subquery_count = sql_query.count('(') - sql_query.count(')')
        if subquery_count > 0:
            cost += subquery_count * 1.5
        
        # LIKE operations increase cost
        like_count = len(re.findall(r'\bLIKE\b', sql_upper))
        cost += like_count * 1.2
        
        # ORDER BY without LIMIT increases cost
        if 'ORDER BY' in sql_upper and not self._has_limit_clause(sql_query):
            cost += 2.0
        
        # Functions increase cost
        function_count = len(re.findall(r'\w+\(', sql_query))
        cost += function_count * 0.5
        
        return round(cost, 2)
    
    def _check_dangerous_operations(self, sql_query: str) -> list:
        """Check for potentially dangerous SQL operations"""
        dangerous_ops = []
        sql_upper = sql_query.upper()
        
        # Data modification operations
        if any(op in sql_upper for op in ['DROP ', 'DELETE ', 'TRUNCATE ', 'ALTER ']):
            dangerous_ops.append("Data modification operations not allowed")
        
        # Infinite loops or resource exhaustion
        if 'WHILE ' in sql_upper:
            dangerous_ops.append("WHILE loops not allowed")
        
        # Recursive CTEs without limits
        if 'WITH RECURSIVE' in sql_upper and not self._has_limit_clause(sql_query):
            dangerous_ops.append("Recursive CTE without LIMIT clause")
        
        # Cross joins without conditions
        if 'CROSS JOIN' in sql_upper:
            dangerous_ops.append("CROSS JOIN operations not allowed")
        
        # Cartesian products (JOIN without ON clause)
        join_pattern = r'\bJOIN\s+\w+\s*(?!ON\b)'
        if re.search(join_pattern, sql_upper):
            dangerous_ops.append("JOIN without ON clause (potential cartesian product)")
        
        return dangerous_ops