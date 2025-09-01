"""
Main visualization service that orchestrates the conversion of natural language queries 
into data visualizations using LangChain and the database connection.
"""

import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .chart_detector import ChartTypeDetector
from .plotly_generator import PlotlyGenerator

logger = logging.getLogger(__name__)

class VisualizationService:
    """
    Main service for converting natural language queries into visualizations.
    Integrates with the existing database connection and query engine architecture.
    """
    
    def __init__(self, db_connection, openai_api_key: str):
        """
        Initialize the visualization service.
        
        Args:
            db_connection: Database connection instance
            openai_api_key: OpenAI API key for LLM operations
        """
        self.db_connection = db_connection
        self.openai_api_key = openai_api_key
        self.chart_detector = ChartTypeDetector()
        self.plotly_generator = PlotlyGenerator()
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.0
        )
        
        # Create the SQL generation chain
        self.sql_chain = self._create_sql_chain()
        
        logger.info("VisualizationService initialized successfully")
    
    def _create_sql_chain(self) -> LLMChain:
        """Create a LangChain for generating visualization-optimized SQL queries."""
        
        sql_template = """
You are a SQL expert specializing in generating queries for data visualization.
Given a user's natural language request for a visualization, generate an appropriate SQL query.

Database Schema:
{schema_context}

User Request: {user_query}

IMPORTANT GUIDELINES:
1. Generate SQL that returns data suitable for visualization
2. Include appropriate aggregations (COUNT, SUM, AVG, etc.) when needed
3. Use GROUP BY for categorical breakdowns
4. Include ORDER BY for better chart presentation
5. Limit results to reasonable numbers (e.g., TOP 20 for rankings)
6. For time-based queries, ensure proper date formatting
7. Only use tables and columns that exist in the schema above

VISUALIZATION-SPECIFIC RULES:
- For "distribution" queries: Return individual values or frequency counts
- For "comparison" queries: Group by categories with aggregated values
- For "trend" queries: Include time columns and aggregate by time periods
- For "proportion" queries: Return categories with their counts/percentages

Generate ONLY the SQL query, no explanations:
"""
        
        prompt = PromptTemplate(
            input_variables=["schema_context", "user_query"],
            template=sql_template
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def process_visualization_request(self, user_query: str, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Process a visualization request from natural language to final chart.
        
        Args:
            user_query: Natural language query from user
            context: Database and application context
            
        Returns:
            Tuple of (success, result) where result is either a Plotly figure or error message
        """
        logger.info(f"Processing visualization request: '{user_query}'")
        
        try:
            # Step 1: Get database schema context
            schema_context = self._build_schema_context()
            if not schema_context:
                return False, "Unable to access database schema"
            
            # Step 2: Generate visualization-optimized SQL query
            sql_success, sql_query = self._generate_visualization_sql(user_query, schema_context)
            if not sql_success:
                return False, f"Failed to generate SQL: {sql_query}"
            
            logger.info(f"Generated SQL: {sql_query}")
            
            # Step 3: Execute the SQL query
            exec_success, data = self._execute_query(sql_query)
            if not exec_success:
                return False, f"Query execution failed: {data}"
            
            if data.empty:
                return False, "Query returned no data to visualize"
            
            logger.info(f"Query returned {len(data)} rows, {len(data.columns)} columns")
            
            # Step 4: Analyze data and detect appropriate chart type
            columns_info = self._analyze_columns(data)
            chart_type, chart_config = self.chart_detector.detect_chart_type(
                user_query, data, columns_info
            )
            
            # Step 5: Generate the visualization
            chart_config['title'] = self._generate_chart_title(user_query, chart_type)
            figure = self.plotly_generator.generate_chart(chart_type, data, chart_config)
            
            # Step 6: Return success with metadata
            result = {
                'figure': figure,
                'chart_type': chart_type,
                'sql_query': sql_query,
                'data_summary': {
                    'rows': len(data),
                    'columns': len(data.columns),
                    'column_names': list(data.columns)
                },
                'chart_config': chart_config
            }
            
            logger.info(f"Successfully generated {chart_type} visualization")
            return True, result
            
        except Exception as e:
            error_msg = f"Visualization processing failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _build_schema_context(self) -> str:
        """Build schema context for the LLM, similar to SchemaBasedQueryEngine."""
        try:
            if not self.db_connection.is_connected():
                return ""
            
            # Get tables and their schemas
            tables_success, tables_result = self.db_connection.get_tables()
            if not tables_success:
                return ""
            
            schema_context = "AVAILABLE DATABASE SCHEMA FOR VISUALIZATION:\n\n"
            
            for _, table_row in tables_result.iterrows():
                table_name = table_row['table_name']
                table_type = table_row['table_type']
                
                if table_type == 'BASE TABLE':
                    # Get table schema
                    schema_success, schema_result = self.db_connection.get_table_schema(table_name)
                    if schema_success:
                        schema_context += f"TABLE: {table_name}\n"
                        schema_context += "COLUMNS:\n"
                        
                        for _, col_row in schema_result.iterrows():
                            col_name = col_row['column_name']
                            col_type = col_row['data_type']
                            nullable = col_row['is_nullable']
                            
                            schema_context += f"  - {col_name}: {col_type}"
                            if nullable == 'NO':
                                schema_context += " (NOT NULL)"
                            schema_context += "\n"
                        
                        schema_context += "\n"
            
            return schema_context
            
        except Exception as e:
            logger.error(f"Failed to build schema context: {str(e)}")
            return ""
    
    def _generate_visualization_sql(self, user_query: str, schema_context: str) -> Tuple[bool, str]:
        """Generate SQL query optimized for visualization using LangChain."""
        try:
            # Use the LangChain to generate SQL
            result = self.sql_chain.run(
                schema_context=schema_context,
                user_query=user_query
            )
            
            # Clean up the response
            sql_query = result.strip()
            
            # Remove markdown formatting if present
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            sql_query = sql_query.strip()
            
            # Add schema prefix for PostgreSQL if needed
            if self.db_connection.connection_info.get('type') == 'postgresql':
                import re
                sql_query = re.sub(r'\bFROM\s+(\w+)\b', r'FROM public.\1', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\bJOIN\s+(\w+)\b', r'JOIN public.\1', sql_query, flags=re.IGNORECASE)
            
            return True, sql_query
            
        except Exception as e:
            error_msg = f"SQL generation failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _execute_query(self, sql_query: str) -> Tuple[bool, pd.DataFrame]:
        """Execute the SQL query and return results."""
        try:
            success, result = self.db_connection.execute_query(sql_query)
            if success:
                return True, result
            else:
                return False, result
        except Exception as e:
            return False, str(e)
    
    def _analyze_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column characteristics for chart type detection."""
        columns_info = {}
        
        for col in data.columns:
            col_info = {
                'dtype': str(data[col].dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(data[col]),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(data[col]),
                'unique_values': data[col].nunique(),
                'has_nulls': data[col].isnull().any(),
                'sample_values': data[col].dropna().head(3).tolist()
            }
            columns_info[col] = col_info
        
        return columns_info
    
    def _generate_chart_title(self, user_query: str, chart_type: str) -> str:
        """Generate an appropriate title for the chart based on the user query."""
        # Simple title generation - could be enhanced with LLM
        query_words = user_query.lower().split()
        
        # Remove common words
        stop_words = {'show', 'me', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [word for word in query_words if word not in stop_words]
        
        if meaningful_words:
            title = ' '.join(meaningful_words[:5]).title()
        else:
            title = f"{chart_type.replace('_', ' ').title()}"
        
        return title
    
    def get_supported_visualizations(self) -> List[str]:
        """Return list of supported visualization types."""
        return self.chart_detector.get_supported_chart_types()
    
    def validate_visualization_request(self, user_query: str) -> Tuple[bool, str]:
        """
        Validate if a user query is suitable for visualization.
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Simple validation - could be enhanced
        visualization_keywords = [
            'show', 'display', 'plot', 'chart', 'graph', 'visualize', 'distribution',
            'trend', 'compare', 'comparison', 'over time', 'breakdown', 'proportion'
        ]
        
        query_lower = user_query.lower()
        has_viz_intent = any(keyword in query_lower for keyword in visualization_keywords)
        
        if has_viz_intent:
            return True, "Query appears suitable for visualization"
        else:
            return False, "Query doesn't seem to request a visualization. Try using words like 'show', 'plot', 'chart', or 'visualize'."
