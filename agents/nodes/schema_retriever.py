"""
Schema retrieval node for database context
"""

import logging
from typing import Dict, Any
from ..states import AgentState

logger = logging.getLogger(__name__)


class SchemaRetrieverNode:
    """Retrieve and format database schema information"""
    
    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    def __call__(self, state: AgentState) -> AgentState:
        """Retrieve database schema information"""
        
        try:
            if not self.db_connection.is_connected():
                state['error_message'] = "Not connected to database"
                state['next_action'] = 'output_formatter'
                return state
            
            # Get tables
            tables_success, tables_result = self.db_connection.get_tables()
            if not tables_success:
                state['error_message'] = f"Failed to retrieve tables: {tables_result}"
                state['next_action'] = 'output_formatter'
                return state
            
            # Build comprehensive schema context
            schema_parts = ["DATABASE SCHEMA:\n"]
            
            for _, table_row in tables_result.iterrows():
                table_name = table_row['table_name']
                table_type = table_row.get('table_type', 'BASE TABLE')
                
                if table_type == 'BASE TABLE':
                    schema_parts.append(f"\nTABLE: {table_name}")
                    
                    # Get table schema
                    schema_success, schema_result = self.db_connection.get_table_schema(table_name)
                    if schema_success:
                        schema_parts.append("COLUMNS:")
                        
                        for _, col_row in schema_result.iterrows():
                            col_name = col_row['column_name']
                            col_type = col_row['data_type']
                            nullable = col_row.get('is_nullable', 'YES')
                            default_val = col_row.get('column_default', None)
                            
                            col_info = f"  - {col_name}: {col_type}"
                            if nullable == 'NO':
                                col_info += " (NOT NULL)"
                            if default_val:
                                col_info += f" DEFAULT {default_val}"
                            
                            schema_parts.append(col_info)
                    else:
                        schema_parts.append(f"  Error retrieving schema: {schema_result}")
            
            schema_context = "\n".join(schema_parts)
            state['database_schema'] = schema_context
            state['next_action'] = 'context_retriever'
            
            logger.info("Database schema retrieved successfully")
            
        except Exception as e:
            error_msg = f"Schema retrieval failed: {str(e)}"
            logger.error(error_msg)
            state['error_message'] = error_msg
            state['next_action'] = 'output_formatter'
        
        return state