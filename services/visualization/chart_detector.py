"""
Chart type detection logic for determining the best visualization type based on data characteristics.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ChartTypeDetector:
    """Detects appropriate chart types based on data characteristics and user intent."""
    
    def __init__(self):
        self.chart_patterns = {
            "distribution": ["histogram", "box_plot", "violin_plot"],
            "comparison": ["bar_chart", "column_chart", "grouped_bar"],
            "trend": ["line_chart", "area_chart", "time_series"],
            "proportion": ["pie_chart", "donut_chart", "treemap"],
            "correlation": ["scatter_plot", "bubble_chart", "heatmap"],
            "geographic": ["map", "choropleth"],
            "hierarchical": ["sunburst", "treemap", "sankey"]
        }
        
        # Keywords that suggest specific chart types
        self.intent_keywords = {
            "trend": ["over time", "trend", "change", "growth", "decline", "timeline", "monthly", "yearly", "daily"],
            "comparison": ["compare", "vs", "versus", "difference", "between", "across", "by category"],
            "distribution": ["distribution", "spread", "range", "histogram", "frequency"],
            "proportion": ["percentage", "proportion", "share", "part of", "breakdown", "composition"],
            "correlation": ["relationship", "correlation", "vs", "against", "related to"],
            "top": ["top", "best", "worst", "highest", "lowest", "most", "least", "ranking"]
        }
    
    def detect_chart_type(self, user_query: str, data: pd.DataFrame, columns_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the most appropriate chart type based on user query and data characteristics.
        
        Args:
            user_query: The original user query
            data: The resulting DataFrame from the query
            columns_info: Information about column types and characteristics
            
        Returns:
            Tuple of (chart_type, chart_config)
        """
        logger.info(f"Detecting chart type for query: '{user_query}' with data shape: {data.shape}")
        
        # Analyze user intent from query
        intent = self._analyze_user_intent(user_query.lower())
        logger.info(f"Detected user intent: {intent}")
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(data, columns_info)
        logger.info(f"Data characteristics: {data_characteristics}")
        
        # Determine best chart type
        chart_type, config = self._select_chart_type(intent, data_characteristics, data)
        
        logger.info(f"Selected chart type: {chart_type}")
        return chart_type, config
    
    def _analyze_user_intent(self, query: str) -> List[str]:
        """Analyze user query to determine visualization intent."""
        detected_intents = []
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query for keyword in keywords):
                detected_intents.append(intent)
        
        # Default to comparison if no specific intent detected
        if not detected_intents:
            detected_intents.append("comparison")
            
        return detected_intents
    
    def _analyze_data_characteristics(self, data: pd.DataFrame, columns_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the characteristics of the resulting data."""
        characteristics = {
            "num_rows": len(data),
            "num_columns": len(data.columns),
            "column_types": {},
            "has_time_column": False,
            "has_categorical": False,
            "has_numerical": False,
            "categorical_columns": [],
            "numerical_columns": [],
            "time_columns": []
        }
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            characteristics["column_types"][col] = dtype
            
            # Check for time-based columns
            if pd.api.types.is_datetime64_any_dtype(data[col]) or 'date' in col.lower() or 'time' in col.lower():
                characteristics["has_time_column"] = True
                characteristics["time_columns"].append(col)
            
            # Check for numerical columns
            elif pd.api.types.is_numeric_dtype(data[col]):
                characteristics["has_numerical"] = True
                characteristics["numerical_columns"].append(col)
            
            # Check for categorical columns
            else:
                characteristics["has_categorical"] = True
                characteristics["categorical_columns"].append(col)
        
        return characteristics
    
    def _select_chart_type(self, intents: List[str], characteristics: Dict[str, Any], data: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Select the most appropriate chart type based on intent and data characteristics."""
        
        num_cols = characteristics["num_columns"]
        num_rows = characteristics["num_rows"]
        has_time = characteristics["has_time_column"]
        has_categorical = characteristics["has_categorical"]
        has_numerical = characteristics["has_numerical"]
        
        config = {
            "x_column": None,
            "y_column": None,
            "color_column": None,
            "title": "Data Visualization",
            "aggregation": None
        }
        
        # Time series data
        if has_time and "trend" in intents:
            config["x_column"] = characteristics["time_columns"][0]
            if characteristics["numerical_columns"]:
                config["y_column"] = characteristics["numerical_columns"][0]
            return "line_chart", config
        
        # Single numerical column - distribution
        if num_cols == 1 and has_numerical:
            config["x_column"] = characteristics["numerical_columns"][0]
            return "histogram", config
        
        # Two columns: categorical + numerical
        if num_cols == 2 and has_categorical and has_numerical:
            config["x_column"] = characteristics["categorical_columns"][0]
            config["y_column"] = characteristics["numerical_columns"][0]
            
            # Check for proportion intent
            if "proportion" in intents and num_rows <= 10:
                return "pie_chart", config
            else:
                return "bar_chart", config
        
        # Multiple numerical columns - correlation
        if len(characteristics["numerical_columns"]) >= 2 and "correlation" in intents:
            config["x_column"] = characteristics["numerical_columns"][0]
            config["y_column"] = characteristics["numerical_columns"][1]
            return "scatter_plot", config
        
        # Three columns: categorical + 2 numerical
        if num_cols == 3 and len(characteristics["categorical_columns"]) == 1 and len(characteristics["numerical_columns"]) == 2:
            config["x_column"] = characteristics["numerical_columns"][0]
            config["y_column"] = characteristics["numerical_columns"][1]
            config["color_column"] = characteristics["categorical_columns"][0]
            return "scatter_plot", config
        
        # Default: bar chart for categorical vs numerical
        if has_categorical and has_numerical:
            config["x_column"] = characteristics["categorical_columns"][0]
            config["y_column"] = characteristics["numerical_columns"][0]
            return "bar_chart", config
        
        # Fallback: table view for complex data
        return "table", config
    
    def get_supported_chart_types(self) -> List[str]:
        """Return list of all supported chart types."""
        all_types = []
        for chart_list in self.chart_patterns.values():
            all_types.extend(chart_list)
        return list(set(all_types))
