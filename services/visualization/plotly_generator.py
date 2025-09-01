"""
Plotly chart generation module for creating interactive visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PlotlyGenerator:
    """Generates Plotly charts based on chart type and configuration."""
    
    def __init__(self):
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Default styling for dark theme (matching the app)
        self.default_layout = {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#ffffff', 'family': 'Arial, sans-serif'},
            'title': {'font': {'size': 18, 'color': '#ffffff'}},
            'xaxis': {
                'gridcolor': '#404040',
                'linecolor': '#404040',
                'tickcolor': '#404040',
                'title': {'font': {'color': '#ffffff'}}
            },
            'yaxis': {
                'gridcolor': '#404040',
                'linecolor': '#404040',
                'tickcolor': '#404040',
                'title': {'font': {'color': '#ffffff'}}
            },
            'legend': {
                'font': {'color': '#ffffff'},
                'bgcolor': 'rgba(0,0,0,0.5)'
            }
        }
    
    def generate_chart(self, chart_type: str, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """
        Generate a Plotly chart based on the specified type and configuration.
        
        Args:
            chart_type: Type of chart to generate
            data: DataFrame containing the data
            config: Chart configuration dictionary
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Generating {chart_type} chart with config: {config}")
        
        try:
            # Generate chart based on type
            if chart_type == "bar_chart":
                return self._create_bar_chart(data, config)
            elif chart_type == "line_chart":
                return self._create_line_chart(data, config)
            elif chart_type == "pie_chart":
                return self._create_pie_chart(data, config)
            elif chart_type == "scatter_plot":
                return self._create_scatter_plot(data, config)
            elif chart_type == "histogram":
                return self._create_histogram(data, config)
            elif chart_type == "box_plot":
                return self._create_box_plot(data, config)
            elif chart_type == "heatmap":
                return self._create_heatmap(data, config)
            elif chart_type == "area_chart":
                return self._create_area_chart(data, config)
            else:
                # Fallback to table view
                return self._create_table(data, config)
                
        except Exception as e:
            logger.error(f"Error generating {chart_type}: {str(e)}")
            # Return a simple table as fallback
            return self._create_table(data, config)
    
    def _create_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a bar chart."""
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        
        if color_col and color_col in data.columns:
            fig = px.bar(data, x=x_col, y=y_col, color=color_col,
                        title=config.get('title', 'Bar Chart'),
                        color_discrete_sequence=self.color_palette)
        else:
            fig = px.bar(data, x=x_col, y=y_col,
                        title=config.get('title', 'Bar Chart'),
                        color_discrete_sequence=self.color_palette)
        
        fig.update_layout(**self.default_layout)
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a line chart."""
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        
        if color_col and color_col in data.columns:
            fig = px.line(data, x=x_col, y=y_col, color=color_col,
                         title=config.get('title', 'Line Chart'),
                         color_discrete_sequence=self.color_palette)
        else:
            fig = px.line(data, x=x_col, y=y_col,
                         title=config.get('title', 'Line Chart'),
                         color_discrete_sequence=self.color_palette)
        
        fig.update_layout(**self.default_layout)
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a pie chart."""
        x_col = config.get('x_column')  # categories
        y_col = config.get('y_column')  # values
        
        fig = px.pie(data, names=x_col, values=y_col,
                    title=config.get('title', 'Pie Chart'),
                    color_discrete_sequence=self.color_palette)
        
        fig.update_layout(**self.default_layout)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    def _create_scatter_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a scatter plot."""
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        
        if color_col and color_col in data.columns:
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                           title=config.get('title', 'Scatter Plot'),
                           color_discrete_sequence=self.color_palette)
        else:
            fig = px.scatter(data, x=x_col, y=y_col,
                           title=config.get('title', 'Scatter Plot'),
                           color_discrete_sequence=self.color_palette)
        
        fig.update_layout(**self.default_layout)
        return fig
    
    def _create_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a histogram."""
        x_col = config.get('x_column')
        
        fig = px.histogram(data, x=x_col,
                          title=config.get('title', 'Distribution'),
                          color_discrete_sequence=self.color_palette)
        
        fig.update_layout(**self.default_layout)
        return fig
    
    def _create_box_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a box plot."""
        y_col = config.get('y_column')
        x_col = config.get('x_column')
        
        if x_col and x_col in data.columns:
            fig = px.box(data, x=x_col, y=y_col,
                        title=config.get('title', 'Box Plot'),
                        color_discrete_sequence=self.color_palette)
        else:
            fig = px.box(data, y=y_col,
                        title=config.get('title', 'Box Plot'),
                        color_discrete_sequence=self.color_palette)
        
        fig.update_layout(**self.default_layout)
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a heatmap."""
        # For correlation heatmap
        if len(data.select_dtypes(include=['number']).columns) > 1:
            corr_matrix = data.select_dtypes(include=['number']).corr()
            fig = px.imshow(corr_matrix,
                           title=config.get('title', 'Correlation Heatmap'),
                           color_continuous_scale='RdBu_r')
        else:
            # Fallback to table
            return self._create_table(data, config)
        
        fig.update_layout(**self.default_layout)
        return fig
    
    def _create_area_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create an area chart."""
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        
        if color_col and color_col in data.columns:
            fig = px.area(data, x=x_col, y=y_col, color=color_col,
                         title=config.get('title', 'Area Chart'),
                         color_discrete_sequence=self.color_palette)
        else:
            fig = px.area(data, x=x_col, y=y_col,
                         title=config.get('title', 'Area Chart'),
                         color_discrete_sequence=self.color_palette)
        
        fig.update_layout(**self.default_layout)
        return fig
    
    def _create_table(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a table view as fallback."""
        # Limit to first 100 rows for performance
        display_data = data.head(100)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_data.columns),
                fill_color='#404040',
                font=dict(color='white', size=12),
                align="left"
            ),
            cells=dict(
                values=[display_data[col] for col in display_data.columns],
                fill_color='#2c2c2c',
                font=dict(color='white', size=11),
                align="left"
            )
        )])
        
        fig.update_layout(
            title=config.get('title', 'Data Table'),
            **self.default_layout
        )
        
        return fig
    
    def customize_chart(self, fig: go.Figure, customizations: Dict[str, Any]) -> go.Figure:
        """Apply custom styling to a chart."""
        if 'title' in customizations:
            fig.update_layout(title=customizations['title'])
        
        if 'xlabel' in customizations:
            fig.update_xaxes(title_text=customizations['xlabel'])
        
        if 'ylabel' in customizations:
            fig.update_yaxes(title_text=customizations['ylabel'])
        
        if 'colors' in customizations:
            # Update color scheme if provided
            pass
        
        return fig
