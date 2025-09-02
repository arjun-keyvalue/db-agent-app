# Visualization Module

This module provides natural language to visualization capabilities for the Database Analysis Agent.

## Overview

The visualization module converts natural language queries into interactive charts and graphs using:
- **LangChain** for LLM orchestration and prompt engineering
- **Plotly** for interactive chart generation
- **OpenAI GPT-3.5-turbo** for intelligent SQL generation and chart type selection

## 📊 Sample Visualizations for Library Schema

Here are some possible user queries and their expected chart types:

- **"Show me employee distribution by department"** → Bar Chart
- **"Compare average salaries across different roles"** → Bar Chart
- **"Compare Salary vs performance rating"** → Scatter Plot

## Architecture

```
services/visualization/
├── __init__.py              # Module exports
├── service.py               # Main VisualizationService class
├── chart_detector.py        # Chart type detection logic
├── plotly_generator.py      # Plotly chart generation
└── README.md               # This file
```

## Components

### 1. VisualizationService
Main orchestrator that:
- Generates visualization-optimized SQL queries using LangChain
- Executes queries against the database
- Determines appropriate chart types
- Creates interactive Plotly visualizations

### 2. ChartTypeDetector
Intelligent chart type selection based on:
- **User Intent**: Keywords like "trend", "comparison", "distribution"
- **Data Characteristics**: Column types, row counts, relationships
- **Visualization Best Practices**: Automatic chart type mapping

### 3. PlotlyGenerator
Creates interactive charts with:
- **Dark Theme**: Matches the application's UI
- **Multiple Chart Types**: Bar, line, pie, scatter, histogram, heatmap
- **Responsive Design**: Optimized for web display
- **Interactive Features**: Zoom, pan, hover tooltips

## Supported Chart Types

| Intent | Chart Types | Example Queries |
|--------|-------------|-----------------|
| **Comparison** | Bar Chart, Column Chart | "Compare book counts by genre" |
| **Trend** | Line Chart, Area Chart | "Show loans over time" |
| **Distribution** | Histogram, Box Plot | "Distribution of user ages" |
| **Proportion** | Pie Chart, Donut Chart | "Percentage of books by genre" |
| **Correlation** | Scatter Plot, Bubble Chart | "Rating vs publication year" |
| **Ranking** | Bar Chart (sorted) | "Top 10 most popular books" |

## Usage Examples

### Basic Visualization Queries
```
"Show me book distribution by genre"
"Plot the number of loans per month"
"Compare average ratings by author"
"Visualize user registration trends"
```

### Advanced Queries
```
"Show the relationship between publication year and average rating"
"Display top 10 most borrowed books"
"Create a heatmap of loan patterns by day of week"
"Plot overdue book trends over the last year"
```

## Integration

The module integrates with the existing query engine architecture:

1. **Query Engine Pattern**: Extends the abstract `QueryEngine` class
2. **Factory Integration**: Registered as "visualize" engine type
3. **UI Integration**: Added to strategy dropdown in the web interface
4. **Chat Display**: Visualizations appear inline in the chat interface

## Configuration

Required environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Dependencies

New dependencies added to `requirements.txt`:
```
langchain==0.1.0
langchain-openai==0.0.5
langgraph==0.0.26
seaborn==0.13.0
matplotlib==3.8.2
```

## Error Handling

The module includes comprehensive error handling:
- **Invalid Queries**: Validates visualization intent
- **SQL Errors**: Graceful fallback with error messages
- **Data Issues**: Handles empty results and data type mismatches
- **Chart Generation**: Falls back to table view if chart creation fails

## Testing

Run the test suite:
```bash
python test_visualization.py
```

This will verify:
- Component imports and initialization
- Chart type detection logic
- Plotly chart generation
- Query engine integration

## Future Enhancements

Potential improvements:
1. **More Chart Types**: Gantt charts, network graphs, geographic maps
2. **Advanced Analytics**: Statistical overlays, trend lines, forecasting
3. **Custom Styling**: User-configurable themes and colors
4. **Export Options**: Save charts as PNG, PDF, or HTML
5. **Interactive Filters**: Dynamic chart filtering and drill-down
6. **Caching**: Cache generated visualizations for performance

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install langchain langchain-openai langgraph seaborn matplotlib
   ```

2. **OpenAI API Errors**: Check API key configuration
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. **Empty Visualizations**: Ensure queries return data
   - Check database connection
   - Verify table names and columns exist
   - Try simpler queries first

4. **Chart Display Issues**: 
   - Refresh the browser
   - Check browser console for JavaScript errors
   - Ensure Plotly is properly loaded

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about:
- SQL query generation
- Chart type detection decisions
- Data processing steps
- Error details
