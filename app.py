import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv
from database.connection import db_connection
from database.query_engine import query_engine_factory
from config import Config
import uuid
import plotly.graph_objects as go
import os

# Add the hardcoded SQLite3 database path
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'sqllite3', 'library.db')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Database Analysis Agent"
)

# Custom CSS for ChatGPT-like styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #343a40 !important;
                color: #ffffff !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            }
            .chat-container {
                background-color: #2c2c2c;
                border-radius: 8px;
                border: 1px solid #404040;
            }
            .chat-message {
                padding: 16px 20px;
                border-bottom: 1px solid #404040;
                animation: fadeIn 0.3s ease-in;
            }
            .chat-message:last-child {
                border-bottom: none;
            }
            .user-message {
                background-color: #343a40;
            }
            .agent-message {
                background-color: #2c2c2c;
            }
            .chat-input-container {
                background-color: #2c2c2c;
                border-top: 1px solid #404040;
                padding: 20px;
                position: sticky;
                bottom: 0;
            }
            .chat-input {
                background-color: #404040 !important;
                border: 1px solid #555 !important;
                color: #ffffff !important;
                border-radius: 20px !important;
                padding: 12px 20px !important;
            }
            .chat-input:focus {
                border-color: #007bff !important;
                box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
            }
            .send-button {
                border-radius: 20px !important;
                padding: 12px 24px !important;
                background-color: #007bff !important;
                border: none !important;
                white-space: nowrap !important;
                min-width: 80px !important;
            }
            .send-button:hover {
                background-color: #0056b3 !important;
            }
            .db-connect-button {
                background-color: #28a745 !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 8px 16px !important;
                margin-bottom: 20px !important;
                font-size: 14px !important;
                max-width: 200px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            .db-connect-button:hover {
                background-color: #218838 !important;
            }
            .modal-content {
                background-color: #2c2c2c !important;
                color: #ffffff !important;
                border: 1px solid #404040 !important;
            }
            .modal-header {
                border-bottom: 1px solid #404040 !important;
                background-color: #343a40 !important;
            }
            .modal-body {
                background-color: #2c2c2c !important;
            }
            .modal-footer {
                border-top: 1px solid #404040 !important;
                background-color: #343a40 !important;
            }
            .form-control {
                background-color: #404040 !important;
                border: 1px solid #555 !important;
                color: #ffffff !important;
            }
            .form-control:focus {
                background-color: #404040 !important;
                border-color: #007bff !important;
                color: #ffffff !important;
                box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
            }
            .form-select {
                background-color: #404040 !important;
                border: 1px solid #555 !important;
                color: #ffffff !important;
            }
            .form-select:focus {
                background-color: #404040 !important;
                border-color: #007bff !important;
                color: #ffffff !important;
                box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .status-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-connected {
                background-color: #28a745;
            }
            .status-disconnected {
                background-color: #dc3545;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Database Analysis Agent", className="text-center mb-3", style={"color": "#ffffff"}),
            html.Hr(style={"borderColor": "#404040"})
        ])
    ]),
    
    # Database Connection Button
    dbc.Row([
        dbc.Col([
            dbc.Button([
                html.I(className="fas fa-database me-2"),
                "Connect to Database"
            ], id="open-db-modal", className="db-connect-button w-100")
        ], md=6, className="mx-auto")
    ]),
    
    # Connection Status
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span(className="status-indicator status-disconnected", id="status-indicator"),
                html.Span("Not connected to database", id="connection-status-text"),
                html.Br(),
                html.Div(id="connection-actions", style={"marginTop": "10px"})
            ], className="text-center")
        ])
    ]),
    
    # Main Chat Interface
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # Chat messages display
                    html.Div(id="chat-messages", className="chat-container", style={
                        "height": "600px",
                        "overflowY": "auto",
                        "marginBottom": "0"
                    }),
                    
                    # Chat input area
                    html.Div([
                        # Query Strategy Controls
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Span("Strategy: ", className="text-light me-2"),
                                    dbc.Select(
                                        id="query-strategy",
                                        options=[
                                            {"label": "Schema-Based Querying", "value": "schema"},
                                            {"label": "RAG (Retrieval-Augmented Generation)", "value": "rag"},
                                            {"label": "Visualize", "value": "visualize"},
                                            {"label": "Multi-Table Join", "value": "multitablejoin"}
                                        ],
                                        value="schema",
                                        size="sm",
                                        style={"display": "inline-block", "width": "auto", "minWidth": "200px", "marginTop": "5px"}
                                    )
                                ], md=8, className="d-flex align-items-center justify-content-center"),
                                dbc.Col([
                                    dbc.Checkbox(
                                        id="security-guardrail",
                                        label="Security Guardrails",
                                        value=True,
                                        className="text-light"
                                    )
                                ], md=4, className="d-flex align-items-center justify-content-center", style={"marginTop": "10px"})
                            ], className="mb-1", style={"alignItems": "center"})
                        ], className="mb-2", style={"backgroundColor": "#404040", "borderRadius": "5px", "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                        
                        # Chat input
                        dbc.InputGroup([
                            dbc.Input(
                                id="chat-input",
                                placeholder="Ask me about your database...",
                                type="text",
                                className="chat-input",
                                style={"flex": "1"}
                            ),
                            dbc.Button([
                                html.I(className="fas fa-paper-plane me-2"),
                                "Send"
                            ], id="send-button", className="send-button ms-2")
                        ])
                    ], className="chat-input-container")
                ])
            ], className="border-0", style={"backgroundColor": "transparent"})
        ], md=10, className="mx-auto")
    ]),
    
    # Database Connection Modal
    dbc.Modal([
        dbc.ModalHeader([
            html.H4("Database Connection", className="mb-0")
        ]),
        dbc.ModalBody([
            dbc.Form([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Database Type", className="mb-2"),
                        dbc.Select(
                            id="db-type",
                            options=[
                                {"label": "PostgreSQL", "value": "postgresql"},
                                {"label": "MySQL", "value": "mysql"},
                                {"label": "SQLite3", "value": "sqlite3"}
                            ],
                            value="postgresql"
                        )
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Host", className="mb-2"),
                        dbc.Input(id="db-host", placeholder="localhost", value="localhost")
                    ], md=6)
                ], className="mb-3"),
                
                # Add SQLite file selector
                # html.Div(id="sqlite-file-selector", style={"display": "none"}, children=[
                #     dbc.Label("SQLite Database File", className="mb-2"),
                #     dbc.Input(id="sqlite-file", type="text", placeholder="Path to SQLite database file")
                # ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Port", className="mb-2"),
                        dbc.Input(id="db-port", placeholder="5432", value="5432", type="number")
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Database Name", className="mb-2"),
                        dbc.Input(id="db-name", placeholder="your_database")
                    ], md=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Username", className="mb-2"),
                        dbc.Input(id="db-username", placeholder="username")
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Password", className="mb-2"),
                        dbc.Input(id="db-password", placeholder="password", type="password")
                    ], md=6)
                ], className="mb-3"),
                
                html.Div(id="modal-connection-status")
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="close-db-modal", color="secondary", outline=True),
            dbc.Button("Connect", id="connect-button", color="success")
        ])
    ], id="db-modal", size="lg", is_open=False),
    
    # Store for connection state
    dcc.Store(id="connection-store", data={"connected": False}),
    
    # Store for chat history
    dcc.Store(id="chat-store", data=[]),
    
    # Store for query strategy and security settings
    dcc.Store(id="settings-store", data={"strategy": "schema", "security": True})
    
], fluid=True, style={"backgroundColor": "#343a40", "minHeight": "100vh", "padding": "20px"})

# Callback to open/close database modal
@app.callback(
    Output("db-modal", "is_open"),
    [Input("open-db-modal", "n_clicks"),
     Input("close-db-modal", "n_clicks"),
     Input("connect-button", "n_clicks")],
    [State("db-modal", "is_open")]
)
def toggle_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open

# Callback for chat functionality
@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-input", "value"),
     Output("chat-store", "data")],
    [Input("send-button", "n_clicks"),
     Input("chat-input", "n_submit")],
    [State("chat-input", "value"),
     State("chat-store", "data"),
     State("settings-store", "data"),
     State("connection-store", "data"),
     State("db-modal", "is_open")]
)
def update_chat(n_clicks, n_submit, input_value, chat_history, settings, connection, modal_is_open):
    # Prevent chat processing when database modal is open
    if modal_is_open:
        return chat_history or [], "", chat_history or []
    
    if not input_value:
        return chat_history or [], "", chat_history or []
    
    # Get the trigger
    ctx = callback_context
    if not ctx.triggered:
        return chat_history or [], "", chat_history or []
    
    # Initialize chat history if empty
    if not chat_history:
        chat_history = []
    
    # Add user message
    user_message = {
        "type": "user",
        "text": input_value,
        "timestamp": datetime.now().strftime("%H:%M")
    }
    chat_history.append(user_message)
    
    # Check if connected to database
    if not connection.get("connected", False):
        agent_response = "❌ Please connect to a database first before asking questions."
    else:
        # Get current settings
        strategy = settings.get("strategy", "schema")
        security = settings.get("security", True)
        
        # Initialize variables for SQL query and results
        sql_query = None
        results = None
        
        try:
            # Create query engine
            engine_config = {"openai_api_key": Config.OPENAI_API_KEY, "db_uri": SQLITE_DB_PATH}
            query_engine = query_engine_factory.create_query_engine(strategy, engine_config)
            
            # Create security guardrail if enabled
            security_guardrail = None
            if security:
                security_guardrail = query_engine_factory.create_security_guardrail("basic", {})
            
            # Get database context
            db_context = {
                "db_type": connection.get("info", {}).get("type", "postgresql"),
                "database": connection.get("info", {}).get("database", "unknown"),
                "user_input": input_value,
                "security_enabled": security,
            }
            
            # Generate SQL query
            sql_success, sql_result = query_engine.generate_query(input_value, db_context)
            
            if sql_success:
                # Apply security validation if enabled
                if security_guardrail:
                    security_success, security_message = security_guardrail.validate_query(sql_result, db_context)
                    if not security_success:
                        agent_response = f"🚫 Security Check Failed: {security_message}\n\nQuery blocked for security reasons."
                    else:
                        # Execute the query
                        exec_success, exec_result = query_engine.execute_query(sql_result)
                        if exec_success:
                            # Handle visualization results differently
                            if strategy == "visualize" and isinstance(exec_result, dict) and 'figure' in exec_result:
                                # Visualization result
                                agent_response = f"Generated visualization: {exec_result.get('chart_type', 'chart').replace('_', ' ').title()}"
                                sql_query = sql_result
                                results = dcc.Graph(
                                    figure=exec_result['figure'],
                                    style={'height': '500px'},
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                                # Store visualization metadata
                                results_data = {
                                    'type': 'visualization',
                                    'chart_type': exec_result.get('chart_type'),
                                    'data_summary': exec_result.get('data_summary'),
                                    'chart_config': exec_result.get('chart_config')
                                }
                            # Format regular tabular results
                            elif isinstance(exec_result, pd.DataFrame):
                                result_html = dbc.Table.from_dataframe(
                                    exec_result, 
                                    striped=True, 
                                    bordered=True, 
                                    hover=True,
                                    className="mt-3"
                                )
                                agent_response = "Response:"
                                sql_query = sql_result
                                results = result_html
                                # Serialize non-JSON-serializable columns before storing
                                exec_result_serialized = exec_result.copy()
                                for col in exec_result_serialized.columns:
                                    # Convert UUID objects to strings
                                    if exec_result_serialized[col].apply(lambda x: isinstance(x, uuid.UUID)).any():
                                        exec_result_serialized[col] = exec_result_serialized[col].astype(str)
                                    # Convert Timestamp objects to strings
                                    if pd.api.types.is_datetime64_any_dtype(exec_result_serialized[col]):
                                        exec_result_serialized[col] = exec_result_serialized[col].astype(str)
                                    # Convert generic datetime objects to strings
                                    if exec_result_serialized[col].apply(lambda x: isinstance(x, datetime)).any():
                                        exec_result_serialized[col] = exec_result_serialized[col].astype(str)
                                    
                                    # Handle None/NaN values that cause React rendering issues
                                    exec_result_serialized[col] = exec_result_serialized[col].fillna('').astype(str)
                                    # Replace any remaining None values with empty string
                                    exec_result_serialized[col] = exec_result_serialized[col].replace('None', '').replace('nan', '')
                                
                                results_data = exec_result_serialized.to_dict('records')  # Store as serializable data
                            else:
                                agent_response = f"Response:\n{exec_result}"
                                sql_query = sql_result
                                results = None
                                results_data = str(exec_result)
                        else:
                            agent_response = f"❌ Execution Failed:\n{exec_result}"
                            sql_query = sql_result
                            results = None
                            results_data = str(exec_result)
                else:
                    # Security disabled - execute directly
                    exec_success, exec_result = query_engine.execute_query(sql_result)
                    if exec_success:
                        # Handle visualization results differently
                        if strategy == "visualize" and isinstance(exec_result, dict) and 'figure' in exec_result:
                            # Visualization result
                            agent_response = f"Generated visualization: {exec_result.get('chart_type', 'chart').replace('_', ' ').title()}"
                            sql_query = sql_result
                            results = dcc.Graph(
                                figure=exec_result['figure'],
                                style={'height': '500px'},
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                            # Store visualization metadata
                            results_data = {
                                'type': 'visualization',
                                'chart_type': exec_result.get('chart_type'),
                                'data_summary': exec_result.get('data_summary'),
                                'chart_config': exec_result.get('chart_config')
                            }
                        elif isinstance(exec_result, pd.DataFrame):
                            result_html = dbc.Table.from_dataframe(
                                exec_result, 
                                striped=True, 
                                bordered=True, 
                                hover=True,
                                className="mt-3"
                            )
                            agent_response = "Response:"
                            sql_query = sql_result
                            results = result_html
                            # Serialize non-JSON-serialized columns before storing
                            exec_result_serialized = exec_result.copy()
                            for col in exec_result_serialized.columns:
                                # Convert UUID objects to strings
                                if exec_result_serialized[col].apply(lambda x: isinstance(x, uuid.UUID)).any():
                                    exec_result_serialized[col] = exec_result_serialized[col].astype(str)
                                # Convert Timestamp objects to strings
                                if pd.api.types.is_datetime64_any_dtype(exec_result_serialized[col]):
                                    exec_result_serialized[col] = exec_result_serialized[col].astype(str)
                                # Convert generic datetime objects to strings
                                if exec_result_serialized[col].apply(lambda x: isinstance(x, datetime)).any():
                                    exec_result_serialized[col] = exec_result_serialized[col].astype(str)
                                
                                # Handle None/NaN values that cause React rendering issues
                                exec_result_serialized[col] = exec_result_serialized[col].fillna('').astype(str)
                                # Replace any remaining None values with empty string
                                exec_result_serialized[col] = exec_result_serialized[col].replace('None', '').replace('nan', '')
                            
                            results_data = exec_result_serialized.to_dict('records')  # Store as serializable data
                        else:
                            agent_response = f"🔍 Generated SQL Query:\n```sql\n{sql_result}\n```\n\n📊 Query Results:\n{exec_result}"
                            sql_query = sql_result
                            results = None
                            results_data = str(exec_result)
                    else:
                        agent_response = f"🔍 Generated SQL Query:\n```sql\n{sql_result}\n```\n\n❌ Execution Failed:\n{exec_result}"
                        sql_query = sql_result
                        results = None
                        results_data = str(exec_result)
            else:
                agent_response = f"❌ Failed to generate SQL query: {sql_result}"
                sql_query = None
                results = None
                results_data = None
                
        except Exception as e:
            agent_response = f"❌ Error processing query: {str(e)}"
            logger.error(f"Query processing error: {str(e)}")
            sql_query = None
            results = None
            results_data = None
    
    agent_message = {
        "type": "agent",
        "text": agent_response,
        "timestamp": datetime.now().strftime("%H:%M")
    }
    
    # Add SQL query and results if available
    if sql_query:
        agent_message["sql_query"] = sql_query
    # Don't store HTML components in the message
    if 'results_data' in locals() and results_data is not None:
        agent_message["results_data"] = results_data
        agent_message["has_results"] = True  # Set the flag for rendering
    # Store live visualization component for current message
    if 'results' in locals() and isinstance(results, dcc.Graph):
        agent_message["live_visualization"] = results
    
    chat_history.append(agent_message)
    
    # Convert to HTML
    chat_html = []
    
    for msg in chat_history:
        if msg["type"] == "user":
            chat_html.append(html.Div([
                html.Div([
                    html.Strong("You", className="text-primary"),
                    html.Span(f" • {msg['timestamp']}", className="text-muted ms-2")
                ], className="mb-1"),
                html.Div(msg["text"], className="text-light")
            ], className="chat-message user-message"))
        else:
            # Create agent message content
            message_content = [
                html.Div([
                    html.Strong("Agent", className="text-success"),
                    html.Span(f" • {msg['timestamp']}", className="text-muted ms-2")
                ], className="mb-1"),
                html.Div(msg["text"], className="text-light")
            ]
            
            # Add SQL query if present
            if "sql_query" in msg:
                message_content.append(html.Div([
                    html.Hr(style={"borderColor": "#404040", "margin": "10px 0"}),
                    html.Div([
                        html.Strong("Generated SQL:", className="text-info"),
                        html.Pre(msg["sql_query"], className="bg-dark text-light p-2 rounded", style={"fontSize": "12px", "overflowX": "auto"})
                    ])
                ]))
            
            # Add live visualization if present (for current message)
            if "live_visualization" in msg:
                message_content.append(html.Div([
                    html.Hr(style={"borderColor": "#404040", "margin": "10px 0"}),
                    html.Div([
                        html.Strong("Visualization:", className="text-info"),
                        msg["live_visualization"]
                    ])
                ]))
            
            # Add results if present
            elif "has_results" in msg and msg.get("results_data"):
                # Handle visualization results
                if isinstance(msg["results_data"], dict) and msg["results_data"].get("type") == "visualization":
                    viz_data = msg["results_data"]
                    message_content.append(html.Div([
                        html.Hr(style={"borderColor": "#404040", "margin": "10px 0"}),
                        html.Div([
                            html.Strong("Visualization:", className="text-info"),
                            html.P(f"Chart Type: {viz_data.get('chart_type', 'Unknown').replace('_', ' ').title()}", 
                                  className="text-light mt-2 mb-2"),
                            html.P(f"Data: {viz_data.get('data_summary', {}).get('rows', 0)} rows, "
                                  f"{viz_data.get('data_summary', {}).get('columns', 0)} columns", 
                                  className="text-muted small")
                        ])
                    ]))
                # Regenerate the results table from stored data
                elif isinstance(msg["results_data"], list) and len(msg["results_data"]) > 0:
                    # Convert back to DataFrame and ensure it's serializable
                    df = pd.DataFrame(msg["results_data"])
                    
                    # Ensure all columns are serializable
                    for col in df.columns:
                        # Convert UUID objects to strings
                        if df[col].apply(lambda x: isinstance(x, uuid.UUID)).any():
                            df[col] = df[col].astype(str)
                        # Convert Timestamp objects to strings
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            df[col] = df[col].astype(str)
                        # Convert generic datetime objects to strings
                        if df[col].apply(lambda x: isinstance(x, datetime)).any():
                            df[col] = df[col].astype(str)
                        
                        # Handle None/NaN values that cause React rendering issues
                        df[col] = df[col].fillna('').astype(str)
                        # Replace any remaining None values with empty string
                        df[col] = df[col].replace('None', '').replace('nan', '')
                    
                    # Create the table
                    results_table = dbc.Table.from_dataframe(
                        df, 
                        striped=True, 
                        bordered=True, 
                        hover=True,
                        className="mt-3"
                    )
                    message_content.append(html.Div([
                        html.Hr(style={"borderColor": "#404040", "margin": "10px 0"}),
                        html.Div([
                            html.Strong("Query Results:", className="text-info"),
                            results_table
                        ])
                    ]))
                else:
                    # Handle string results
                    message_content.append(html.Div([
                        html.Hr(style={"borderColor": "#404040", "margin": "10px 0"}),
                        html.Div([
                            html.Strong("Query Results:", className="text-info"),
                            html.Pre(str(msg["results_data"]), className="text-light")
                        ])
                    ]))
            
            chat_html.append(html.Div(message_content, className="chat-message agent-message"))
    
    # Store only serializable data in chat_history, not HTML components
    serializable_chat_history = []
    for msg in chat_history:
        serializable_msg = {
            "type": msg["type"],
            "text": msg["text"],
            "timestamp": msg["timestamp"]
        }
        if "sql_query" in msg:
            serializable_msg["sql_query"] = msg["sql_query"]
        if "results_data" in msg:
            # Store results as a flag, not the actual HTML component
            serializable_msg["has_results"] = True
            serializable_msg["results_data"] = msg.get("results_data", None)
        serializable_chat_history.append(serializable_msg)
    
    # Ensure chat_html is properly formatted for Dash
    # The first output should be HTML components, which Dash can handle
    return chat_html, "", serializable_chat_history

# Combined callback for database connection management
@app.callback(
    [Output("modal-connection-status", "children"),
     Output("connection-store", "data"),
     Output("status-indicator", "className"),
     Output("connection-status-text", "children"),
     Output("connection-actions", "children")],
    [Input("connect-button", "n_clicks")],
    [State("db-type", "value"),
     State("db-host", "value"),
     State("db-port", "value"),
     State("db-name", "value"),
     State("db-username", "value"),
     State("db-password", "value")],
    prevent_initial_call=True
)
def manage_database_connection(connect_clicks, db_type, host, port, db_name, username, password):
    if not connect_clicks:
        return "", {"connected": False}, "status-indicator status-disconnected", "Not connected to database", ""
    
    try:
        if db_type == "sqlite3":
            # Use the hardcoded SQLite database path
            success, message = db_connection.connect(db_type, "", "", SQLITE_DB_PATH, "", "")
        else:
            # For other databases, check all required fields
            if not all([host, port, db_name, username, password]):
                return (
                    dbc.Alert("Please fill in all database connection fields.", color="warning"),
                    {"connected": False},
                    "status-indicator status-disconnected",
                    "Not connected to database",
                    ""
                )
            success, message = db_connection.connect(db_type, host, port, db_name, username, password)
        
        if success:
            # Get connection info for display
            conn_info = db_connection.get_connection_info()
            status_text = f"Connected to {conn_info['type']} database: {conn_info['database']}"
            
            # Create connection actions
            actions = html.Div([
                dbc.Button("View Tables", id="view-tables-btn", color="info", size="sm", className="me-2"),
                dbc.Button("Disconnect", id="disconnect-btn", color="danger", size="sm", outline=True)
            ])
            
            return (
                dbc.Alert(f"✅ {message}", color="success"),
                {"connected": True, "info": conn_info},
                "status-indicator status-connected",
                status_text,
                actions
            )
        else:
            return (
                dbc.Alert(f"❌ {message}", color="danger"),
                {"connected": False},
                "status-indicator status-disconnected",
                "Connection failed",
                ""
            )
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return (
            dbc.Alert(f"❌ {error_msg}", color="danger"),
            {"connected": False},
            "status-indicator status-disconnected",
            "Connection error",
            ""
        )

# Callback for view tables button
@app.callback(
    Output("chat-messages", "children", allow_duplicate=True),
    [Input("view-tables-btn", "n_clicks")],
    [State("chat-store", "data")],
    prevent_initial_call=True
)
def view_database_tables(n_clicks, chat_history):
    if not n_clicks:
        return dash.no_update
    
    if not db_connection.is_connected():
        return chat_history or []
    
    try:
        success, result = db_connection.get_tables()
        
        if success:
            # Convert tables to HTML table
            tables_df = result
            tables_html = dbc.Table.from_dataframe(
                tables_df, 
                striped=True, 
                bordered=True, 
                hover=True,
                className="mt-3"
            )
            
            # Add agent message about tables
            if not chat_history:
                chat_history = []
            
            agent_message = {
                "type": "agent",
                "text": f"Here are the tables in your database:",
                "timestamp": datetime.now().strftime("%H:%M"),
                "tables": tables_html
            }
            chat_history.append(agent_message)
            
            # Convert to HTML
            chat_html = []
            for msg in chat_history:
                if msg["type"] == "user":
                    chat_html.append(html.Div([
                        html.Div([
                            html.Strong("You", className="text-primary"),
                            html.Span(f" • {msg['timestamp']}", className="text-muted ms-2")
                        ], className="mb-1"),
                        html.Div(msg["text"], className="text-light")
                    ], className="chat-message user-message"))
                else:
                    chat_html.append(html.Div([
                        html.Div([
                            html.Strong("Agent", className="text-success"),
                            html.Span(f" • {msg['timestamp']}", className="text-muted ms-2")
                        ], className="mb-1"),
                        html.Div(msg["text"], className="text-light"),
                        html.Div(msg.get("tables", ""), className="mt-2") if "tables" in msg else ""
                    ], className="chat-message agent-message"))
            
            return chat_html
        else:
            # Add error message
            if not chat_history:
                chat_history = []
            
            agent_message = {
                "type": "agent",
                "text": f"Failed to get tables: {result}",
                "timestamp": datetime.now().strftime("%H:%M")
            }
            chat_history.append(agent_message)
            
            # Convert to HTML
            chat_html = []
            for msg in chat_history:
                if msg["type"] == "user":
                    chat_html.append(html.Div([
                        html.Div([
                            html.Strong("You", className="text-primary"),
                            html.Span(f" • {msg['timestamp']}", className="text-muted ms-2")
                        ], className="mb-1"),
                        html.Div(msg["text"], className="text-light")
                    ], className="chat-message user-message"))
                else:
                    chat_html.append(html.Div([
                        html.Div([
                            html.Strong("Agent", className="text-success"),
                            html.Span(f" • {msg['timestamp']}", className="text-muted ms-2")
                        ], className="mb-1"),
                        html.Div(msg["text"], className="text-light")
                    ], className="chat-message agent-message"))
            
            return chat_html
            
    except Exception as e:
        # Add error message
        if not chat_history:
            chat_history = []
        
        agent_message = {
            "type": "agent",
            "text": f"Error getting tables: {str(e)}",
            "timestamp": datetime.now().strftime("%H:%M")
        }
        chat_history.append(agent_message)
        
        # Convert to HTML
        chat_html = []
        for msg in chat_history:
            if msg["type"] == "user":
                chat_html.append(html.Div([
                    html.Div([
                        html.Strong("You", className="text-primary"),
                        html.Span(f" • {msg['timestamp']}", className="text-muted ms-2")
                    ], className="mb-1"),
                    html.Div(msg["text"], className="text-light")
                ], className="chat-message user-message"))
            else:
                chat_html.append(html.Div([
                    html.Div([
                        html.Strong("Agent", className="text-success"),
                        html.Span(f" • {msg['timestamp']}", className="text-muted ms-2")
                    ], className="mb-1"),
                    html.Div(msg["text"], className="text-light")
                ], className="chat-message agent-message"))
        
        return chat_html

# Callback to update settings store
@app.callback(
    Output("settings-store", "data"),
    [Input("query-strategy", "value"),
     Input("security-guardrail", "value")]
)
def update_settings(strategy, security):
    return {"strategy": strategy, "security": security}

# Callback for disconnect button (only active when button exists)
@app.callback(
    [Output("connection-store", "data", allow_duplicate=True),
     Output("status-indicator", "className", allow_duplicate=True),
     Output("connection-status-text", "children", allow_duplicate=True),
     Output("connection-actions", "children", allow_duplicate=True)],
    [Input("disconnect-btn", "n_clicks")],
    prevent_initial_call=True
)
def disconnect_database(n_clicks):
    if n_clicks:
        db_connection.disconnect()
        return (
            {"connected": False},
            "status-indicator status-disconnected",
            "Not connected to database",
            ""
        )
    return no_update

@app.callback(
    [Output("db-host", "disabled"),
     Output("db-port", "disabled"),
     Output("db-username", "disabled"),
     Output("db-password", "disabled"),
     Output("db-name", "disabled")],
    [Input("db-type", "value")]
)
def toggle_connection_fields(db_type):
    is_sqlite = db_type == "sqlite3"
    return is_sqlite, is_sqlite, is_sqlite, is_sqlite, is_sqlite

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
