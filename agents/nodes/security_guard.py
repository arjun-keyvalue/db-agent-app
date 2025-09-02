"""
Security guardrails - Layer 5: Security and Privacy Protection
"""

import logging
import re
from typing import Dict, Any
from ..states import AgentState
from ..step_logger import AgentStepLogger

logger = logging.getLogger(__name__)


class SecurityGuardNode:
    """Layer 5: Security guardrails for sensitive data protection"""

    def __init__(self):
        # Define sensitive tables that contain PII or credentials
        self.sensitive_tables = [
            "users", "user", "accounts", "account", 
            "employees", "employee", "employee_management"
        ]

        # Define safe columns for sensitive tables (columns that can be safely exposed)
        self.safe_columns = {
            "users": [
                "id",
                "username",
                "full_name",
                "is_active",
                "created_at",
                "updated_at",
            ],
            "user": [
                "id",
                "username",
                "full_name",
                "is_active",
                "created_at",
                "updated_at",
            ],
            "accounts": [
                "id",
                "username",
                "is_active",
                "created_at",
                "updated_at",
            ],
            "account": [
                "id",
                "username",
                "is_active",
                "created_at",
                "updated_at",
            ],
            "employees": [
                "employee_id",
                "first_name",
                "last_name",
                "hire_date",
                "status",
                "role_id",
                "manager_id",
                "team_id",
            ],
            "employee": [
                "employee_id",
                "first_name",
                "last_name",
                "hire_date",
                "status",
                "role_id",
                "manager_id",
                "team_id",
            ],
            "employee_management": [
                "employee_id",
                "first_name",
                "last_name",
                "hire_date",
                "status",
                "role_id",
                "manager_id",
                "team_id",
            ],
        }

    def __call__(self, state: AgentState) -> AgentState:
        """Apply security guardrails to protect sensitive data"""

        sql_query = state.get("sql_query", "")
        user_query = state.get("user_query", "")

        if not sql_query:
            state["error_message"] = "No SQL query to apply security guardrails"
            state["next_action"] = "output_formatter"
            return state

        # Check if security guardrail is enabled
        # The security setting should be passed through the context or state
        security_enabled = state.get("security_enabled", True)  # Default to True for safety
        
        if not security_enabled:
            # Security guardrail is disabled, just pass through to query executor
            logger.info("Security guardrail is disabled, skipping security validation")
            state["next_action"] = "query_executor"
            return state

        try:
            # Check user intent first - if they're explicitly asking for sensitive data, block the query
            user_input = user_query.lower()
            sensitive_intent_markers = [
                "email", "emails", "e-mail", "e-mails",
                "phone", "phone number", "phone numbers", "telephone", "telephone number",
                "password", "passwords", "passwd", "pwd",
                "date of birth", "birth date", "dob", "birthday",
                "social security", "ssn", "ss number",
                "credit card", "cc number", "card number",
                "address", "home address", "mailing address",
                "personal information", "pii", "private data",
            ]

            # If user is explicitly asking for sensitive data, block the query
            if any(marker in user_input for marker in sensitive_intent_markers):
                state["error_message"] = "🚫 Security Check Failed: Blocked by guardrails - request for sensitive personal information is not allowed."
                state["next_action"] = "self_correction"
                state["success"] = False
                state["validation_errors"] = ["Security violation: Request for sensitive data blocked"]
                return state

            # Apply security modifications to the query
            modified_query = self._apply_security_modifications(sql_query)
            
            if modified_query != sql_query:
                state["sql_query"] = modified_query
                logger.info("Security guardrail modified query to exclude sensitive columns")
                AgentStepLogger.log_validation("security", True, "Query modified to exclude sensitive data")
            else:
                AgentStepLogger.log_validation("security", True, "No sensitive data detected")

            # Proceed to execution
            state["next_action"] = "query_executor"
            return state

        except Exception as e:
            logger.error(f"Security validation error: {e}")
            state["error_message"] = "🚫 Security Check Failed: Security validation failed due to internal error"
            state["next_action"] = "self_correction"
            state["success"] = False
            state["validation_errors"] = ["Security validation error: Internal error occurred"]
            return state

    def _apply_security_modifications(self, sql_query: str) -> str:
        """Apply security modifications to the SQL query"""
        
        # Check for SELECT queries on sensitive tables
        select_match = re.search(
            r"SELECT\s+(.*?)\s+FROM\s+([\w.]+)",
            sql_query,
            re.IGNORECASE | re.DOTALL,
        )
        
        if select_match:
            select_clause = select_match.group(1).strip()
            full_table_name = select_match.group(2).strip()
            # Extract table name without schema prefix
            table_name = full_table_name.split('.')[-1].lower()
            
            # If querying a sensitive table, modify the SELECT clause to only include safe columns
            if table_name in self.sensitive_tables:
                if select_clause == "*":
                    # Replace SELECT * with only safe columns
                    safe_cols = self.safe_columns.get(table_name, [])
                    if safe_cols:
                        modified_select = ", ".join(safe_cols)
                        sql_query = re.sub(
                            r"SELECT\s+\*\s+FROM\s+" + re.escape(full_table_name),
                            f"SELECT {modified_select} FROM {full_table_name}",
                            sql_query,
                            flags=re.IGNORECASE,
                        )
                        logger.info(f"Modified SELECT * query on sensitive table '{table_name}' to only include safe columns: {safe_cols}")
                    else:
                        logger.warning(f"No safe columns defined for sensitive table '{table_name}'")
                else:
                    # Check if any sensitive columns are explicitly selected
                    columns = [c.strip() for c in select_clause.split(",")]
                    sensitive_cols_found = []
                    safe_cols_to_keep = []

                    for col in columns:
                        # Normalize column name
                        normalized_col = col.lower()
                        # Remove AS alias
                        if " as " in normalized_col:
                            normalized_col = normalized_col.split(" as ")[0].strip()
                        # Remove table prefix
                        if "." in normalized_col:
                            normalized_col = normalized_col.split(".")[-1].strip()

                        # Check if it's a sensitive column
                        if re.search(
                            r"\b(user_?name|username|password|passwd|password_?hash|pwd_hash|email|phone|phone_?number|address|date_?of_?birth|dob|ssn|social_?security|credit_?card|card_?number)\b",
                            normalized_col,
                        ):
                            sensitive_cols_found.append(col.strip())
                        else:
                            safe_cols_to_keep.append(col.strip())

                    # If sensitive columns were found, remove them from the query
                    if sensitive_cols_found:
                        if safe_cols_to_keep:
                            # Replace the SELECT clause with only safe columns
                            modified_select = ", ".join(safe_cols_to_keep)
                            sql_query = re.sub(
                                r"SELECT\s+"
                                + re.escape(select_clause)
                                + r"\s+FROM\s+"
                                + re.escape(full_table_name),
                                f"SELECT {modified_select} FROM {full_table_name}",
                                sql_query,
                                flags=re.IGNORECASE,
                            )
                            logger.info(f"Removed sensitive columns from query on '{table_name}': {sensitive_cols_found}")
                        else:
                            # If no safe columns remain, use default safe columns
                            safe_cols = self.safe_columns.get(table_name, [])
                            if safe_cols:
                                modified_select = ", ".join(safe_cols)
                                sql_query = re.sub(
                                    r"SELECT\s+"
                                    + re.escape(select_clause)
                                    + r"\s+FROM\s+"
                                    + re.escape(full_table_name),
                                    f"SELECT {modified_select} FROM {full_table_name}",
                                    sql_query,
                                    flags=re.IGNORECASE,
                                )
                                logger.info(f"Replaced query with safe columns for '{table_name}': {safe_cols}")
                            else:
                                logger.warning(f"No safe columns available for sensitive table '{table_name}'")

        return sql_query
