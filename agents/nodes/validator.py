"""
Validation nodes for SQL queries - Layer 1 & 2: Syntactic and Semantic Validation
"""

import logging
from typing import Dict, Any
import sqlfluff
import sqlglot
from sqlglot import parse_one, ParseError
from ..states import AgentState
from ..step_logger import AgentStepLogger

logger = logging.getLogger(__name__)


class SyntacticValidatorNode:
    """Layer 1: Syntactic validation using SQLFluff and SQLGlot"""

    def __init__(self):
        # Configure SQLFluff for different dialects
        self.sqlfluff_config = {
            "postgresql": {"dialect": "postgres"},
            "mysql": {"dialect": "mysql"},
            "sqlite3": {"dialect": "sqlite"},
        }

    def __call__(self, state: AgentState) -> AgentState:
        """Validate SQL syntax"""
        sql_query = state.get("sql_query", "")
        db_type = state.get("database_type", "postgresql")

        if not sql_query:
            state["validation_errors"] = ["No SQL query provided"]
            state["syntax_valid"] = False
            state["next_action"] = "generate_query"
            return state

        validation_errors = []

        # SQLFluff validation
        try:
            config = self.sqlfluff_config.get(
                db_type, self.sqlfluff_config["postgresql"]
            )
            lint_result = sqlfluff.lint(sql_query, **config)

            for violation in lint_result:
                if violation["code"] in [
                    "L001",
                    "L002",
                    "L003",
                ]:  # Critical syntax errors
                    validation_errors.append(
                        f"SQLFluff {violation['code']}: {violation['description']}"
                    )

        except Exception as e:
            validation_errors.append(f"SQLFluff validation error: {str(e)}")

        # SQLGlot parsing validation
        try:
            dialect_map = {
                "postgresql": "postgres",
                "mysql": "mysql",
                "sqlite3": "sqlite",
            }
            dialect = dialect_map.get(db_type, "postgres")

            parsed = parse_one(sql_query, dialect=dialect)
            if not parsed:
                validation_errors.append("SQLGlot: Failed to parse SQL query")

        except ParseError as e:
            validation_errors.append(f"SQLGlot parse error: {str(e)}")
        except Exception as e:
            validation_errors.append(f"SQLGlot validation error: {str(e)}")

        # Update state - be less strict, allow minor syntax issues to proceed
        state["validation_errors"] = validation_errors
        state["syntax_valid"] = len(validation_errors) == 0

        # Count critical vs minor errors
        critical_errors = [
            err
            for err in validation_errors
            if any(
                critical in err.lower()
                for critical in ["parse error", "syntax error", "invalid sql"]
            )
        ]

        if len(critical_errors) == 0:
            # No critical errors, proceed to semantic validation
            state["next_action"] = "semantic_validation"
            AgentStepLogger.log_validation("syntactic", True)
            if validation_errors:
                logger.debug(
                    f"Minor syntax issues detected but proceeding: {validation_errors}"
                )
        else:
            # Critical syntax errors, try correction
            state["next_action"] = "self_correction"
            AgentStepLogger.log_validation("syntactic", False, critical_errors)

        return state


class SemanticValidatorNode:
    """Layer 2: Semantic validation - check tables and columns exist"""

    def __init__(self, db_connection):
        self.db_connection = db_connection

    def __call__(self, state: AgentState) -> AgentState:
        """Validate that tables and columns referenced in SQL actually exist"""
        sql_query = state.get("sql_query", "")
        db_type = state.get("database_type", "postgresql")

        if not state.get("syntax_valid", False):
            # Skip semantic validation if syntax is invalid
            state["semantic_valid"] = False
            return state

        validation_errors = []

        try:
            # Parse the SQL to extract table and column references
            dialect_map = {
                "postgresql": "postgres",
                "mysql": "mysql",
                "sqlite3": "sqlite",
            }
            dialect = dialect_map.get(db_type, "postgres")

            parsed = parse_one(sql_query, dialect=dialect)

            # Extract table names
            referenced_tables = set()
            referenced_columns = {}  # table -> [columns]

            for table in parsed.find_all(sqlglot.expressions.Table):
                table_name = table.name
                referenced_tables.add(table_name)
                referenced_columns[table_name] = []

            # Extract column references
            for column in parsed.find_all(sqlglot.expressions.Column):
                column_name = column.name
                table_name = column.table if column.table else None

                if table_name:
                    if table_name not in referenced_columns:
                        referenced_columns[table_name] = []
                    referenced_columns[table_name].append(column_name)
                else:
                    # Column without table prefix - need to check all referenced tables
                    for table in referenced_tables:
                        if table not in referenced_columns:
                            referenced_columns[table] = []
                        referenced_columns[table].append(column_name)

            # Validate tables exist
            tables_success, tables_result = self.db_connection.get_tables()
            if not tables_success:
                validation_errors.append(
                    f"Failed to retrieve database tables: {tables_result}"
                )
                state["validation_errors"] = validation_errors
                state["semantic_valid"] = False
                state["next_action"] = "self_correction"
                return state

            existing_tables = set(tables_result["table_name"].str.lower())

            for table in referenced_tables:
                if table.lower() not in existing_tables:
                    validation_errors.append(
                        f"Table '{table}' does not exist in database"
                    )

            # Validate columns exist for each table
            for table_name, columns in referenced_columns.items():
                if table_name.lower() not in existing_tables:
                    continue  # Already reported table doesn't exist

                schema_success, schema_result = self.db_connection.get_table_schema(
                    table_name
                )
                if not schema_success:
                    validation_errors.append(
                        f"Failed to retrieve schema for table '{table_name}'"
                    )
                    continue

                existing_columns = set(schema_result["column_name"].str.lower())

                for column in columns:
                    if column.lower() not in existing_columns:
                        validation_errors.append(
                            f"Column '{column}' does not exist in table '{table_name}'"
                        )

        except Exception as e:
            validation_errors.append(f"Semantic validation error: {str(e)}")

        # Update state - be less strict with semantic validation too
        current_errors = state.get("validation_errors", [])
        state["validation_errors"] = current_errors + validation_errors
        state["semantic_valid"] = len(validation_errors) == 0

        # Count critical semantic errors vs minor ones
        critical_semantic_errors = [
            err for err in validation_errors if "does not exist" in err.lower()
        ]

        # If we have too many correction attempts, proceed anyway
        correction_attempts = state.get("correction_attempts", 0)
        max_corrections = state.get("max_corrections", 3)

        if (
            len(critical_semantic_errors) == 0
            or correction_attempts >= max_corrections - 1
        ):
            # No critical semantic errors OR we've tried enough corrections, proceed
            state["next_action"] = "performance_guard"
            AgentStepLogger.log_validation("semantic", True)
            if validation_errors and correction_attempts < max_corrections - 1:
                logger.debug(
                    f"Minor semantic issues detected but proceeding: {validation_errors}"
                )
            elif correction_attempts >= max_corrections - 1:
                logger.debug("Max corrections reached, proceeding with query execution")
        else:
            # Critical semantic errors and we haven't exceeded correction limit
            state["next_action"] = "self_correction"
            AgentStepLogger.log_validation("semantic", False, critical_semantic_errors)

        return state
