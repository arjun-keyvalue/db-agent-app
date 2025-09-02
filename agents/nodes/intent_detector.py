"""
Intent detection node for understanding user query intent
"""

import logging
from typing import Dict, Any
from ..llm_client import SmartLLMClient
from ..states import AgentState

logger = logging.getLogger(__name__)


class IntentDetectorNode:
    """Detect user intent from natural language query"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        from ..llm_client import SmartLLMClient

        self.llm_client = SmartLLMClient()

        # Define intent categories
        self.intent_categories = {
            "data_retrieval": "User wants to retrieve/view data",
            "aggregation": "User wants to calculate sums, counts, averages, etc.",
            "filtering": "User wants to filter data based on conditions",
            "joining": "User wants to combine data from multiple tables",
            "sorting": "User wants to order/sort results",
            "comparison": "User wants to compare different data points",
            "temporal": "User wants to analyze data over time",
            "statistical": "User wants statistical analysis",
            "visualization": "User wants to create charts or graphs",
            "schema_exploration": "User wants to understand database structure",
            "complex_analysis": "User wants complex multi-step analysis",
        }

    def __call__(self, state: AgentState) -> AgentState:
        """Detect intent from user query"""

        user_query = state.get("user_query", "")

        if not user_query:
            state["error_message"] = "No user query provided for intent detection"
            state["next_action"] = "output_formatter"
            return state

        try:
            # Create intent detection prompt
            intent_prompt = self._create_intent_prompt(user_query)

            # Use Smart LLM client for intent detection
            response = self.llm_client.completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at understanding database query intentions. Analyze user queries and identify their intent.",
                    },
                    {"role": "user", "content": intent_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            intent_analysis = response.choices[0].message.content.strip()

            # Parse the intent analysis
            intent_info = self._parse_intent_response(intent_analysis)

            # Store intent information in state
            state["user_intent"] = intent_info

            # Check if query should be rejected
            if (
                intent_info.get("should_reject", False)
                or intent_info.get("primary_intent") == "non_database_query"
            ):
                state["error_message"] = (
                    "I can only help with database-related queries. Please ask about your data, tables, or specific information you'd like to retrieve from the database."
                )
                state["next_action"] = "output_formatter"
                state["success"] = False
                logger.debug("Query rejected: Not a database-related query")
                return state

            state["next_action"] = "context_retriever"

            logger.debug(
                f"Intent detected: {intent_info.get('primary_intent', 'unknown')}"
            )

        except Exception as e:
            error_msg = f"LLM intent detection failed: {str(e)}"
            logger.warning(error_msg)
            logger.info("Falling back to keyword-based intent detection")

            # Use fallback intent detection
            intent_info = self._fallback_intent_detection(user_query)
            state["user_intent"] = intent_info

            # Check if query should be rejected (same logic as above)
            if (
                intent_info.get("should_reject", False)
                or intent_info.get("primary_intent") == "non_database_query"
            ):
                state["error_message"] = (
                    "I can only help with database-related queries. Please ask about your data, tables, or specific information you'd like to retrieve from the database."
                )
                state["next_action"] = "output_formatter"
                state["success"] = False
                logger.info(
                    "Query rejected by fallback intent detector: Not a database-related query"
                )
                return state

            state["next_action"] = "context_retriever"
            logger.info(
                f"Fallback intent detected: {intent_info.get('primary_intent', 'unknown')}"
            )

        return state

    def _create_intent_prompt(self, user_query: str) -> str:
        """Create prompt for intent detection"""

        intent_descriptions = "\n".join(
            [
                f"- {intent}: {description}"
                for intent, description in self.intent_categories.items()
            ]
        )

        prompt = f"""
Analyze the following database query and determine the user's intent.

USER QUERY: "{user_query}"

POSSIBLE INTENTS:
{intent_descriptions}

Please analyze the query and respond in the following JSON format:
{{
    "primary_intent": "most_likely_intent_category",
    "secondary_intents": ["other_relevant_intents"],
    "confidence": 0.95,
    "complexity": "low|medium|high",
    "requires_joins": true/false,
    "requires_aggregation": true/false,
    "requires_filtering": true/false,
    "requires_sorting": true/false,
    "temporal_aspect": true/false,
    "estimated_tables_needed": 2,
    "key_entities": ["entity1", "entity2"],
    "reasoning": "Brief explanation of why this intent was chosen"
}}

Respond with valid JSON only:
"""

        return prompt

    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """Parse the intent detection response"""

        try:
            import json

            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            intent_data = json.loads(response)

            # Validate required fields
            required_fields = ["primary_intent", "confidence"]
            for field in required_fields:
                if field not in intent_data:
                    intent_data[field] = "unknown" if field == "primary_intent" else 0.0

            # Ensure confidence is between 0 and 1
            if intent_data["confidence"] > 1.0:
                intent_data["confidence"] = intent_data["confidence"] / 100.0

            return intent_data

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse intent response: {e}")

            # Fallback: simple keyword-based intent detection
            return self._fallback_intent_detection(response)

    def _fallback_intent_detection(self, user_query: str) -> Dict[str, Any]:
        """Fallback intent detection using simple keyword matching"""

        query_lower = user_query.lower().strip()

        # Check for non-database queries that should be rejected
        non_db_patterns = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "what's up",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "ok",
            "okay",
            "yes",
            "no",
            "maybe",
            "sure",
            "fine",
            "great",
            "awesome",
            "help",
            "what can you do",
            "who are you",
            "what are you",
        ]

        # If query is too short or matches non-database patterns, reject it
        # Also check if query starts with common greetings
        starts_with_greeting = any(
            query_lower.startswith(pattern) for pattern in non_db_patterns[:6]
        )  # hi, hello, hey, good morning, etc.

        if (
            len(query_lower) < 3
            or query_lower in non_db_patterns
            or starts_with_greeting
        ):
            return {
                "primary_intent": "non_database_query",
                "secondary_intents": [],
                "confidence": 0.9,
                "complexity": "low",
                "requires_joins": False,
                "requires_aggregation": False,
                "requires_filtering": False,
                "requires_sorting": False,
                "temporal_aspect": False,
                "estimated_tables_needed": 0,
                "key_entities": [],
                "reasoning": "Query appears to be a greeting or non-database related",
                "should_reject": True,
            }

        # Simple keyword-based detection for database queries
        if any(
            word in query_lower
            for word in ["show", "list", "display", "get", "find", "select"]
        ):
            primary_intent = "data_retrieval"
        elif any(
            word in query_lower
            for word in ["count", "sum", "average", "total", "max", "min"]
        ):
            primary_intent = "aggregation"
        elif any(
            word in query_lower for word in ["chart", "graph", "plot", "visualize"]
        ):
            primary_intent = "visualization"
        elif any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            primary_intent = "comparison"
        elif any(
            word in query_lower
            for word in ["over time", "trend", "history", "timeline"]
        ):
            primary_intent = "temporal"
        else:
            # Check if it contains database-related keywords
            db_keywords = [
                "table",
                "database",
                "record",
                "row",
                "column",
                "data",
                "query",
                "sql",
            ]
            if any(keyword in query_lower for keyword in db_keywords):
                primary_intent = "data_retrieval"
            else:
                # If no database keywords found, might be non-database query
                return {
                    "primary_intent": "unclear_intent",
                    "secondary_intents": [],
                    "confidence": 0.3,
                    "complexity": "medium",
                    "requires_joins": False,
                    "requires_aggregation": False,
                    "requires_filtering": False,
                    "requires_sorting": False,
                    "temporal_aspect": False,
                    "estimated_tables_needed": 1,
                    "key_entities": [],
                    "reasoning": "Query intent unclear - may not be database related",
                    "should_reject": False,  # Let it proceed but with low confidence
                }

        return {
            "primary_intent": primary_intent,
            "secondary_intents": [],
            "confidence": 0.6,  # Lower confidence for fallback
            "complexity": "medium",
            "requires_joins": "join" in query_lower or "with" in query_lower,
            "requires_aggregation": any(
                word in query_lower for word in ["count", "sum", "average", "total"]
            ),
            "requires_filtering": any(
                word in query_lower for word in ["where", "filter", "only", "specific"]
            ),
            "requires_sorting": any(
                word in query_lower for word in ["order", "sort", "rank"]
            ),
            "temporal_aspect": any(
                word in query_lower for word in ["time", "date", "year", "month", "day"]
            ),
            "estimated_tables_needed": 2 if "join" in query_lower else 1,
            "key_entities": [],
            "reasoning": "Fallback keyword-based detection",
            "should_reject": False,
        }
