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
            "complex_analysis": "User wants complex multi-step analysis"
        }
    
    def __call__(self, state: AgentState) -> AgentState:
        """Detect intent from user query"""
        
        user_query = state.get('user_query', '')
        
        if not user_query:
            state['error_message'] = "No user query provided for intent detection"
            state['next_action'] = 'output_formatter'
            return state
        
        try:
            # Create intent detection prompt
            intent_prompt = self._create_intent_prompt(user_query)
            
            # Use Smart LLM client for intent detection
            response = self.llm_client.completion(
                messages=[
                    {"role": "system", "content": "You are an expert at understanding database query intentions. Analyze user queries and identify their intent."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            intent_analysis = response.choices[0].message.content.strip()
            
            # Parse the intent analysis
            intent_info = self._parse_intent_response(intent_analysis)
            
            # Store intent information in state
            state['user_intent'] = intent_info
            state['next_action'] = 'schema_retriever'
            
            logger.info(f"Intent detected: {intent_info.get('primary_intent', 'unknown')}")
            
        except Exception as e:
            error_msg = f"Intent detection failed: {str(e)}"
            logger.error(error_msg)
            # Continue without intent detection
            state['user_intent'] = {"primary_intent": "unknown", "confidence": 0.0}
            state['next_action'] = 'schema_retriever'
        
        return state
    
    def _create_intent_prompt(self, user_query: str) -> str:
        """Create prompt for intent detection"""
        
        intent_descriptions = "\n".join([
            f"- {intent}: {description}" 
            for intent, description in self.intent_categories.items()
        ])
        
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
            required_fields = ['primary_intent', 'confidence']
            for field in required_fields:
                if field not in intent_data:
                    intent_data[field] = 'unknown' if field == 'primary_intent' else 0.0
            
            # Ensure confidence is between 0 and 1
            if intent_data['confidence'] > 1.0:
                intent_data['confidence'] = intent_data['confidence'] / 100.0
            
            return intent_data
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse intent response: {e}")
            
            # Fallback: simple keyword-based intent detection
            return self._fallback_intent_detection(response)
    
    def _fallback_intent_detection(self, user_query: str) -> Dict[str, Any]:
        """Fallback intent detection using simple keyword matching"""
        
        query_lower = user_query.lower()
        
        # Simple keyword-based detection
        if any(word in query_lower for word in ['show', 'list', 'display', 'get', 'find']):
            primary_intent = 'data_retrieval'
        elif any(word in query_lower for word in ['count', 'sum', 'average', 'total', 'max', 'min']):
            primary_intent = 'aggregation'
        elif any(word in query_lower for word in ['chart', 'graph', 'plot', 'visualize']):
            primary_intent = 'visualization'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            primary_intent = 'comparison'
        elif any(word in query_lower for word in ['over time', 'trend', 'history', 'timeline']):
            primary_intent = 'temporal'
        else:
            primary_intent = 'data_retrieval'  # Default
        
        return {
            'primary_intent': primary_intent,
            'secondary_intents': [],
            'confidence': 0.6,  # Lower confidence for fallback
            'complexity': 'medium',
            'requires_joins': 'join' in query_lower or 'with' in query_lower,
            'requires_aggregation': any(word in query_lower for word in ['count', 'sum', 'average', 'total']),
            'requires_filtering': any(word in query_lower for word in ['where', 'filter', 'only', 'specific']),
            'requires_sorting': any(word in query_lower for word in ['order', 'sort', 'rank']),
            'temporal_aspect': any(word in query_lower for word in ['time', 'date', 'year', 'month', 'day']),
            'estimated_tables_needed': 2 if 'join' in query_lower else 1,
            'key_entities': [],
            'reasoning': 'Fallback keyword-based detection'
        }