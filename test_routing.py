"""
Test script to verify RAG agent routing
"""

import os
import sys
from database.query_engine import query_engine_factory
from config import Config

def test_routing():
    """Test that the RAG advanced option routes correctly"""
    
    print("🧪 Testing RAG Agent Routing")
    print("=" * 40)
    
    # Test configuration
    config = {
        'openai_api_key': 'test-key'  # Dummy key for testing
    }
    
    try:
        # Test schema engine
        schema_engine = query_engine_factory.create_query_engine("schema", config)
        print(f"✅ Schema Engine: {schema_engine.get_name()}")
        
        # Test basic RAG engine
        rag_engine = query_engine_factory.create_query_engine("rag", config)
        print(f"✅ RAG Engine: {rag_engine.get_name()}")
        
        # Test advanced RAG engine
        rag_advanced_engine = query_engine_factory.create_query_engine("rag_advanced", config)
        print(f"✅ Advanced RAG Engine: {rag_advanced_engine.get_name()}")
        
        print("\n🎉 All engines created successfully!")
        print(f"Advanced RAG uses model: {Config.RAG_MODEL}")
        print(f"LanceDB path: {Config.LANCEDB_PATH}")
        print(f"Max corrections: {Config.MAX_CORRECTIONS}")
        
        return True
        
    except Exception as e:
        print(f"❌ Routing test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_routing()
    if not success:
        print("\n💡 Make sure to install dependencies:")
        print("pip install langgraph litellm lancedb sqlfluff sqlglot")
        sys.exit(1)