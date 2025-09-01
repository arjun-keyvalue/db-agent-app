from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Tuple

class SQLGenerator:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        self.model = ChatOpenAI(
            model_name=model_name,
            openai_api_key=openai_api_key,
            temperature=0
        )
        
        self.system_template = """You are an expert SQL generator.

You will be given:
1. Database schema details (tables, columns, data types).
2. Join relationships between tables.
3. Sample rows from relevant tables.
4. A natural language question.

Your task:
- Think step by step before writing SQL.
- Identify which tables are needed and which can be ignored.
- Figure out correct joins based on the provided relationships.
- Use sample rows to reason about column semantics if needed.
- Write a single, syntactically correct SQL query.
- Do not invent tables or columns that are not provided.
- Return only the SQL query, no explanations, no markdown.
- The db used is sqlite3

---

Schema and sample data:
{retrieved_context}

---

Let's reason step by step:
1. Identify tables needed and why.
2. Plan the joins.
3. Apply filters, grouping, and aggregations.
4. Write final SQL.

Now return only the SQL query."""

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{user_query}")
        ])

    def generate_sql(self, user_query: str, schema_chunks: List[str]) -> str:
        """Generate SQL query from natural language using LLM"""
        # Prepare the prompt
        prompt = self.prompt_template.invoke({
            "retrieved_context": "\n".join(schema_chunks),
            "user_query": user_query
        })
        
        # Get response from LLM
        response = self.model.invoke(prompt)
        
        # Clean up the response
        sql = response.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return sql