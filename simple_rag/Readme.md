# Demo Platform

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/pthom/northwind_psql.git
   ```

2. Navigate into the cloned directory:
   ```
   cd northwind_psql
   ```

3. Run Docker Compose to start the services:
   ```
   docker compose up
   ```

4. Set up Qdrant using Docker:
   ```
   docker run -p 6333:6333 -p 6334:6334 \
       -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
       qdrant/qdrant
   ```

5. Navigate to the parent directory:
   ```
   cd ..
   ```


6. Set up the `.env` file with the following content:
   ```
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=northwind_rag
   GROQ_API_KEY=
   ```

7. Run the embedding insertion script (first time only):
   ```
   python embedd_insert.py
   ```

## Sample Query

- **Query:** List the top 5 customers by total order value.
