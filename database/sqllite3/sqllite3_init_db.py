import sqlite3
import os

DB_FILE = "library.db"        # The SQLite database file
SEED_FILE = "sqllite3_seed.sql" # The SQL file with schema + inserts

def init_db(db_file=DB_FILE, seed_file=SEED_FILE):
    # Read the SQL script
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"Seed file '{seed_file}' not found")

    with open(seed_file, "r", encoding="utf-8") as f:
        sql_script = f.read()

    # Connect to SQLite (creates the DB file if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Execute the SQL script
    try:
        cursor.executescript(sql_script)
        conn.commit()
        print(f"✅ Database '{db_file}' initialized successfully using '{seed_file}'")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
    finally:
        conn.close()

def test_db(db_file=DB_FILE):
    """Run some simple tests to verify DB contents."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    print("\n📋 Tables in the database:")
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    for t in tables:
        print(" -", t[0])

    print("\n📊 Row counts per table:")
    for t in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {t[0]};").fetchone()[0]
        print(f"   {t[0]}: {count} rows")

    print("\n👤 Sample users:")
    users = cursor.execute("SELECT username, email, full_name FROM users LIMIT 3;").fetchall()
    for u in users:
        print("  ", u)

    print("\n📚 Sample books:")
    books = cursor.execute("SELECT title, author, publication_year FROM books LIMIT 3;").fetchall()
    for b in books:
        print("  ", b)

    conn.close()

if __name__ == "__main__":
    init_db()
    test_db()
