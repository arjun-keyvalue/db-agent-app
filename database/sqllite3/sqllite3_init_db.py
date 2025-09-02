import sqlite3
import os

DB_FILE = "employee_management.db"        # The SQLite database file
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
    """Run simple tests to verify DB contents."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    print("\n📋 Tables in the database:")
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';").fetchall()
    for t in tables:
        print(" -", t[0])

    print("\n📊 Row counts per table:")
    for t in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {t[0]};").fetchone()[0]
        print(f"   {t[0]}: {count} rows")

    print("\n👤 Sample Employees:")
    employees = cursor.execute("""
        SELECT employee_id, first_name || ' ' || last_name AS full_name, email, status
        FROM Employees LIMIT 3;
    """).fetchall()
    for u in employees:
        print("  ", u)

    print("\n📚 Sample Projects:")
    projects = cursor.execute("""
        SELECT project_id, name, status, start_date, end_date
        FROM Projects LIMIT 3;
    """).fetchall()
    for p in projects:
        print("  ", p)

    print("\n🛠 Sample Skills:")
    skills = cursor.execute("""
        SELECT skill_id, name, category
        FROM Skills LIMIT 3;
    """).fetchall()
    for s in skills:
        print("  ", s)

    print("\n📈 Sample Performance Reviews:")
    reviews = cursor.execute("""
        SELECT review_id, employee_id, reviewer_id, rating, review_cycle
        FROM Performance_Reviews LIMIT 3;
    """).fetchall()
    for r in reviews:
        print("  ", r)

    conn.close()


if __name__ == "__main__":
    init_db()
    test_db()
