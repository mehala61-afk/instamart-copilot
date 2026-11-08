
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "instamart.db"

with sqlite3.connect(DB_PATH) as conn:

    print("\n--- DATABASE VALIDATION ---")

    # 1. Check row counts
    for table in ["customers", "orders", "products"]:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {count}")

    # 2. Check duplicate order IDs
    duplicates = conn.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT order_id
            FROM orders
            GROUP BY order_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    print("Duplicate order IDs:", duplicates)

    # 3. Check missing order amounts
    missing_amount = conn.execute("""
        SELECT COUNT(*)
        FROM orders
        WHERE order_amount IS NULL
    """).fetchone()[0]

    print("Missing order amounts:", missing_amount)

    # 4. Check order status
    print("\nOrder status counts:")

    result = conn.execute("""
        SELECT status, COUNT(*)
        FROM orders
        GROUP BY status
    """).fetchall()

    for row in result:
        print(row)

print("\nDatabase validation completed!")
