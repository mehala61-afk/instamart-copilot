import pandas as pd
import sqlite3
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Clean data folder
DATA_DIR = BASE_DIR / "data" / "clean"

# Database path
DB_PATH = BASE_DIR / "data" / "instamart.db"

# Read clean CSV files
customers = pd.read_csv(DATA_DIR / "customers.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

# Load into SQLite database
with sqlite3.connect(DB_PATH) as conn:

    customers.to_sql(
        "customers", conn,
        if_exists="replace", index=False
    )

    orders.to_sql(
        "orders", conn,
        if_exists="replace", index=False
    )

    products.to_sql(
        "products", conn,
        if_exists="replace", index=False
    )

    print("Database created:", DB_PATH)

    print("\nTable row counts:")

    for table in ["customers", "orders", "products"]:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {count} rows")
