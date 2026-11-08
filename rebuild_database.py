
import pandas as pd
import sqlite3
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Clean CSV files
customers = pd.read_csv(BASE_DIR / "data/clean/customers.csv")
orders = pd.read_csv(
    BASE_DIR / "data/clean/orders.csv",
    dtype={"order_ts": "string"}
)
products = pd.read_csv(BASE_DIR / "data/clean/products.csv")

# Database path
db_path = BASE_DIR / "data/instamart.db"

# Rebuild database
with sqlite3.connect(db_path) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)

    print("Database tables created successfully!")

    for table in ["customers", "orders", "products"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}",
            conn
        )["total"][0]

        print(f"{table}: {count} rows")
