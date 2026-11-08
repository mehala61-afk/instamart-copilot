import sqlite3
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "data" / "instamart.db"
CLEAN_DIR = BASE_DIR / "data" / "clean"

print("\n--- PART A VALIDATION ---")

# 1. Check required files
required_files = [
    DB_PATH,
    CLEAN_DIR / "customers.csv",
    CLEAN_DIR / "orders.csv",
    CLEAN_DIR / "products.csv"
]

for file in required_files:
    assert file.exists(), f"Missing file: {file.name}"

print("Checkpoint 1: Required files exist - PASSED")

# 2. Database validation
conn = sqlite3.connect(DB_PATH)

customers = pd.read_sql_query(
    "SELECT * FROM customers", conn
)

orders = pd.read_sql_query(
    "SELECT * FROM orders", conn
)

products = pd.read_sql_query(
    "SELECT * FROM products", conn
)

assert len(customers) == 1200
assert len(orders) == 12309
assert len(products) == 40

print("Checkpoint 2: Database row counts - PASSED")

# 3. Duplicate order IDs
duplicate_ids = orders["order_id"].duplicated().sum()

assert duplicate_ids == 0

print("Checkpoint 3: No duplicate order IDs - PASSED")

# 4. Order status counts
status_counts = orders["status"].value_counts().to_dict()

assert status_counts.get("completed", 0) == 11110
assert status_counts.get("cancelled", 0) == 1199
assert status_counts.get("pending", 0) == 0

print("Checkpoint 4: Order status counts - PASSED")

# 5. Missing order amounts
missing_amounts = orders["order_amount"].isna().sum()

assert missing_amounts == 0

print("Checkpoint 5: No missing order amounts - PASSED")

# 6. Completed order delivery time
completed = orders[orders["status"] == "completed"]

assert completed["delivery_minutes"].isna().sum() == 0

print("Checkpoint 6: Completed delivery times - PASSED")

# 7. Cancelled order delivery time
cancelled = orders[orders["status"] == "cancelled"]

assert cancelled["delivery_minutes"].isna().sum() == 1199

print("Checkpoint 7: Cancelled delivery times - PASSED")

# 8. Completed revenue
total_revenue = round(completed["order_amount"].sum(), 2)

assert total_revenue == 2998668.80

print("Checkpoint 8: Completed revenue - PASSED")

# 9. Customer data validation
assert customers["age"].isna().sum() == 0
assert customers["city"].nunique() == 6

print("Checkpoint 9: Customer data - PASSED")

# 10. Product data validation
assert products["product_id"].nunique() == 40

print("Checkpoint 10: Product data - PASSED")

conn.close()

print("\nALL CHECKPOINTS PASSED")