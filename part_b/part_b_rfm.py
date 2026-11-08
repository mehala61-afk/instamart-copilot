from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

ORDERS_PATH = BASE_DIR / "data" / "clean" / "orders.csv"
OUTPUT_DIR = BASE_DIR / "data" / "clean"
OUTPUT_PATH = OUTPUT_DIR / "customer_rfm.csv"

# Read orders
orders = pd.read_csv(ORDERS_PATH)

# Keep completed orders only
orders = orders[orders["status"] == "completed"].copy()

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Fixed snapshot moment
snapshot = pd.Timestamp("2025-10-01 00:00:00")

# Calculate RFM
rfm = orders.groupby("customer_id").agg(
    recency=("order_ts",
             lambda x: (snapshot - x.max()).days),

    frequency=("order_id", "nunique"),

    monetary=("order_amount", "sum")
).reset_index()

# Round monetary
rfm["monetary"] = rfm["monetary"].round(2)

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rfm.to_csv(OUTPUT_PATH, index=False)

print("RFM Analysis completed!")
print("Customers:", len(rfm))
print("\nRFM Summary:")
print(rfm[["recency", "frequency", "monetary"]].agg(
    ["mean", "median", "min", "max"]
))

print("\nOutput saved:", OUTPUT_PATH)