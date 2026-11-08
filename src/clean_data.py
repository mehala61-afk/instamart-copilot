from pathlib import Path
import pandas as pd
#1. Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#2. Read raw data
customers = pd.read_csv(RAW_DIR / "customers.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")
products = pd.read_csv(RAW_DIR / "products.csv")

#3. CUSTOMER CLEANING
print("\n--- CUSTOMER CLEANING ---")
customers["city"] = (
    customers["city"].astype("string").str.strip().str.title()
)
customers["gender"] = (
    customers["gender"].astype("string").str.strip().str.lower()
)
gender_mapping = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female"
}
customers["gender"] = (
    customers["gender"].map(gender_mapping).fillna("unknown")
)
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)
invalid_age = (customers["age"] < 10) | (customers["age"] > 100)
print("Invalid ages changed to missing:", invalid_age.sum())
customers.loc[invalid_age, "age"] = pd.NA
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)
customers["age"] = customers["age"].fillna(
    customers["age"].median()
)
customers["age"] = customers["age"].round().astype("Int64")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)
customers = customers.drop_duplicates().copy()
print("Missing ages after cleaning:", customers["age"].isna().sum())

#4. PRODUCT CLEANING
print("\n--- PRODUCT CLEANING ---")
products["product_name"] = (
    products["product_name"].astype("string").str.strip().str.title()
)
products["category"] = (
    products["category"].astype("string").str.strip().str.title()
)
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)
products = products.drop_duplicates().copy()

#5. ORDER CLEANING
print("\n--- ORDER CLEANING ---")
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["status"] = (
    orders["status"].astype("string").str.strip().str.lower()
)
for column in [
    "quantity",
    "discount_pct",
    "order_amount",
    "delivery_minutes"
]:
    orders[column] = pd.to_numeric(
        orders[column], errors="coerce"
    )

#Remove exact duplicate rows
before = len(orders)
orders = orders.drop_duplicates().copy()
print("Exact duplicate rows removed:", before - len(orders))

#Keep latest record for each order_id
before = len(orders)
orders = (
    orders.sort_values("order_ts", kind="stable")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)
print("Older order records removed:", before - len(orders))

#Merge product information once
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

#Repair missing order amounts
missing_amount = orders["order_amount"].isna()
orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
).round(2)
print("Order amounts repaired:", missing_amount.sum())

#Negative delivery times become missing
negative_delivery = orders["delivery_minutes"] < 0
orders.loc[negative_delivery, "delivery_minutes"] = pd.NA
print(
    "Negative delivery times changed to missing:",
    negative_delivery.sum()
)

#Fill missing delivery time for completed orders
completed_mask = orders["status"].eq("completed")
category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .median()
)
missing_completed = (
    completed_mask & orders["delivery_minutes"].isna()
)
orders.loc[missing_completed, "delivery_minutes"] = (
    orders.loc[missing_completed, "category"]
    .map(category_medians)
)

#Remove temporary product columns
orders = orders.drop(
    columns=["price_inr", "category"]
)

#6. SAVE CLEAN DATA
customers.to_csv(
    OUTPUT_DIR / "customers.csv", index=False
)
orders.to_csv(
    OUTPUT_DIR / "orders.csv", index=False
)
products.to_csv(
    OUTPUT_DIR / "products.csv", index=False
)

#7. FINAL VALIDATION
print("\n========== CLEANING SUMMARY ==========")
print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Products:", products.shape)
print("\nMissing customer ages:", customers["age"].isna().sum())
print(
    "Duplicate order IDs:",
    orders["order_id"].duplicated().sum()
)
print(
    "Missing order amounts:",
    orders["order_amount"].isna().sum()
)
print(
    "Missing completed delivery times:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)
print(
    "Missing cancelled delivery times:",
    orders.loc[
        orders["status"].eq("cancelled"),
        "delivery_minutes"
    ].isna().sum()
)
print("\nOrders by status:")
print(orders["status"].value_counts())
print("\nCompleted order amount total:")
print(
    round(
        orders.loc[completed_mask, "order_amount"].sum(),
        2
    )
)
print("\nClean CSV files saved in:")
print(OUTPUT_DIR)
print("\nCleaning completed.")