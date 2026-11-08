import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

completed = orders[orders["status"] == "completed"]

eda_data = (
    completed
    .merge(customers, on="customer_id")
    .merge(products, on="product_id")
)

pivot = pd.pivot_table(
    eda_data,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    fill_value=0
)

print(pivot)

# City-wise GroupBy Analysis

city_summary = (
    eda_data
    .groupby("city")
    .agg(
        total_revenue=("order_amount", "sum"),
        average_order=("order_amount", "mean"),
        order_count=("order_amount", "count")
    )
    .round(2)
)
print("\nCity-wise Summary:")
print(city_summary)

# Outlier Analysis: Completed Orders
amount = pd.to_numeric(completed["order_amount"], errors="coerce").dropna()

mean_value = amount.mean()
median_value = amount.median()
skew_value = amount.skew()

Q1 = amount.quantile(0.25)
Q3 = amount.quantile(0.75)

IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

below_outliers = (amount < lower_fence).sum()
above_outliers = (amount > upper_fence).sum()

print("\n--- OUTLIER ANALYSIS ---")
print("Mean:", round(mean_value, 2))
print("Median:", round(median_value, 2))
print("Skewness:", round(skew_value, 2))
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower Fence:", round(lower_fence, 2))
print("Upper Fence:", round(upper_fence, 2))
print("Below Outliers:", below_outliers)
print("Above Outliers:", above_outliers)

# Correlation Analysis: Completed Orders
correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_data = completed[correlation_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

correlation_matrix = correlation_data.corr()

print("\n--- CORRELATION MATRIX ---")
print(correlation_matrix)

print("\n--- ORDER AMOUNT CORRELATIONS ---")

order_amount_corr = (
    correlation_matrix["order_amount"]
    .drop("order_amount")
)

print(order_amount_corr)

strongest_pair = (
    correlation_matrix
    .where(
        ~pd.DataFrame(
            [[i == j for j in range(len(correlation_columns))]
             for i in range(len(correlation_columns))],
            index=correlation_columns,
            columns=correlation_columns
        )
    )
    .stack()
)

if not strongest_pair.empty:
    strongest_pair = strongest_pair[
        strongest_pair.index.map(lambda pair: pair[0] < pair[1])
    ]
    strongest_pair = strongest_pair.dropna()

    if not strongest_pair.empty:
        pair = strongest_pair.abs().idxmax()
        print(
            "\nLargest absolute correlation:",
            pair[0], "and", pair[1],
            "=", round(strongest_pair.loc[pair], 3)
        )