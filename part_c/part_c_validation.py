import pandas as pd

tickets = pd.read_csv("data/raw/tickets.csv")

print("Shape:", tickets.shape)

print("\nColumns:")
print(tickets.columns.tolist())

print("\nEmpty tickets:")
print(tickets["text"].isna().sum() + (tickets["text"].fillna("").str.strip() == "").sum())

print("\nCategory counts:")
print(tickets["true_category"].value_counts())

print("\nSample tickets:")
print(tickets.head())