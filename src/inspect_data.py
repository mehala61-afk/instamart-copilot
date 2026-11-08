import pandas as pd
from pathlib import Path

data_folder = Path("data/raw")

files = ["customers.csv", "orders.csv", "products.csv"]

for file in files:
    file_path = data_folder / file

    df = pd.read_csv(file_path)

    print("\n" + "=" * 50)
    print("File:", file)
    print("Rows and columns:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())