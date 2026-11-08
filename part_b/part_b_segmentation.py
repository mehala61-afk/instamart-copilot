from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load RFM Data
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "clean" / "customer_rfm.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rfm = pd.read_csv(INPUT_PATH)
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# 2. Scale RFM Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Means Evaluation (K=2 to 8)
print("\nK-Means Evaluation")
evaluation_results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )
    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(
        X_scaled,
        labels
    )
    evaluation_results.append({
        "K": k,
        "Inertia": round(inertia, 2),
        "Silhouette": round(silhouette, 3)
    })

evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df.to_string(index=False))

# 4. Final K-Means Model (K=4)
final_kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)
rfm["segment_id"] = final_kmeans.fit_predict(X_scaled)

# 5. Segment Profile
summary = rfm.groupby("segment_id").agg(
    customers=("customer_id", "count"),
    recency=("recency", "mean"),
    frequency=("frequency", "mean"),
    monetary=("monetary", "mean")
).round(2)

print("\nFinal Segment Profile")
print(summary)

# 6. Segment Names
segment_names = {
    0: "Regular Low-Spend",
    1: "Premium",
    2: "Frequent Active",
    3: "At-Risk Inactive"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Premium segment flag
rfm["is_premium"] = (
    rfm["segment_id"] == 1
).astype(int)

# 7. PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("\nTotal Explained Variance:")
print(pca.explained_variance_ratio_.sum())

# 8. Save Customer Segments
output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

output_path = OUTPUT_DIR / "customer_segments.csv"

output.to_csv(
    output_path,
    index=False
)

print("\nCustomer segments saved successfully!")
print("File:", output_path)
print(output.head())