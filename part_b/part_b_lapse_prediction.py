from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Paths

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load Data

customers = pd.read_csv(DATA_DIR / "customers.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")

# 3. Separate Pre-Cutoff Orders

pre_orders = orders[
    orders["order_ts"] < CUTOFF
].copy()

completed = pre_orders[
    pre_orders["status"] == "completed"
].copy()

cancelled = pre_orders[
    pre_orders["status"] == "cancelled"
].copy()

# 4. Create Lapse Labels

label_orders = orders[
    (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < LABEL_END)
    & (orders["status"] == "completed")
]

cohort_ids = completed["customer_id"].unique()
active_ids = label_orders["customer_id"].unique()

labels = pd.DataFrame({
    "customer_id": cohort_ids
})

labels["lapsed"] = (
    ~labels["customer_id"].isin(active_ids)
).astype(int)

print("\nLapse Cohort Summary")
print("Total customers:", len(labels))
print("Lapsed customers:", labels["lapsed"].sum())

print(
    "Lapse percentage:",
    round(labels["lapsed"].mean() * 100, 2),
    "%"
)

# 5. RFM and Other Features

recency = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean")
).reset_index()

recency["recency"] = (
    CUTOFF - recency["last_order"]
).dt.days

recency = recency.drop(columns=["last_order"])

# 6. Cancellation Rate

final_orders = pre_orders[
    pre_orders["status"].isin(["completed", "cancelled"])
]

total_orders = final_orders.groupby(
    "customer_id"
)["order_id"].nunique()

cancel_counts = cancelled.groupby(
    "customer_id"
)["order_id"].nunique()

cancel_rate = (
    cancel_counts / total_orders
).fillna(0).rename("cancellation_rate")

# 7. Favourite Product Category

category_orders = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_category = (
    category_orders.groupby("customer_id")["category"]
    .agg(lambda x: x.value_counts().index[0])
    .rename("favourite_category")
)

# 8. Merge Features

features = recency.merge(
    cancel_rate,
    on="customer_id",
    how="left"
)

features = features.merge(
    favourite_category,
    on="customer_id",
    how="left"
)

features = features.merge(
    customers[
        [
            "customer_id",
            "city",
            "age",
            "gender",
            "signup_date"
        ]
    ],
    on="customer_id",
    how="left"
)

features["tenure_days"] = (
    CUTOFF - features["signup_date"]
).dt.days

features = features.drop(columns=["signup_date"])

# Add labels
dataset = features.merge(
    labels,
    on="customer_id",
    how="inner"
)

# 9. Check Dataset

print("\nDataset Shape:", dataset.shape)
print("\nMissing Values:")
print(dataset.isnull().sum())
print("\nClass Distribution:")
print(dataset["lapsed"].value_counts())
print("\nDataset Preview:")
print(dataset.head())

# 10. Train Test Split

X = dataset.drop(
    columns=["customer_id", "lapsed"]
)

y = dataset["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
print("\nTrain Class Distribution:")
print(y_train.value_counts())
print("\nTest Class Distribution:")
print(y_test.value_counts())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# 11. Identify Feature Types

numeric_features = [
    "frequency",
    "monetary",
    "avg_discount",
    "avg_delivery",
    "recency",
    "cancellation_rate",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

# 12. Preprocessing

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 13. Baseline Model

baseline_predictions = [0] * len(y_test)

print("\nBASELINE MODEL")
print(
    "Accuracy:",
    accuracy_score(y_test, baseline_predictions)
)

print(
    "Lapsed F1:",
    f1_score(
        y_test,
        baseline_predictions,
        zero_division=0
    )
)

# 14. Logistic Regression GridSearch

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

logistic_grid = GridSearchCV(
    logistic_pipeline,
    logistic_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

logistic_grid.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION")
print("Best Parameters:", logistic_grid.best_params_)
print("Best CV F1:", logistic_grid.best_score_)

# 15. Random Forest GridSearch

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_params = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nRANDOM FOREST")
print("Best Parameters:", rf_grid.best_params_)
print("Best CV F1:", rf_grid.best_score_)

# 16. Test Evaluation

def evaluate_model(name, model):

    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(
        "Precision:",
        precision_score(y_test, predictions, zero_division=0)
    )

    print(
        "Recall:",
        recall_score(y_test, predictions, zero_division=0)
    )

    print(
        "F1 Score:",
        f1_score(y_test, predictions, zero_division=0)
    )

evaluate_model(
    "Logistic Regression Test Results",
    logistic_grid.best_estimator_
)

evaluate_model(
    "Random Forest Test Results",
    rf_grid.best_estimator_
)

# 17. Scaler Mean Comparison

best_logistic = logistic_grid.best_estimator_

scaler_fitted = (
    best_logistic
    .named_steps["preprocessor"]
    .named_transformers_["num"]
    .named_steps["scaler"]
)

frequency_index = numeric_features.index("frequency")

print("\nFREQUENCY MEAN COMPARISON")

print(
    "Full cohort frequency mean:",
    dataset["frequency"].mean()
)


# 18. Oversampling Training Data

train_data = X_train.copy()
train_data["lapsed"] = y_train

majority_count = train_data["lapsed"].value_counts().max()
minority_count = train_data["lapsed"].value_counts().min()

minority_data = train_data[
    train_data["lapsed"] == 1
]

extra_samples = minority_data.sample(
    n=majority_count - minority_count,
    replace=True,
    random_state=42
)

balanced_train = pd.concat(
    [train_data, extra_samples],
    ignore_index=True
)

X_train_balanced = balanced_train.drop(
    columns=["lapsed"]
)

y_train_balanced = balanced_train["lapsed"]

print("\nBalanced Training Distribution:")
print(y_train_balanced.value_counts())

# 19. Refit Logistic Regression

best_C = logistic_grid.best_params_["classifier__C"]

balanced_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        C=best_C,
        max_iter=1000
    ))
])

balanced_lr.fit(
    X_train_balanced,
    y_train_balanced
)

print("\nOVERSAMPLED LOGISTIC REGRESSION")

evaluate_model(
    "Balanced Logistic Regression",
    balanced_lr
)

## 20. Random Forest Feature Importance

best_rf = rf_grid.best_estimator_

feature_names = (
    best_rf
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    best_rf
    .named_steps["classifier"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 5 RANDOM FOREST FEATURES")
print(importance_df.head(5).to_string(index=False))

# 21. Logistic Regression Coefficients

lr_model = logistic_grid.best_estimator_

lr_feature_names = (
    lr_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

lr_coefficients = (
    lr_model
    .named_steps["classifier"]
    .coef_[0]
)

coefficient_df = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_coefficients
})

print("\nTOP 3 POSITIVE COEFFICIENTS")
print(
    coefficient_df.sort_values(
        "coefficient",
        ascending=False
    ).head(3).to_string(index=False)
)

print("\nTOP 3 NEGATIVE COEFFICIENTS")
print(
    coefficient_df.sort_values(
        "coefficient",
        ascending=True
    ).head(3).to_string(index=False)
)

# 22. Save Random Forest Model

import joblib

MODEL_PATH = OUTPUT_DIR / "lapse_model.joblib"
joblib.dump(best_rf, MODEL_PATH)
print("\nModel saved:", MODEL_PATH)

# 23. Reload Model Validation

loaded_model = joblib.load(MODEL_PATH)

original_predictions = best_rf.predict(X_test)

loaded_predictions = loaded_model.predict(X_test)

print(
    "\nReload Prediction Match:",
    (original_predictions == loaded_predictions).all()
)

# 24. Premium Retention List

segments = pd.read_csv(
    OUTPUT_DIR / "customer_segments.csv"
)

premium_ids = set(
    segments.loc[
        segments["is_premium"] == 1,
        "customer_id"
    ]
)

retention = pd.DataFrame({
    "customer_id": dataset.loc[
        X_test.index,
        "customer_id"
    ].values,

    "lapse_probability": loaded_model.predict_proba(
        X_test
    )[:, 1],

    "actually_lapsed": y_test.values
})

retention = retention[
    retention["customer_id"].isin(premium_ids)
]

retention = retention.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention["lapse_probability"] = (
    retention["lapse_probability"].round(4)
)

retention.to_csv(
    OUTPUT_DIR / "retention_list.csv",
    index=False
)

print("\nTOP 10 PREMIUM RETENTION CUSTOMERS")
print(retention.to_string(index=False))

print(
    "\nActually lapsed in Top 10:",
    retention["actually_lapsed"].sum()
)

print("\nPart B Lapse Prediction Completed!")


# README - Feature Importance

# 1. Random Forest - Top 5 Features

print("\nRANDOM FOREST - TOP 5 FEATURES")
print(
    importance_df.head(5).to_string(index=False)
)

# 2. Logistic Regression - Top 3 Positive Coefficients

print("\nLOGISTIC REGRESSION - POSITIVE COEFFICIENTS")
print(
    coefficient_df.sort_values(
        by="coefficient",
        ascending=False
    ).head(3).to_string(index=False)
)

# 3. Logistic Regression - Top 3 Negative Coefficients

print("\nLOGISTIC REGRESSION - NEGATIVE COEFFICIENTS")
print(
    coefficient_df.sort_values(
        by="coefficient",
        ascending=True
    ).head(3).to_string(index=False)
)