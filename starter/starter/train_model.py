
import re
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    train_model,
    save_model,
    compute_model_metrics,
    compute_slice_metrics,
    inference,
)

# ---------------------------------------------------------
# 1. Load and Clean Data
# ---------------------------------------------------------
data = pd.read_csv("./data/census.csv")

# Strip whitespace from column names and values
data.columns = data.columns.str.strip()
data = data.map(
    lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
)

# ---------------------------------------------------------
# 2. OPTIONAL Feature Removal (Improves Performance)
# ---------------------------------------------------------
# fnlgt and native-country often degrade model performance
data = data.drop(columns=["fnlgt", "native-country"])

# ---------------------------------------------------------
# 3. Train/Test Split
# ---------------------------------------------------------
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex"
]

# ---------------------------------------------------------
# 4. Process Training Data
# ---------------------------------------------------------
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# ---------------------------------------------------------
# 5. Process Test Data
# ---------------------------------------------------------
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# ---------------------------------------------------------
# 6. Hyperparameter‑Tuned RandomForest Model
# ---------------------------------------------------------
print("\nRunning hyperparameter search...")

best_model = train_model(X_train, y_train)
print(f"\nBest model: {best_model}")

# ---------------------------------------------------------
# 7. Evaluate Model
# ---------------------------------------------------------
preds = inference(best_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("\nFINAL MODEL PERFORMANCE:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F-Beta:    {fbeta:.4f}\n")

# ---------------------------------------------------------
# 8. Slice Metrics
# ---------------------------------------------------------
print("Computing slice metrics...")

slice_metrics = compute_slice_metrics(
    model=best_model,
    X=test,
    label="salary",
    encoder=encoder,
    lb=lb,
    categorical_features=cat_features
)

with open("slice_output.txt", "w") as f:
    for feature, categories in slice_metrics.items():
        f.write(f"=== Feature: {feature} ===\n")
        for category, metrics in categories.items():
            f.write(f"  Slice: {category}\n")
            f.write(f"    Precision: {metrics['precision']:.4f}\n")
            f.write(f"    Recall:    {metrics['recall']:.4f}\n")
            f.write(f"    F-Beta:    {metrics['fbeta']:.4f}\n")
            f.write(f"    N:         {metrics['n_samples']}\n\n")

print("Slice metrics saved to slice_output.txt")

# ---------------------------------------------------------
# 9. Save All Artifacts
# ---------------------------------------------------------
save_model(best_model, "./model/rf_clss_model.pkl")
save_model(encoder, "./model/encoder.pkl")
save_model(lb, "./model/lbinarizer.pkl")

print("Saved model artifacts successfully.\n")
