import re
import pandas as pd
import numpy as np
# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, load_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv('./data/census.csv')
# Remove any containing whitespaces from dataset.
data.columns = data.columns.str.strip()
data = data.applymap(
    lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
)

model = load_model('./model/rf_clss_model.pkl')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


def test_train_model():
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_inference():
    preds = inference(model, X_test)
    assert len(preds) == len(y_test)
    assert isinstance(preds, np.ndarray)

def test_compute_model_metrics():
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
