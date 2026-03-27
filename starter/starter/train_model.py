import re
import pandas as pd
# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, save_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv('./data/census.csv')
# Remove any containing whitespaces from dataset.
data.columns = data.columns.str.strip()
data = data.applymap(
    lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
)

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

# Train and save a model.
model = train_model(X_train, y_train)

pred = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, pred)
print(f"Model: RandomForestClassifier:\nprecision:{precision}, recall:{recall}, fbeta: {fbeta}")

save_model(model, './model/rf_clss_model.pkl')
save_model(encoder, './model/encoder.pkl')
save_model(lb, './model/lbinarizer.pkl')