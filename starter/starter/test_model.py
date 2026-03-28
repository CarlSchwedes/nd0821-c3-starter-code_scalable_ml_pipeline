import pytest
import pandas as pd
import numpy as np
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="session")
def raw_data():
    """Load and clean census dataset once per test session."""
    data = pd.read_csv("./data/census.csv")

    # Strip spaces from columns + cell values
    data.columns = data.columns.str.strip()
    data = data.map(
        lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
    )
    # fnlgt and native-country often degrade model performance
    data = data.drop(columns=["fnlgt", "native-country"])
    return data


@pytest.fixture(scope="session")
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex"
    ]


@pytest.fixture(scope="session")
def train_test(raw_data):
    """Return train/test split once."""
    return train_test_split(raw_data, test_size=0.20, random_state=42)


@pytest.fixture(scope="session")
def processed_data(train_test, cat_features):
    """Return processed X_train, y_train, X_test, y_test + encoder & label binarizer."""
    train, test = train_test

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    return X_train, y_train, X_test, y_test, encoder, lb


@pytest.fixture(scope="session")
def trained_model(processed_data):
    """Train and return a RandomForest model."""
    X_train, y_train, *_ = processed_data
    model = train_model(X_train, y_train)
    return model


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_process_data_shapes(raw_data, cat_features):
    X, y, enc, lb = process_data(
        raw_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == len(raw_data)
    assert len(y) == len(raw_data)
    assert enc is not None
    assert lb is not None


def test_no_nans_after_processing(processed_data):
    X_train, y_train, X_test, y_test, _, _ = processed_data

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()


def test_train_model(trained_model):
    assert isinstance(trained_model, RandomForestClassifier)


def test_inference(trained_model, processed_data):
    _, _, X_test, y_test, *_ = processed_data

    preds = inference(trained_model, X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y_test)


def test_model_deterministic(trained_model, processed_data):
    _, _, X_test, _, _, _ = processed_data

    pred1 = inference(trained_model, X_test)
    pred2 = inference(trained_model, X_test)

    assert np.array_equal(pred1, pred2)


def test_inference_output_values(trained_model, processed_data):
    _, _, X_test, _, _, _ = processed_data
    preds = inference(trained_model, X_test)

    assert set(np.unique(preds)).issubset({0, 1})


def test_metrics_edge_case():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 1])  # extreme edge-case

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_compute_model_metrics(trained_model, processed_data):
    _, _, X_test, y_test, *_ = processed_data

    preds = inference(trained_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
