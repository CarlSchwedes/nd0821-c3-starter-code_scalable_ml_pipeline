import pandas as pd
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

full_feature_list = [
    "age",
    "workclass",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]


# -----------------------------
# GET ENDPOINT TEST
# -----------------------------
def test_get_greet_user():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the income prediction API!"
    }


# -----------------------------
# POST ENDPOINT TESTS
# -----------------------------
def test_post_inference_high_income():
    """Test high-income prediction (>50K)."""

    input_data = pd.DataFrame(
        [
            [
                56, "Local-gov", "Bachelors", 13,
                "Married-civ-spouse", "Tech-support", "Husband",
                "White", "Male", 0, 0, 40
            ]
        ],
        columns=full_feature_list
    )

    with TestClient(app) as client:
        response = client.post(
            "/inference",
            json=input_data.iloc[0, :].to_dict()
        )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json() == {"prediction": ">50K"}


def test_post_inference_low_income():
    """Test low-income prediction (<=50K)."""

    input_data = pd.DataFrame(
        [
            [
                38, "Private", "HS-grad", 9,
                "Divorced", "Handlers-cleaners", "Not-in-family",
                "White", "Male", 0, 0, 40
            ]
        ],
        columns=full_feature_list
    )

    with TestClient(app) as client:
        response = client.post(
            "/inference",
            json=input_data.iloc[0, :].to_dict()
        )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json() == {"prediction": "<=50K"}


# -----------------------------
# EDGE-CASE TESTS
# -----------------------------
def test_post_inference_missing_field():
    """Should fail because required fields are missing."""

    response = client.post("/inference", json={})
    assert response.status_code == 422  # validation error


def test_post_inference_invalid_type():
    """Should fail if type is incorrect (age must be int)."""

    invalid_payload = {"age": "not-an-integer"}
    response = client.post("/inference", json=invalid_payload)

    assert response.status_code == 422


def test_post_inference_alias_hyphen_support():
    """
    Tests that alias_generator=clean_hyphens is working.
    For example, API expects 'education-num' but model field is 'education_num'.
    """

    payload = {
        "age": 40,
        "workclass": "Private",
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40
    }

    with TestClient(app) as client:
        response = client.post("/inference", json=payload)
        
    assert response.status_code == 200
    assert "prediction" in response.json()
