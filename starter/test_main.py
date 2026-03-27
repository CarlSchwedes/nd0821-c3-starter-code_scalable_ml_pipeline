import pandas as pd

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)

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

def test_get_greet_user():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "data": f"Test: GET/<greet_user>; status {r.status_code}"
    }

def test_post_model_predict_higher():
    input_data = pd.DataFrame(
        [[56, "Local-gov", 216851, "Bachelors", 13, "Married-civ-spouse",
            "Tech-support", "Husband", "White", "Male", 0, 0, 40, "United-States"]],
        columns=cat_features
    )
    r = client.post("/inference", json=input_data.iloc[0, :].to_dict())

    assert r.status_code == 200
    assert r.json() == {"result": {'0': ">50k"}}


def test_post_model_predict_lower():
    input_data = pd.DataFrame(
        [[38, "Private", 215646, "HS-grad", 9, "Divorced", "Handlers-cleaners",
            "Not-in-family", "White", "Male", 0, 0, 40, "United-States"]],
        columns=cat_features
    )
    r = client.post("/inference", json=input_data.iloc[0, :].to_dict())

    assert r.status_code == 200
    assert r.json() == {"result": {'0': "<=50k"}}