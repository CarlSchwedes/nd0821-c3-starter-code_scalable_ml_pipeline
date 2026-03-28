from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference, load_model


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def alias_cleaner(string: str) -> str:
    """
    Pydantic model fields use underscores, but the dataset uses hyphens
    (e.g. marital-status, education-num).
    This function maps endpoint_input.education_num → "education-num" etc.
    """
    return string.replace('_', '-' )


# -------------------------------------------------------------
# Input Schema for FastAPI
# -------------------------------------------------------------
class CensusData(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    model_config = {
        "populate_by_name": True,
        "alias_generator": alias_cleaner,
        "extra": "ignore"
    }


# -------------------------------------------------------------
# App initialization
# -------------------------------------------------------------
app = FastAPI(
    title="Scalable ML Pipeline - Inference API",
    description="Random Forest income inference API",
    version="1.0.0"
)

model = None
encoder = None
lb = None

@app.on_event("startup")
def load_artifacts():
    global model, encoder, lb
    model = load_model("./model/rf_clss_model.pkl")
    encoder = load_model("./model/encoder.pkl")
    lb = load_model("./model/lbinarizer.pkl")


# Expected categorical features (MUST match training order)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex"
]


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the income prediction API!"}


@app.post("/inference")
async def run_inference(record: CensusData):
    # Convert Pydantic input to a DataFrame
    record_dict = record.model_dump(by_alias=True)
    input_df = pd.DataFrame([record_dict])

    # Preprocess data to match model expectations
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run inference
    pred = inference(model, X)[0]

    # Convert numeric label back to its string class
    prediction = lb.inverse_transform(np.array([[pred]]))[0]

    return {"prediction": prediction}
