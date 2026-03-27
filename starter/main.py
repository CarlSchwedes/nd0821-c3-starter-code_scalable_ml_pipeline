from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference, load_model


def clean_hyphens(string: str) -> str:
    return string.replace('_', '-')


class CensusData(BaseModel):
    age: int = 39
    workclass: str = 'Private'
    fnlgt: int = 123456
    education: str = 'Bachelor'
    education_num: int = 13
    marital_status: str = 'Never-married'
    occupation: str = 'Tech-support'
    relationship: str = 'Not-in-family'
    race: str = 'White'
    sex: str = 'Male'
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = 'United-States'

    class Config:
        clean = clean_hyphens


app = FastAPI(title="Scalable ML Pipeline - Inference API",
              description="API that runs random-forest model inference on sample dataset",
              version="1.0.0")

model = load_model('./model/rf_clss_model.pkl')
encoder = load_model('./model/encoder.pkl')
lb = load_model('./model/lbinarizer.pkl')

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


@app.get("/")
async def greet_user():
    return {
        "data": "Hello, this is a scalable ML pipeline project..."
    }


@app.post("/inference")
async def model_predict(endpoint_input: CensusData):
    endpoint_input_dict = endpoint_input.dict(by_alias=True)
    model_input = pd.DataFrame([endpoint_input_dict])

    processed_model_input, _, _, _ = process_data(
        model_input, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )

    inference_list = list(inference(model, processed_model_input))

    result = {}
    for i in range(len(inference_list)):
        if inference_list[i] == 0:
            result[i] = '<=50k'
        else:
            result[i] = '>50k'

    return {"result": result}


if __name__ == '__main__':
    pass
