import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

import src.data_processing.ml.model_data as md

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class InputData(BaseModel):
    age: int
    workclass: str
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }

app = FastAPI()

@app.get('/')
async def say_hello():
    return({'greeting': 'Hi, what about making an inference?'})

@app.post('/inference')
async def make_inference(data: InputData):
    #prediction = md.make_inference_from_api(data, os.path.join(ROOT_DIR, 'deploying_ml_model_to_cloud_with_fastapi/src/model'), 'trained_model.pkl')
    #prediction = md.make_inference_from_api(data, os.path.join('./model'), 'trained_model.pkl')
    prediction = md.make_inference_from_api(data, os.path.join(ROOT_DIR, '/src/model'), 'trained_model.pkl')
    return prediction


if __name__ == '__main__':
    pass
