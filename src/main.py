from fastapi import FastAPI
from pydantic import BaseModel, Field

import data_processing.ml.model_data as md

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours_per_week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True

app = FastAPI()

@app.get('/')
async def say_hello():
    return({'greeting': 'Hi, what about making an inference?'})

@app.post('/inference')
async def make_inference(data: InputData):
    
    prediction = md.make_prediction_from_api(data)
    return prediction
