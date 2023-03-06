from fastapi.testclient import TestClient

import main as mn

client = TestClient(mn.app)

def test_api_locally_get_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'greeting': 'Hi, what about making an inference?'}


def test_api_prediction_no_input():
    input_data = {}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 404


def test_api_prediction():
    input_data = {
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
    response = client.post("/inference", json=input_data)
    print(response)
    #assert response.status_code == 200
    assert response.text in {'"<=50K"', '">50K"'}
    