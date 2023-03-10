import json
import requests


data = {"age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        }

response = requests.post('https://census-app.herokuapp.com/inference',
                          data=json.dumps(data))

print('Status:', response.status_code)
print('Predicted value:', response.json())