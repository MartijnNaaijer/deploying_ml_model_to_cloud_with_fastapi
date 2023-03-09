import json
import requests

response = requests.post('https://census-app.herokuapp.com/inference/', 
                         auth=('martijn.naayer@upcmail.nl', 'Bg%67n(/>mN'))

print(response.status_code)
print(response.json())


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
response2 = requests.post('https://census-app.herokuapp.com/inference', 
                          auth=('martijn.naayer@upcmail.nl', 'Bg%67n(/>mN'),
                          data=json.dumps(data))

print(response2.status_code)
print(response2.json())