import json

import requests

data = {
    "age": 50,
    "workclass": "private",
    "fnlgt": 201490,
    "education": "doctorate",
    "education-num": 16,
    "marital-status": "married-civ-spouse",
    "occupation": "prof-specialty",
    "relationship": "husband",
    "race": "white",
    "sex": "male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 60,
}


if __name__ == "__main__":
    response = requests.post(
        "https://udacity-deploy-app.herokuapp.com/predict", data=json.dumps(data)
    )
    print(f"Status Code: {response.status_code}")
    print(response.json())
