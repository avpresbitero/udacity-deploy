from fastapi.testclient import TestClient
import pandas as pd

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_get_status_code():
    r = client.get("/")
    assert r.status_code == 200


def test_get_status_message():
    r = client.get("/")
    data = r.json()
    assert data['message'] == 'Hey there!'


def test_post_status_code():
    data = pd.DataFrame([{"age": 18,
                          "workclass": 'state-gov',
                          "fnlgt": 201490,
                          "education": 'bachelors',
                          "education-num": 9,
                          "marital-status": 'never-married',
                          "occupation": 'handlers-cleaners',
                          "relationship": 'own-child',
                          "race": 'white',
                          "sex": 'female',
                          "capital-gain": 0,
                          "capital-loss": 0,
                          "hours-per-week": 20}])

    response = client.post("/predict", json=data)
    assert response.status_code == 200


def test_post_predict_less():

    data = pd.DataFrame([{"age": 18,
                                "workclass": 'state-gov',
                                "fnlgt": 201490,
                                "education": 'bachelors',
                                "education-num": 9,
                                "marital-status": 'never-married',
                                "occupation": 'handlers-cleaners',
                                "relationship": 'own-child',
                                "race": 'white',
                                "sex": 'female',
                                "capital-gain": 0,
                                "capital-loss": 0,
                                "hours-per-week": 20}])

    response = client.post("/predict", json=data)
    prediction = response.json()
    assert prediction['prediction'] == "Salary <= 50k"


def test_post_predict_greater():
    data = pd.DataFrame([{"age": 40,
                          "workclass": 'private',
                          "fnlgt": 201490,
                          "education": 'doctorate',
                          "education-num": 16,
                          "marital-status": 'married-civ-spouse',
                          "occupation": 'prof-specialty',
                          "relationship": 'husband',
                          "race": 'white',
                          "sex": 'male',
                          "capital-gain": 0,
                          "capital-loss": 0,
                          "hours-per-week": 60}])

    response = client.post("/predict", json=data)
    prediction = response.json()
    assert prediction['prediction'] == "Salary > 50k"