from main import app
from fastapi.testclient import TestClient

import os

# Import our app from main.py.

def test_get_status_code():
    # Instantiate the testing client with our app.
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200


def test_get_status_message():
    with TestClient(app) as client:
        response = client.get("/")
    data = response.json()
    assert data['message'] == 'Hey there!'


def test_post_status_code():
    data = {"age": 18,
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
                          "hours-per-week": 20}

    with TestClient(app) as client:
        response = client.post("/predict", json=data)
    assert response.status_code == 200


def test_post_predict_greater():
    data = {"age": 50,
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
          "hours-per-week": 60}

    with TestClient(app) as client:
        response = client.post("/predict", json=data)
    prediction = response.json()
    assert prediction['prediction'] == "Salary > 50k"


def test_post_predict_less():
    data = {"age": 18,
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
                                "hours-per-week": 20}

    with TestClient(app) as client:
        response = client.post("/predict", json=data)
    prediction = response.json()
    assert prediction['prediction'] == "Salary <= 50k"
