"""
This file is used to test the api.
"""

from main import app
from fastapi.testclient import TestClient


def test_get_status_code():
    """
    Tests the status of the get call.
    Returns
    -------
    None
    """
    # Instantiate the testing client with our app.
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200


def test_get_status_message():
    """
    Tests the message in the json file via the get method.
    Returns
    -------
    None
    """
    with TestClient(app) as client:
        response = client.get("/")
    data = response.json()
    assert data["message"] == "Hey there!"


def test_post_status_code():
    """
    Tests the status code of the post method.
    Returns
    -------
    None
    """
    data = {
        "age": 18,
        "workclass": "state-gov",
        "fnlgt": 201490,
        "education": "bachelors",
        "education-num": 9,
        "marital-status": "never-married",
        "occupation": "handlers-cleaners",
        "relationship": "own-child",
        "race": "white",
        "sex": "female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=data)
    assert response.status_code == 200


def test_post_predict_greater():
    """
    Tests the prediction of the data via the post call. This method expects greater than 50k.
    Returns
    -------
    None
    """
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

    with TestClient(app) as client:
        response = client.post("/predict", json=data)
    prediction = response.json()
    assert prediction["prediction"] == "Salary > 50k"


def test_post_predict_less():
    """
    Tests the prediction of the data via the post call. This method expects less than 50k.
    Returns
    -------
    None
    """
    data = {
        "age": 18,
        "workclass": "state-gov",
        "fnlgt": 201490,
        "education": "bachelors",
        "education-num": 9,
        "marital-status": "never-married",
        "occupation": "handlers-cleaners",
        "relationship": "own-child",
        "race": "white",
        "sex": "female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=data)
    prediction = response.json()
    assert prediction["prediction"] == "Salary <= 50k"
