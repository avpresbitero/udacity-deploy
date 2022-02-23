import pandas as pd
import os
import pytest
import numpy as np
import pathlib as pl

from predict_income.ml.model import inference, compute_model_metrics
from predict_income.ml.utils import CAT_FEAT
from predict_income.ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

@pytest.fixture(scope="module")
def data():
    path = os.path.join(CURRENT_DIRECTORY, '../data/clean_census.csv')
    data = pd.read_csv(path).drop("native-country", axis=1)

    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEAT,
        label="salary",
        training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=CAT_FEAT,
        label="salary",
        training=True
    )

    return [X_train, y_train, X_test, y_test]


@pytest.fixture(scope="module")
def model(data):
    X_train, y_train, X_test, y_test = data
    model = DummyClassifier()
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def predictions(model, data):
    X_train, y_train, X_test, y_test = data
    predictions = inference(model, X_test)
    return predictions


def test_inference(predictions, data):
    X_train, y_train, X_test, y_test = data
    assert len(predictions) == len(y_test)


def test_compute_model_metrics(predictions, data):
    X_train, y_train, X_test, y_test = data
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    assert precision != np.nan
    assert recall != np.nan
    assert fbeta != np.nan


def test_trained_model_artifacts():
    MODEL = os.path.join(CURRENT_DIRECTORY, '../model/model.pkl')
    ENCODER = os.path.join(CURRENT_DIRECTORY, '../model/encoder.pkl')
    LB = os.path.join(CURRENT_DIRECTORY, '../model/lb.pkl')

    assert pl.Path(MODEL).is_file()
    assert pl.Path(ENCODER).is_file()
    assert pl.Path(LB).is_file()