#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

import data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    path = 'clean_census.csv'
    categorical_features = ['education', 'marital-status', 'relationship',
                            'race', 'sex', 'occupation', 'workclass', 'native-country']
    target = 'salary'
    data_ = pd.read_csv(path)
    X, y, encoder, lb = data.process_data(X=data_,
                                     categorical_features=categorical_features,
                                     label=target)

    model = train_model(X, y)
    predictions = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, predictions)
    print(precision)
