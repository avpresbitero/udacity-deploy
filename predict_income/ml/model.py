#!/usr/bin/env python3

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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

    # Create the parameter grid based on the
    # results of random search
    param_grid = {
        "bootstrap": [True],
        "max_depth": [80, 90, 100, 110],
        "max_features": [2, 3],
        "min_samples_leaf": [3, 4, 5],
        "min_samples_split": [8, 10, 12],
        "n_estimators": [100, 200, 300, 1000],
    }

    # Create a base model
    gbc = RandomForestClassifier()

    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=gbc,
        param_grid=param_grid,
        cv=10, n_jobs=-1,
        verbose=2
    )
    # Fit the data
    grid_search.fit(X_train, y_train)

    # Return best model
    return grid_search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using
    precision, recall, and F1.

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
    """Run model inferences and return the predictions.

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
