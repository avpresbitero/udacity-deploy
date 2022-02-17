import pandas as pd

from starter.ml.model import train_model
from starter.ml.data import process_data

from sklearn.ensemble import RandomForestClassifier


def test_train_model():
    path = '../data/clean_census.csv'
    categorical_features = ['education', 'marital-status', 'relationship',
                            'race', 'sex', 'occupation', 'workclass', 'native-country']
    target = 'salary'
    data = pd.read_csv(path)
    X, y, encoder, lb = process_data(X=data,
                                      categorical_features=categorical_features,
                                      label=target)

    model = train_model(X, y)

    assert type(model) == type(RandomForestClassifier())



