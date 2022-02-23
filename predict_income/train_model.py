# Script to train machine learning model.
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from predict_income.ml.data import process_data
from predict_income.ml.model import train_model, \
    inference, compute_model_metrics
from predict_income.ml.slice_data import data_slice_metric
from predict_income.ml.utils import CAT_FEAT, dump_file, load_file

CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

# Add code to load in the data.
path = os.path.join(CURRENT_DIRECTORY, "../data/clean_census.csv")
data = pd.read_csv(path).drop("native-country", axis=1)

# set training boolean
to_train = False

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=CAT_FEAT,
    label="salary",
    training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=CAT_FEAT,
    label="salary",
    training=True
)


if to_train:
    # Train model
    model = train_model(X_train, y_train)

    # Save model artifacts
    dump_file(PATH="../model/model.pkl", MODEL=model)
    dump_file(PATH="../model/encoder.pkl", MODEL=encoder)
    dump_file(PATH="../model/lb.pkl", MODEL=lb)
else:
    # Load model
    model = load_file(PATH="../model/model.pkl")

data_slice_metric(train, X_train, y_train, model, feature="race")

predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)

print(
    f"Overall metric: precision = {precision}, recall = {recall}, fbeta = {fbeta}")