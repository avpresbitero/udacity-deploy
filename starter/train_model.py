# Script to train machine learning model.
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics
from starter.ml.slice_data import data_slice_metric

# Add code to load in the data.
path = '../data/clean_census.csv'
data = pd.read_csv(path).drop("native-country", axis=1)

# set training boolean
to_train = False

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    # "native-country",
]


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=True
)

if to_train == True:
    # Train and save a model.
    model = train_model(X_train, y_train)

    with open("../model/model.pkl", "wb") as output_file:
        pickle.dump(model, output_file)

    with open("../model/encoder.pkl", "wb") as output_file:
        pickle.dump(encoder, output_file)

    with open("../model/lb.pkl", "wb") as output_file:
        pickle.dump(lb, output_file)
else:
    with open("../model/model.pkl", "rb") as file:
        model = pickle.load(file)


data_slice_metric(train, X_train, y_train, model, feature="race")

predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
#
print(f"Overall metric: precision = {precision}, recall = {recall}, fbeta = {fbeta}")