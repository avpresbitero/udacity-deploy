import pandas as pd
from predict_income.ml.model import compute_model_metrics, inference


def data_slice_metric(df, X, y, model, feature):
    # Create dictionary
    metric = {}
    metric['feature'] = []
    metric['value'] = []
    metric['precision'] = []
    metric['recall'] = []
    metric['fbeta'] = []

    # Make sure that indices are in order
    df = df.reset_index(drop=True)
    for i in df[feature].unique():
        # Access a group of columns and rows by labels
        # Outputs a dataframe
        slice_labels = df[df[feature] == i]
        X_slice = X[slice_labels.index.values, :]
        y_slice = y[slice_labels.index.values]

        slice_predictions = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, slice_predictions)

        metric['feature'].append(feature)
        metric['value'].append(i)
        metric['precision'].append(precision)
        metric['recall'].append(recall)
        metric['fbeta'].append(fbeta)

        print(f"For value {i}, precision = {precision}, recall = {recall}, fbeta = {fbeta}")

    metric_df = pd.DataFrame(metric)
    metric_df.to_csv(f'../data/{feature}_slice_data.csv')






