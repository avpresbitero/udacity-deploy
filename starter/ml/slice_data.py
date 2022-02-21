import pandas as pd
from model import compute_model_metrics, inference


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
        slice_labels = X.loc[df[feature] == i]
        X_slice = X[slice_labels.index.values, :]
        y_slice = y[slice.index.values]

        slice_predictions = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y, slice_predictions)

        metric['feature'].append(feature)
        metric['value'].append(i)
        metric['precision'].append(precision)
        metric['recall'].append(recall)
        metric['fbeta'].append(fbeta)

    metric_df = pd.DataFrame.from_dic(metric)
    metric_df.to_csv(f'/model/{feature}_slice_data.csv')






