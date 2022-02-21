# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Alva Presbitero
- February 2, 2022
- Version 1
- A RandomForestClassifier was used for training the data
- I used GridSearch to get the optimized parameters
– Please sent comments to avpresbitero@gmail.com


## Intended Use
- This project is intended to show an example of deploying a Scalable ML Pipeline in Production
- Predict whether income exceeds $50K/yr based on census data


## Factors
- This data is quite old as it was collected in 1994. The author recommends that an updated version of the dataset be used instead.

## Metrics
We have used three metrics in this project; fbeta_score, precision_score and recall_score.

## Evaluation Data
We have used 20% of the original dataset for evaluation purposes of the model.

## Training Data
The author used 80% of the original dataset for the training purposes of the model.

## Quantitative Analyses
Model performance is measured based on three metrics: fbeta score, precision score and recall score.
