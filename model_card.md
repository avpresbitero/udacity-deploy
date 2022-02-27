# Model Card
For additional information on model cards see the [Model Card paper](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
The model used was scikit-learn's [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Hyper parameter tuning was carried out using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

## Intended Use
The model is _only_ intended for the demonstration of deploying and serving a machine learning model.

## Training Data
80% of the data was used for training.

## Evaluation Data
20% of the data was used for testing/evaluation.

## Metrics
The metrics used to evaluate the model are:
  - Fbeta score: 0.69
  - Precision 0.81
  - Recall  0.60
  
## Ethical Considerations
The dataset is outdated (1994). It does not provide any 
insight into current income with respect to demographics. 

## Caveats and Recommendations
The current model requires a specific format of the data. In order for the 
prediction to run smoothly, a sample input data is shown below:

```data = {
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
```

One future step is to develop a front-end where a user can input the fields. 