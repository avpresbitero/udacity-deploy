# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data
from starter.ml.model import inference

import pickle
import pandas as pd

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

class Input(BaseModel):
    age : int = 18
    workclass : str = 'state-gov'
    fnlgt: int = 201490
    education: str = 'bachelors'
    education_num: int = 9
    marital_status: str = 'never-married'
    occupation: str = 'handlers-cleaners'
    relationship: str = 'own-child'
    race: str = 'white'
    sex: str = 'female'
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 20


# class Input(BaseModel):
#     age : int = 40
#     workclass : str = 'private'
#     fnlgt: int = 201490
#     education: str = 'doctorate'
#     education_num: int = 16
#     marital_status: str = 'married-civ-spouse'
#     occupation: str = 'prof-specialty'
#     relationship: str = 'husband'
#     race: str = 'white'
#     sex: str = 'male'
#     capital_gain: int = 0
#     capital_loss: int = 0
#     hours_per_week: int = 60


class Output(BaseModel):
    prediction: str

app = FastAPI()

with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("model/lb.pkl", "rb") as file:
    lb = pickle.load(file)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex"
]


@app.get("/")
async def get():
    return {"message": "Hey there!"}


@app.post("/predict")
async def post(data: Input):
    input_data = pd.DataFrame([{"age" : data.age,
                        "workclass" : data.workclass,
                        "fnlgt" : data.fnlgt,
                        "education" : data.education,
                        "education-num" : data.education_num,
                        "marital-status" : data.marital_status,
                        "occupation" : data.occupation,
                        "relationship" : data.relationship,
                        "race" : data.race,
                        "sex" : data.sex,
                        "capital-gain" : data.capital_gain,
                        "capital-loss" : data.capital_loss,
                        "hours-per-week" : data.hours_per_week}])

    X, _, _, _ = process_data(input_data,
                           categorical_features=cat_features,
                           training=False,
                              encoder=encoder,
                              lb=lb)

    prediction = inference(model, X)
    print (prediction)

    if prediction[0] == 1:
        return {"prediction": "Salary > 50k"}
    else:
        return {"prediction": "Salary <= 50k"}