# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

from predict_income.ml.data import process_data
from predict_income.ml.model import inference

import pickle
import pandas as pd
import os

CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -f") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
                "age": 50,
                "workclass": "private",
                "fnlgt": 201490,
                "education": "doctorate",
                "education_num": 16,
                "marital_status": "married-civ-spouse",
                "occupation": "prof-specialty",
                "relationship": "husband",
                "race": "white",
                "sex": "male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 60
            }
        }


class Output(BaseModel):
    prediction: str


app = FastAPI(
    title="Census Data Income Predictor",
    description="Income classification from census data.",
)


@app.on_event("startup")
def startup_event():
    """
    Additionally load model and encoder on startup for faster predictions
    """

    with open(CURRENT_DIRECTORY + "/model/encoder.pkl", "rb") as f:
        global ENCODER
        ENCODER = pickle.load(f)
    with open(CURRENT_DIRECTORY + "/model/model.pkl", "rb") as f:
        global MODEL
        MODEL = pickle.load(f)
    with open(CURRENT_DIRECTORY + "/model/lb.pkl", "rb") as f:
        global LB
        LB = pickle.load(f)


@app.get("/")
async def get():
    return {"message": "Hey there!"}


@app.post("/predict")
async def post(data: Input):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]

    input_data = pd.DataFrame(
        [
            {
                "age": data.age,
                "workclass": data.workclass,
                "fnlgt": data.fnlgt,
                "education": data.education,
                "education-num": data.education_num,
                "marital-status": data.marital_status,
                "occupation": data.occupation,
                "relationship": data.relationship,
                "race": data.race,
                "sex": data.sex,
                "capital-gain": data.capital_gain,
                "capital-loss": data.capital_loss,
                "hours-per-week": data.hours_per_week,
            }
        ]
    )

    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        training=False,
        encoder=ENCODER,
        lb=LB,
    )

    prediction = inference(MODEL, X)
    if prediction[0] == 1:
        return {"prediction": "Salary > 50k"}
    else:
        return {"prediction": "Salary <= 50k"}
