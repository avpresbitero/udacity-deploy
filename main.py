# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict_income.ml.data import process_data
from predict_income.ml.model import inference

import pickle
import pandas as pd
import os

CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

CAT_FEATURES = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -f") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class CensusData(BaseModel):
    age: int = Field(..., example=42)
    workclass: str = Field(..., example="private")
    fnlgt: int = Field(..., example=5178)
    education: str = Field(..., example="bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(
        ..., example="married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="exec-managerial")
    relationship: str = Field(..., example="husband")
    race: str = Field(..., example="white")
    sex: str = Field(..., example="male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")


class Income(BaseModel):
    Income: str = Field(..., example=">50K")


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


@app.post("/predict", response_model=Income)
def predict(payload: CensusData):
    df = pd.DataFrame.from_dict([payload.dict(by_alias=True)])
    X, _, _, _ = process_data(
        df, categorical_features=CAT_FEATURES,
        training=False,
        encoder=ENCODER
    )
    pred = inference(MODEL, X)

    if pred == 1:
        prediction = ">50K"
    elif pred == 0:
        prediction = "<=50K"
    return {"Income": prediction}