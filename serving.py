import json
from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model import get_model

app = FastAPI()


class PredictionInputData(BaseModel):
    number: List[int]


@app.get("/healthcheck")
async def healthcheck():
    return {"is_alive": True}


@app.post("/predict")
async def predict(input_data: PredictionInputData):
    model = get_model()
    X = pd.DataFrame(json.loads(input_data.json()))
    prediction = model.predict(X).tolist()
    return {"result": prediction}
