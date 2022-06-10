import json

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, conlist

from model import get_model

app = FastAPI()


class PredictionInputData(BaseModel):
    number: conlist(int, min_items=1)


@app.get("/healthcheck")
async def healthcheck():
    return {"is_alive": True}


@app.post("/predict")
async def predict(input_data: PredictionInputData):
    model = get_model()
    X = pd.DataFrame({"number": input_data.number})
    prediction = model.predict(X).tolist()
    return {"result": prediction}
