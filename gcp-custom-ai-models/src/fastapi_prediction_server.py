import os
import json
import pickle
from dotenv import load_dotenv

import tensorflow as tf
#from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#from preprocessor import TextPreprocessor
from model_prediction import CustomModelPrediction


load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
assert BASE_DIR, 'error base dir'

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    instances: str


class Item(BaseModel):
    name: str
    description: str


model_dir = os.getenv('MODEL_DIR')
assert model_dir
local_model = tf.keras.models.load_model(model_dir)
local_model.trainable = False

with open('{}/processor_state.pkl'.format(BASE_DIR), 'rb') as f:
    p = pickle.load(f)
predict_server = CustomModelPrediction(local_model, p)


@app.get("/hello_world")
async def hello_world():
    return {"msg": "Hello World"}


@app.post("/hello_world_post")
async def hello_world_post(item: Item):
    return item


@app.post("/predict")
async def predict(input: PredictionInput):
    try:
        instances = json.loads(input.instances)
        predictions = predict_server.predict(instances)
        return {"predictions": predictions}
    except Exception as ex:
        return {"msg": "error", "error": str(ex)}
