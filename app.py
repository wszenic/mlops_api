from functools import lru_cache

import uvicorn
from fastapi import FastAPI, Depends

from starter.model_inference import ModelInput, get_predictions_from_model
from starter.starter.config import Settings

app = FastAPI()


@lru_cache
def get_settings():
    return Settings(_env_file=".env")


@app.get("/")
async def welcome_message(settings: Settings = Depends(get_settings)):
    return {
        "greeting": "Welcome to the model's API",
        "environment": settings.environment
    }


@app.post("/inference/")
async def model_inference(model_input: ModelInput, settings: Settings = Depends(get_settings)):
    pred = get_predictions_from_model(model_input, settings)
    return {
        "prediction": pred,
        "environment": settings.environment
    }
