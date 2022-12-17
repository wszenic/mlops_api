from fastapi import FastAPI

from starter.model_inference import ModelInput, get_predictions_from_model

app = FastAPI()


@app.get("/")
async def welcome_message():
    return {"greeting": "Welcome to the model's API"}


@app.post("/inference/")
async def model_inference(model_input: ModelInput):
    pred = get_predictions_from_model(model_input)
    print(pred)
    return {"prediction": pred}
