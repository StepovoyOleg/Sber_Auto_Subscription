import os
import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Загрузка модели и кодировщика
path = os.environ.get('PROJECT_PATH', '..')
with open(f"{path}/data/model/model.pkl", 'rb') as file:
    model_dict  = dill.load(file)

# Извлечение модели из словаря
model = model_dict['model']


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    device_category: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str
    utm_medium_added_is_organic: int
    utm_source_added_is_social: int

class Prediction(BaseModel):
    utm_medium: str
    pred: int

@app.get("/status")
def status():
    return "I'm OK!"

@app.get("/version")
def version():
    return model_dict['metadata']

@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    # Преобразование данных в DataFrame
    df = pd.DataFrame.from_dict([form.dict()])
    # Предсказание
    y = model.predict(df)

    return {
        'utm_medium': form.utm_medium,
        'pred': int(y[0]),
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)