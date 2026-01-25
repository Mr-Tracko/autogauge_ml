from fastapi import FastAPI
from pydantic import BaseModel
# Used for data validation
# Ensures incoming request data:
# has correct fields
# has correct data types
# Prevents garbage / invalid input from reaching your ML model

import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
model = joblib.load(ARTIFACTS_DIR / "model.pkl")

# Defines what input your API accepts
class CarInput(BaseModel):
    brand : str
    Car_Name : str
    Drive : str
    Type: str
    Distance: int
    Year: int
    Owner: int

@app.post("/predict")
#this returns a decorator
#@app.post("/predict") is a decorator that registers the function below it as a POST endpoint at /predict by adding it to FastAPI’s internal routing table.

def predict(car: CarInput):
    df = pd.DataFrame([{
        "brand": car.brand,
        "Car Name": car.Car_Name,
        "Drive": car.Drive,
        "Type": car.Type,
        "Distance": car.Distance,
        "Year": car.Year,
        "Owner": car.Owner
    }])

    X = preprocessor.transform(df)
    price = model.predict(X)[0]

    return {
        "predicted_price": round(float(price), 2)
    }


#to run the ML folder : uvicorn src.app:app --reload
