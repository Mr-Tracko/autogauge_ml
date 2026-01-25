import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
model = joblib.load(ARTIFACTS_DIR / "model.pkl")

def predict_price(input_dict):
    df = pd.DataFrame([input_dict])
#ML pipelines expect 2D data
#Single row → wrap inside list

    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)

    return float(prediction[0])
    #bcz this float return an array even if we passed only 1 row , the array would be of the type , [1,] , so , we always return the first value only

if __name__ == "__main__":
    sample_input = {
        "brand": "Maruti",
        "Car Name": "Swift",
        "Drive": "Manual",
        "Type": "HatchBack",
        "Distance": 42000,
        "Year": 2019,
        "Owner": -1
    }

    price = predict_price(sample_input)
    print("Predicted Price:", round(price, 2))