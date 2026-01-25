import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "cars_24_combined.csv"

print("Looking for file at:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

df["brand"] = df["Car Name"].str.split().str[0]
# print(df)

cols = df.columns.tolist()
cols[-1], cols[-2] = cols[-2], cols[-1]
df = df[cols]
cols[-2], cols[-9] = cols[-9], cols[-2]
df = df[cols]
cols[-10], cols[-9] = cols[-9], cols[-10]
df = df[cols]
df["Car Name"] = df["Car Name"].str.split(n=1).str[1]
# print(df)
df["Owner"] = -df["Owner"]


transformer = ColumnTransformer(
    transformers=[
        (
            "brand_ohe",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]),
            ["brand"]
        ),

        (
            "drive_ohe",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]),
            ["Drive"]
        ),

        (
            "type_ord",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(
                    categories=[["HatchBack", "Sedan", "Lux_sedan", "SUV", "Lux_SUV"]]
                ))
            ]),
            ["Type"]
        ),

        (
            "car_te",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("te", TargetEncoder())
            ]),
            ["Car Name"]
        ),

        (
            "num_scaled",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            ["Distance", "Year", "Owner"]
        ),
    ],
    remainder="drop"
)

X = df.drop(columns=["Price"])
y = df["Price"]
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , random_state = 42)