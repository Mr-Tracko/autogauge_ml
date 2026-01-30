import joblib
#joblib basically saves and loads the data directly when needed , it reduces the load of our website

# joblib is a Python library used to save and load ML objects efficiently.
# Especially:
# trained models
# preprocessing pipelines
# large NumPy arrays
# In short:
# joblib = save your trained ML brain to disk

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_absolute_error

from preprocess import (
    X_train,
    X_test,
    y_train,
    y_test,
    transformer
)

X_train_processed = transformer.fit_transform(X_train, y_train)
X_test_processed = transformer.transform(X_test)

model = LinearRegression()
model.fit(X_train_processed , y_train)
y_pred = model.predict(X_test_processed)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

joblib.dump(transformer, "artifacts/preprocessor.pkl")
joblib.dump(model, "artifacts/model.pkl")

print("✅ Model and preprocessor saved successfully")

#requirement.txt is used to save the particular versions of everything used in this project , so that in further situation , one can directly run access this repo using those same requirements.txt file folder