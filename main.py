from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Optional
import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

app = FastAPI()

# --- 1. Al arrancar, carga todo ---
model = joblib.load("lgbm_consumption_model_v2.joblib")
imputer: SimpleImputer = joblib.load("imputer.joblib")
feature_order = joblib.load("feature_order.joblib")


# --- 2. Define el esquema de entrada ---
class PredictRequest(BaseModel):
    features: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Diccionario con nombre de feature: valor. Solo las que quieras dar."
    )
    horizon_months: int = Field(
        0,
        description="Cuántos meses en el futuro quieres predecir (0 = mes actual)."
    )


# --- 3. Endpoint de predicción ---
@app.post("/predict")
def predict(req: PredictRequest):
    # 3.1 Arranca un dict con NaN en todas las columnas
    data = {feat: np.nan for feat in feature_order}

    # 3.2 Calcula el mes objetivo
    today = datetime.date.today()
    future_month = (today.month - 1 + req.horizon_months) % 12 + 1
    data["month"] = future_month

    # 3.3 Sobreescribe con los valores que envió el usuario
    for k, v in req.features.items():
        if k not in data:
            return {"error": f"Feature desconocida: {k}"}
        data[k] = v

    # 3.4 Construye DataFrame y aplica imputación
    X = pd.DataFrame([data], columns=feature_order)
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=feature_order,
        index=[0]
    )

    # 3.5 Predicción
    y_pred = model.predict(X_imputed)[0]

    return {
        "predicted_month": future_month,
        "predicted_consumption_kwh": float(y_pred)
    }
