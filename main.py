from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional
import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

app = FastAPI()

# Configuración de CORS
origins = [
    "*",
    # "https://upao-proyects.web.app",
]

# Configuración de CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga del modelo
model = joblib.load("lgbm_consumption_model_v2.joblib")
# Carga del imputador y el orden de las características
imputer: SimpleImputer = joblib.load("imputer.joblib")
# Carga del orden de las características
feature_order = joblib.load("feature_order.joblib")

# Definición del modelo de Request para la predicción
class PredictRequest(BaseModel):
    features: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Diccionario con nombre de feature: valor. Solo las que quieras dar."
    )
    horizon_months: int = Field(
        0,
        description="Cuántos meses en el futuro quieres predecir (0 = mes actual)."
    )


# Endpoint para la predicción
@app.post("/predict")
def predict(req: PredictRequest):
    data = {feat: np.nan for feat in feature_order}

    # Calcula el mes objetivo
    today = datetime.date.today()
    future_month = (today.month - 1 + req.horizon_months) % 12 + 1
    data["month"] = future_month

    # Sobreescribe con los valores que envió el usuario
    for k, v in req.features.items():
        if k not in data:
            return {"error": f"Feature desconocida: {k}"}
        data[k] = v

    # Construye DataFrame y aplica imputación
    X = pd.DataFrame([data], columns=feature_order)
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=feature_order,
        index=[0]
    )

    # Predicción
    y_pred = model.predict(X_imputed)[0]

    return {
        "predicted_month": future_month,
        "predicted_consumption_kwh": float(y_pred)
    }
