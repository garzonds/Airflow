import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODELS_PATH = os.environ.get("MODELS_PATH", "/models")

# Cargar artefactos
with open(os.path.join(MODELS_PATH, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_PATH, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
with open(os.path.join(MODELS_PATH, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

app = FastAPI(title="Penguin Classifier")


class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: str   # Biscoe, Dream, Torgersen
    sex: str      # male, female


@app.post("/predict")
def predict(features: PenguinFeatures):
    # Construir DataFrame
    df = pd.DataFrame([{
        "bill_length_mm":    features.bill_length_mm,
        "bill_depth_mm":     features.bill_depth_mm,
        "flipper_length_mm": features.flipper_length_mm,
        "body_mass_g":       features.body_mass_g,
        "island":            features.island.capitalize(),
        "sex":               features.sex.lower(),
    }])

    # Preprocesar
    df = pd.get_dummies(df, columns=["island", "sex"])
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predecir
    pred = model.predict(df)[0]
    species = le.inverse_transform([pred])[0]

    return {"species": species}