"""
API de inferencia para el modelo de clasificación de penguins.
Carga el modelo entrenado por Airflow y expone endpoints para predicción.
"""
import os
import pickle
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────
MODELS_PATH = os.environ.get("MODELS_PATH", "/app/models")

# ── Carga de artefactos ───────────────────────────────────────────────────────
def load_artifacts():
    """Carga modelo, scaler y encoders desde disco."""
    artifacts = {}
    files = {
        "model":           "model.pkl",
        "scaler":          "scaler.pkl",
        "label_encoder":   "label_encoder.pkl",
        "feature_columns": "feature_columns.pkl",
    }
    for key, filename in files.items():
        path = os.path.join(MODELS_PATH, filename)
        if not os.path.exists(path):
            logger.warning(f"Artefacto no encontrado: {path}")
            artifacts[key] = None
        else:
            with open(path, "rb") as f:
                artifacts[key] = pickle.load(f)
            logger.info(f"Artefacto cargado: {filename}")
    return artifacts


# Cargar al iniciar la app (si ya existen)
artifacts = load_artifacts()

# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Penguin Species Classifier",
    description="API de inferencia para clasificación de especies de pingüinos. "
                "Entrena el modelo primero ejecutando el DAG en Airflow.",
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PenguinFeatures(BaseModel):
    """Features de entrada para predecir la especie de un pingüino."""
    bill_length_mm: float = Field(..., example=45.0, description="Longitud del pico en mm")
    bill_depth_mm: float  = Field(..., example=15.0, description="Profundidad del pico en mm")
    flipper_length_mm: float = Field(..., example=200.0, description="Longitud de la aleta en mm")
    body_mass_g: float    = Field(..., example=4000.0, description="Masa corporal en gramos")
    island: str           = Field(..., example="Biscoe", description="Isla: Biscoe, Dream o Torgersen")
    sex: str              = Field(..., example="male", description="Sexo: male o female")


class PredictionResponse(BaseModel):
    """Respuesta de la predicción."""
    species: str
    confidence: float
    probabilities: dict[str, float]
    input_received: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    artifacts: dict[str, bool]


# ── Helpers ───────────────────────────────────────────────────────────────────
def reload_artifacts():
    """Recarga los artefactos desde disco (útil tras entrenar el modelo)."""
    global artifacts
    artifacts = load_artifacts()


def preprocess_input(features: PenguinFeatures) -> pd.DataFrame:
    """
    Aplica el mismo preprocesamiento que se usó en el entrenamiento:
    - One-hot encoding de island y sex
    - StandardScaler sobre features numéricas
    - Alinear columnas con las del entrenamiento
    """
    # Construir DataFrame base
    raw = {
        "bill_length_mm":   features.bill_length_mm,
        "bill_depth_mm":    features.bill_depth_mm,
        "flipper_length_mm": features.flipper_length_mm,
        "body_mass_g":      features.body_mass_g,
        "island":           features.island.capitalize(),
        "sex":              features.sex.lower(),
    }
    df = pd.DataFrame([raw])

    # One-hot encoding (mismas columnas que en entrenamiento)
    df = pd.get_dummies(df, columns=["island", "sex"], drop_first=False)

    # Escalar numéricas
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    df[numeric_cols] = artifacts["scaler"].transform(df[numeric_cols])

    # Alinear columnas con las del entrenamiento (agregar las que falten con 0)
    feature_columns = artifacts["feature_columns"]
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Penguin Species Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Verifica el estado de la API y si el modelo está cargado."""
    return {
        "status": "ok",
        "model_loaded": artifacts.get("model") is not None,
        "artifacts": {k: v is not None for k, v in artifacts.items()},
    }


@app.post("/reload", tags=["General"])
def reload():
    """
    Recarga el modelo desde disco.
    Útil después de ejecutar el DAG de Airflow para obtener el modelo más reciente.
    """
    reload_artifacts()
    loaded = {k: v is not None for k, v in artifacts.items()}
    return {
        "message": "Artefactos recargados correctamente.",
        "artifacts": loaded,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: PenguinFeatures):
    """
    Predice la especie de un pingüino dado sus características.

    Especies posibles:
    - **Adelie**
    - **Chinstrap**
    - **Gentoo**
    """
    # Verificar que el modelo esté cargado
    if artifacts.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta el DAG de Airflow primero, "
                   "luego llama a POST /reload.",
        )

    try:
        # Preprocesar entrada
        X = preprocess_input(features)

        # Predicción
        model = artifacts["model"]
        le    = artifacts["label_encoder"]

        pred_encoded   = model.predict(X)[0]
        pred_proba     = model.predict_proba(X)[0]
        species        = le.inverse_transform([pred_encoded])[0]
        confidence     = float(np.max(pred_proba))
        probabilities  = {
            cls: round(float(prob), 4)
            for cls, prob in zip(le.classes_, pred_proba)
        }

        logger.info(f"Predicción: {species} (confianza: {confidence:.4f})")

        return {
            "species":        species,
            "confidence":     round(confidence, 4),
            "probabilities":  probabilities,
            "input_received": features.model_dump(),
        }

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.get("/metrics", tags=["Model"])
def get_metrics():
    """
    Retorna las métricas del último modelo entrenado.
    (accuracy, classification report, confusion matrix)
    """
    metrics_path = os.path.join(MODELS_PATH, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Métricas no encontradas. Ejecuta el DAG de Airflow primero.",
        )
    with open(metrics_path, "r") as f:
        return json.load(f)