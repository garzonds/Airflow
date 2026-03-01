"""
Task 4: Entrenamiento del modelo de Logistic Regression.
Lee los datos preprocesados desde SQLite, entrena el modelo,
evalúa métricas y guarda el modelo en disco para la API.
"""
import sqlite3
import os
import logging
import pickle
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH     = os.environ.get("SQLITE_DB_PATH", "/opt/airflow/data/penguins.db")
MODELS_PATH = os.environ.get("MODELS_PATH",    "/opt/airflow/models")


def train_model():
    logger.info("Iniciando entrenamiento del modelo...")

    # ── 1. Cargar datos preprocesados ───────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM penguins_processed", conn)
    conn.close()
    logger.info(f"Datos cargados. Shape: {df.shape}")

    # ── 2. Cargar columnas de features (definidas en preprocesamiento) ───────
    cols_path = os.path.join(MODELS_PATH, "feature_columns.pkl")
    with open(cols_path, "rb") as f:
        feature_cols = pickle.load(f)
    logger.info(f"Features usadas ({len(feature_cols)}): {feature_cols}")

    # ── 3. Separar X e y ────────────────────────────────────────────────────
    X = df[feature_cols]
    y = df["species_encoded"]

    logger.info(f"Distribución del target:\n{y.value_counts()}")

    # ── 4. Split train / test ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 5. Entrenar Logistic Regression ─────────────────────────────────────
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class="multinomial",
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    logger.info("Modelo entrenado.")

    # ── 6. Evaluar métricas ──────────────────────────────────────────────────
    le_path = os.path.join(MODELS_PATH, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=le.classes_)
    cm       = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"\n{'='*50}")
    logger.info(f"ACCURACY: {accuracy:.4f}")
    logger.info(f"\nCLASSIFICATION REPORT:\n{report}")
    logger.info(f"CONFUSION MATRIX: {cm}")
    logger.info(f"{'='*50}")

    # ── 7. Guardar métricas en JSON ──────────────────────────────────────────
    os.makedirs(MODELS_PATH, exist_ok=True)
    metrics = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classes": list(le.classes_),
    }
    metrics_path = os.path.join(MODELS_PATH, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métricas guardadas en: {metrics_path}")

    # ── 8. Guardar modelo ────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_PATH, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Modelo guardado en: {model_path}")

    logger.info("Entrenamiento completado exitosamente.")


if __name__ == "__main__":
    train_model()