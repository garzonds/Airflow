"""
Task 3: Preprocesamiento de datos para entrenamiento.
Lee desde penguins_raw, aplica transformaciones y guarda en penguins_processed.

Pasos:
  - Eliminar filas con nulos
  - Encoding de variables categóricas (island, sex)
  - Encoding del target (species)
  - Escalado de features numéricas con StandardScaler
  - Guardar scaler para usarlo en inferencia
"""
import sqlite3
import os
import logging
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH    = os.environ.get("SQLITE_DB_PATH", "/opt/airflow/data/penguins.db")
MODELS_PATH = os.environ.get("MODELS_PATH",   "/opt/airflow/models")


def preprocess_data():
    logger.info("Iniciando preprocesamiento...")

    # ── 1. Leer datos crudos ────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM penguins_raw", conn)
    conn.close()

    logger.info(f"Datos crudos leídos. Shape: {df.shape}")
    logger.info(f"Nulos antes de limpiar:\n{df.isnull().sum()}")

    # ── 2. Eliminar columna índice si viene de SQLite ───────────────────────
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # ── 3. Eliminar filas con nulos ─────────────────────────────────────────
    df = df.dropna()
    logger.info(f"Shape tras eliminar nulos: {df.shape}")

    # ── 4. Encoding del target: species → número ────────────────────────────
    le_species = LabelEncoder()
    df["species_encoded"] = le_species.fit_transform(df["species"])
    logger.info(f"Clases de species: {list(le_species.classes_)}")

    # ── 5. Encoding de categóricas: island y sex ────────────────────────────
    df = pd.get_dummies(df, columns=["island", "sex"], drop_first=False)

    # ── 6. Escalar features numéricas ───────────────────────────────────────
    numeric_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f"Features escaladas: {numeric_cols}")

    # ── 7. Guardar en tabla procesada ───────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(
        name="penguins_processed",
        con=conn,
        if_exists="replace",
        index=False
    )
    count = pd.read_sql(
        "SELECT COUNT(*) as total FROM penguins_processed", conn
    ).iloc[0, 0]
    conn.close()
    logger.info(f"Registros guardados en 'penguins_processed': {count}")

    # ── 8. Persistir scaler y label encoder para la API ─────────────────────
    os.makedirs(MODELS_PATH, exist_ok=True)

    scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler guardado en: {scaler_path}")

    le_path = os.path.join(MODELS_PATH, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le_species, f)
    logger.info(f"LabelEncoder guardado en: {le_path}")

    # ── 9. Guardar columnas del modelo (orden de features) ──────────────────
    feature_cols = [c for c in df.columns if c not in ["species", "species_encoded"]]
    cols_path = os.path.join(MODELS_PATH, "feature_columns.pkl")
    with open(cols_path, "wb") as f:
        pickle.dump(feature_cols, f)
    logger.info(f"Columnas de features guardadas: {feature_cols}")

    logger.info("Preprocesamiento completado exitosamente.")


if __name__ == "__main__":
    preprocess_data()