"""
DAG: penguin_pipeline
Pipeline completo de MLOps para el dataset de penguins.
Usa MySQL para almacenamiento de datos.

Flujo:
  start >> clear_database >> load_raw_data >> preprocess_data >> train_model >> end
"""

import os
import pickle
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
from palmerpenguins import load_penguins
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sqlalchemy import create_engine, text

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# ── Configuración ─────────────────────────────────────────────────────────────
MYSQL_CONN  = os.environ.get("MYSQL_DATA_CONN", "mysql+pymysql://root:root@mysql_data:3306/penguins_db")
MODELS_PATH = os.environ.get("MODELS_PATH", "/opt/airflow/models")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TASK 1 — Borrar base de datos
# =============================================================================
def clear_database(**context):
    logger.info(f"Conectando a MySQL: {MYSQL_CONN}")
    engine = create_engine(MYSQL_CONN)

    with engine.connect() as conn:
        result = conn.execute(text("SHOW TABLES"))
        tables = [row[0] for row in result]

        if not tables:
            logger.info("No hay tablas que borrar.")
        else:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            for table in tables:
                conn.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
                logger.info(f"Tabla eliminada: {table}")
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            conn.commit()

    logger.info("Base de datos limpiada exitosamente.")


# =============================================================================
# TASK 2 — Cargar datos crudos
# =============================================================================
def load_raw_data(**context):
    logger.info("Cargando dataset de penguins con palmerpenguins...")

    df = load_penguins()
    logger.info(f"Dataset cargado. Shape: {df.shape}")
    logger.info(f"Nulos por columna:\n{df.isnull().sum()}")
    logger.info(f"Distribucion de especies:\n{df['species'].value_counts()}")

    engine = create_engine(MYSQL_CONN)
    df.to_sql("penguins_raw", engine, if_exists="replace", index=True, index_label="id")

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM penguins_raw")).scalar()
    logger.info(f"Registros cargados en 'penguins_raw': {count}")
    logger.info("Carga completada.")


# =============================================================================
# TASK 3 — Preprocesamiento
# =============================================================================
def preprocess_data(**context):
    logger.info("Iniciando preprocesamiento...")

    engine = create_engine(MYSQL_CONN)
    df = pd.read_sql("SELECT * FROM penguins_raw", engine)
    logger.info(f"Datos crudos leidos. Shape: {df.shape}")

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df.dropna()
    logger.info(f"Shape tras eliminar nulos: {df.shape}")

    le_species = LabelEncoder()
    df["species_encoded"] = le_species.fit_transform(df["species"])
    logger.info(f"Clases: {list(le_species.classes_)}")

    df = pd.get_dummies(df, columns=["island", "sex"], drop_first=False)

    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df.to_sql("penguins_processed", engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM penguins_processed")).scalar()
    logger.info(f"Registros en 'penguins_processed': {count}")

    os.makedirs(MODELS_PATH, exist_ok=True)

    with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_PATH, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le_species, f)

    feature_cols = [c for c in df.columns if c not in ["species", "species_encoded"]]
    with open(os.path.join(MODELS_PATH, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    logger.info("Preprocesamiento completado.")


# =============================================================================
# TASK 4 — Entrenamiento del modelo
# =============================================================================
def train_model(**context):
    logger.info("Iniciando entrenamiento del modelo...")

    engine = create_engine(MYSQL_CONN)
    df = pd.read_sql("SELECT * FROM penguins_processed", engine)
    logger.info(f"Datos cargados. Shape: {df.shape}")

    with open(os.path.join(MODELS_PATH, "feature_columns.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(os.path.join(MODELS_PATH, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    X = df[feature_cols]
    y = df["species_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    model = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial", solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=le.classes_)
    cm       = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"ACCURACY: {accuracy:.4f}")
    logger.info(f"CLASSIFICATION REPORT:\n{report}")

    metrics = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classes": list(le.classes_),
    }
    with open(os.path.join(MODELS_PATH, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODELS_PATH, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    logger.info("Entrenamiento completado.")


# =============================================================================
# DEFINICIÓN DEL DAG
# =============================================================================
default_args = {
    "owner": "mlops-taller",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="penguin_pipeline",
    description="Pipeline MLOps: carga, preprocesa y entrena modelo sobre dataset penguins",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "penguins", "taller3"],
) as dag:

    start       = EmptyOperator(task_id="start")
    t_clear     = PythonOperator(task_id="clear_database",  python_callable=clear_database)
    t_load      = PythonOperator(task_id="load_raw_data",   python_callable=load_raw_data)
    t_preprocess= PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t_train     = PythonOperator(task_id="train_model",     python_callable=train_model)
    end         = EmptyOperator(task_id="end")

    start >> t_clear >> t_load >> t_preprocess >> t_train >> end