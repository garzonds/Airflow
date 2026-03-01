"""
DAG: penguin_pipeline
Pipeline completo de MLOps para el dataset de penguins.

Tareas:
  1. clear_database      → Borra tablas existentes en SQLite
  2. load_raw_data       → Carga datos crudos (sin preprocesamiento)
  3. preprocess_data     → Preprocesa y guarda en tabla procesada
  4. train_model         → Entrena Logistic Regression y guarda modelo

Dependencias:
  clear_database >> load_raw_data >> preprocess_data >> train_model
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import importlib.util
import sys
import os


# ── Función helper para importar scripts desde /opt/airflow/scripts ─────────
def import_script(script_name: str):
    """Importa dinámicamente un script Python desde la carpeta scripts."""
    script_path = f"/opt/airflow/scripts/{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Funciones wrapper para cada tarea ────────────────────────────────────────
def task_clear_database(**context):
    module = import_script("clear_db")
    module.clear_database()


def task_load_raw_data(**context):
    module = import_script("load_data")
    module.load_raw_data()


def task_preprocess_data(**context):
    module = import_script("preprocess")
    module.preprocess_data()


def task_train_model(**context):
    module = import_script("train_model")
    module.train_model()


# ── Argumentos por defecto del DAG ───────────────────────────────────────────
default_args = {
    "owner": "mlops-taller",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ── Definición del DAG ───────────────────────────────────────────────────────
with DAG(
    dag_id="penguin_pipeline",
    description="Pipeline MLOps: carga, preprocesa y entrena modelo sobre dataset penguins",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # Solo se ejecuta manualmente (o vía trigger)
    catchup=False,
    tags=["mlops", "penguins", "taller3"],
) as dag:

    # ── Task 0: Inicio (señal visual en la UI) ───────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── Task 1: Borrar base de datos ─────────────────────────────────────────
    clear_db = PythonOperator(
        task_id="clear_database",
        python_callable=task_clear_database,
        doc_md="""
        ### Clear Database
        Borra todas las tablas existentes en la base de datos SQLite.
        Garantiza que cada ejecución del pipeline parta desde cero.
        """,
    )

    # ── Task 2: Cargar datos crudos ──────────────────────────────────────────
    load_data = PythonOperator(
        task_id="load_raw_data",
        python_callable=task_load_raw_data,
        doc_md="""
        ### Load Raw Data
        Descarga el dataset de penguins usando palmerpenguins y lo guarda
        en la tabla `penguins_raw` SIN ningún preprocesamiento.
        Incluye nulos y valores categóricos originales.
        """,
    )

    # ── Task 3: Preprocesamiento ─────────────────────────────────────────────
    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
        doc_md="""
        ### Preprocess Data
        Lee desde `penguins_raw` y aplica:
        - Eliminación de filas con nulos
        - Label encoding del target (species)
        - One-hot encoding de categóricas (island, sex)
        - StandardScaler sobre features numéricas
        Guarda resultado en `penguins_processed` y persiste scaler + encoder.
        """,
    )

    # ── Task 4: Entrenamiento ────────────────────────────────────────────────
    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        doc_md="""
        ### Train Model
        Lee desde `penguins_processed`, entrena una Logistic Regression
        (multinomial, solver=lbfgs), evalúa métricas y guarda:
        - model.pkl       → modelo entrenado
        - metrics.json    → accuracy, classification report, confusion matrix
        """,
    )

    # ── Task 5: Fin (señal visual en la UI) ──────────────────────────────────
    end = EmptyOperator(task_id="end")

    # ── Definir dependencias (flujo secuencial) ───────────────────────────────
    start >> clear_db >> load_data >> preprocess >> train >> end