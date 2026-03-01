"""
Task 2: Cargar datos crudos de penguins a SQLite SIN preprocesamiento.
Se guardan tal cual vienen del dataset original, incluyendo nulos.
"""
import sqlite3
import os
import logging
import pandas as pd
from palmerpenguins import load_penguins

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("SQLITE_DB_PATH", "/opt/airflow/data/penguins.db")


def load_raw_data():
    logger.info("Cargando dataset de penguins...")

    # Cargar dataset desde palmerpenguins
    df = load_penguins()
    logger.info(f"Dataset cargado desde palmerpenguins. Shape: {df.shape}")

    logger.info(f"Columnas: {list(df.columns)}")
    logger.info(f"Nulos por columna:\n{df.isnull().sum()}")
    logger.info(f"Distribución de especies:\n{df['species'].value_counts()}")

    # Guardar en SQLite SIN ningún preprocesamiento
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    df.to_sql(
        name="penguins_raw",
        con=conn,
        if_exists="replace",
        index=True,
        index_label="id"
    )

    # Verificar carga
    count = pd.read_sql("SELECT COUNT(*) as total FROM penguins_raw", conn).iloc[0, 0]
    logger.info(f"Registros cargados en 'penguins_raw': {count}")

    conn.close()
    logger.info("Carga de datos crudos completada exitosamente.")


if __name__ == "__main__":
    load_raw_data()