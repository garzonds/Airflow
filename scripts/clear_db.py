"""
Task 1: Borrar contenido de la base de datos SQLite.
Elimina las tablas si existen, dejando la DB lista para una carga fresca.
"""
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("SQLITE_DB_PATH", "/opt/airflow/data/penguins.db")


def clear_database():
    logger.info(f"Conectando a la base de datos: {DB_PATH}")

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Obtener todas las tablas existentes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    if not tables:
        logger.info("No hay tablas que borrar. La base de datos está vacía.")
    else:
        for (table_name,) in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            logger.info(f"Tabla eliminada: {table_name}")

    conn.commit()
    conn.close()
    logger.info("Base de datos limpiada exitosamente.")


if __name__ == "__main__":
    clear_database()