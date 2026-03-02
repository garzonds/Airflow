# 🐧 Taller 3 MLOps — Pipeline de Penguins con Airflow

Pipeline completo de MLOps que orquesta la carga, preprocesamiento y entrenamiento de un modelo de clasificación de especies de pingüinos, usando Apache Airflow como orquestador y una API de inferencia construida con FastAPI.

---

## 📋 Descripción del proyecto

Este proyecto implementa un pipeline de Machine Learning end-to-end completamente dockerizado. El flujo parte desde la carga de datos crudos hasta la inferencia en producción, pasando por preprocesamiento y entrenamiento de un modelo de Regresión Logística.

El dataset utilizado es [Palmer Penguins](https://pypi.org/project/palmerpenguins/), que contiene mediciones físicas de tres especies de pingüinos: **Adelie**, **Chinstrap** y **Gentoo**.

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose                          │
│                                                             │
│  ┌──────────────┐     ┌──────────────────────────────────┐ │
│  │  mysql_data  │     │           AIRFLOW                │ │
│  │  puerto 3307 │◄────│  init → webserver → scheduler   │ │
│  │  penguins_db │     └──────────────────────────────────┘ │
│  └──────────────┘                    │                      │
│                                      │ genera               │
│  ┌──────────────┐                    ▼                      │
│  │mysql_airflow │     ┌──────────────────────────────────┐ │
│  │  puerto 3308 │     │         ./models/                │ │
│  │  airflow_db  │     │  model.pkl, scaler.pkl, etc.     │ │
│  └──────────────┘     └──────────────┬───────────────────┘ │
│                                      │ lee                  │
│                        ┌─────────────▼───────────────────┐ │
│                        │      FastAPI (puerto 8000)       │ │
│                        │      POST /predict               │ │
│                        └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Servicios

| Servicio | Imagen | Puerto | Descripción |
|---|---|---|---|
| `mysql_data` | mysql:8.0 | 3307 | Base de datos exclusiva para datos de penguins |
| `mysql_airflow` | mysql:8.0 | 3308 | Base de datos exclusiva para metadatos de Airflow |
| `airflow-init` | Dockerfile.airflow | — | Inicializa la DB y crea el usuario admin |
| `airflow-webserver` | Dockerfile.airflow | 8080 | Interfaz web de Airflow |
| `airflow-scheduler` | Dockerfile.airflow | — | Orquesta y ejecuta los DAGs |
| `api` | api/Dockerfile | 8000 | API de inferencia con FastAPI |

---

## 📁 Estructura del proyecto

```
Airflow/
├── api/
│   ├── Dockerfile          # Imagen de la API
│   ├── main.py             # Endpoint de inferencia
│   └── requirements.txt    # Dependencias de FastAPI
├── dags/
│   └── penguin_pipeline.py # DAG con las 4 tareas del pipeline
├── models/                 # Artefactos generados por el DAG
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_columns.pkl
│   └── metrics.json
├── data/                   # Carpeta reservada (generada automáticamente)
├── .env                    # Variables de entorno
├── Dockerfile.airflow      # Imagen personalizada de Airflow
└── docker-compose.yml      # Orquestación de todos los servicios
```

---

## ⚙️ Requisitos previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo
- Al menos **4 GB de RAM** asignados a Docker
- Puertos **8080**, **8000**, **3307** y **3308** disponibles

> **Windows:** Docker Desktop debe estar en modo Linux containers.

---

## 🚀 Instalación paso a paso

### 1. Clona el repositorio

```bash
git clone https://github.com/garzonds/Airflow.git
cd Airflow
```

### 2. Crea el archivo `.env`

**Linux / Mac:**
```bash
echo "AIRFLOW_UID=$(id -u)" > .env
```

**Windows (PowerShell):**
```powershell
Set-Content -Path ".env" -Encoding UTF8 -Value "AIRFLOW_UID=50000"
```

El archivo debe quedar así:
```
AIRFLOW_UID=50000
```

### 3. Levanta todos los servicios

```bash
docker-compose up --build -d
```

> La primera vez tarda entre 3 y 5 minutos porque construye las imágenes personalizadas de Airflow y la API.

### 4. Verifica que todo esté corriendo

```bash
docker-compose ps
```

Debes ver todos los contenedores en estado `Up`:
```
airflow_scheduler   Up
airflow_webserver   Up   0.0.0.0:8080->8080/tcp
mysql_airflow       Up   0.0.0.0:3308->3306/tcp
mysql_data          Up   0.0.0.0:3307->3306/tcp
penguins_api        Up   0.0.0.0:8000->8000/tcp
```

### 5. Verifica que Airflow se inicializó correctamente

```bash
docker-compose logs airflow-init
```

Al final debe aparecer:
```
Airflow init completado!
```

---

## 🔄 Cómo usar el DAG

### 1. Abre la interfaz de Airflow

Ve a **http://localhost:8080** en tu navegador.

- **Usuario:** `admin`
- **Contraseña:** `admin`

### 2. Activa el DAG

Busca el DAG llamado `penguin_pipeline` y actívalo con el toggle de la izquierda (de gris a azul).

### 3. Ejecuta el DAG manualmente

Haz clic en el botón **▶ Trigger DAG** a la derecha del DAG.

### 4. Observa el progreso

Haz clic en el nombre del DAG para ver el gráfico de ejecución. Las tareas deben irse poniendo verdes en orden:

```
start → clear_database → load_raw_data → preprocess_data → train_model → end
```

### ¿Qué hace cada tarea?

| Tarea | Descripción |
|---|---|
| `clear_database` | Elimina todas las tablas de `penguins_db` para partir desde cero |
| `load_raw_data` | Carga los 344 registros de penguins **sin ningún preprocesamiento** en la tabla `penguins_raw` |
| `preprocess_data` | Elimina nulos, aplica one-hot encoding a variables categóricas y StandardScaler a variables numéricas. Guarda en `penguins_processed` |
| `train_model` | Entrena una Logistic Regression, evalúa métricas y guarda `model.pkl` y `metrics.json` en `./models/` |

### Verificar los datos en MySQL

```bash
docker exec -it mysql_data mysql -u root -proot penguins_db
```

```sql
SHOW TABLES;
SELECT COUNT(*) FROM penguins_raw;
SELECT COUNT(*) FROM penguins_processed;
SELECT * FROM penguins_raw LIMIT 5;
EXIT;
```

---

## 🤖 Cómo usar la API

> **Importante:** La API solo funciona después de que el DAG haya corrido exitosamente, ya que necesita los archivos `.pkl` generados por el entrenamiento.

### Reiniciar la API tras el primer entrenamiento

```bash
docker-compose restart api
```

### Hacer una predicción

**Linux / Mac:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 45.0,
    "bill_depth_mm": 15.0,
    "flipper_length_mm": 200.0,
    "body_mass_g": 4000.0,
    "island": "Biscoe",
    "sex": "male"
  }'
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" `
  -ContentType "application/json" `
  -Body '{"bill_length_mm": 45.0, "bill_depth_mm": 15.0, "flipper_length_mm": 200.0, "body_mass_g": 4000.0, "island": "Biscoe", "sex": "male"}'
```

### Respuesta esperada

```json
{
  "species": "Gentoo"
}
```

### Campos del request

| Campo | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `bill_length_mm` | float | Longitud del pico en mm | 45.0 |
| `bill_depth_mm` | float | Profundidad del pico en mm | 15.0 |
| `flipper_length_mm` | float | Longitud de la aleta en mm | 200.0 |
| `body_mass_g` | float | Masa corporal en gramos | 4000.0 |
| `island` | string | Isla de origen | `Biscoe`, `Dream`, `Torgersen` |
| `sex` | string | Sexo del pingüino | `male`, `female` |

---

## 🎯 Objetivos y conclusiones

### Objetivos

Este taller tuvo como objetivo implementar un pipeline de MLOps completo que demuestre la integración de las siguientes prácticas:

- **Orquestación de workflows** con Apache Airflow, definiendo un DAG con múltiples tareas encadenadas.
- **Separación de responsabilidades** entre bases de datos: una exclusiva para datos de negocio y otra para metadatos del orquestador.
- **Reproducibilidad** del pipeline: cada ejecución del DAG parte desde cero, garantizando resultados consistentes.
- **Desacoplamiento de servicios** mediante Docker Compose, donde cada componente corre en su propio contenedor.
- **Exposición del modelo** a través de una API REST lista para consumirse desde cualquier cliente.

### Conclusiones

- Apache Airflow permite visualizar, monitorear y re-ejecutar cada paso del pipeline de forma independiente, lo que facilita la depuración y el mantenimiento.
- Separar la base de datos de datos de la base de datos de metadatos es una buena práctica que evita conflictos y facilita el escalado independiente de cada componente.
- Contenerizar todos los servicios en un único `docker-compose.yml` simplifica el despliegue y garantiza que el entorno sea reproducible en cualquier máquina.
- El uso de volúmenes compartidos entre Airflow y la API es una solución simple y efectiva para pasar artefactos de ML entre servicios sin necesidad de un registry externo como MLflow.

---

## 🛑 Apagar el stack

```bash
# Detener sin borrar datos
docker-compose down

# Detener y borrar volúmenes (bases de datos y modelos)
docker-compose down -v
```
