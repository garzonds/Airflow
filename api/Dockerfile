FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la API
COPY main.py .

# Exponer puerto
EXPOSE 8000

# Arrancar servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]