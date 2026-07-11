FROM python:3.12-7

# Install dependencies sistem yang diperlukan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy semua file dari MLProject ke dalam container
COPY . .

# Install MLflow dan library yang dibutuhkan
RUN pip install --no-cache-dir mlflow==2.19.0 pandas scikit-learn

# Jalankan skrip
ENTRYPOINT ["python", "modelling.py"]