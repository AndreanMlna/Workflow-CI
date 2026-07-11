import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Mengaktifkan pencatatan otomatis
mlflow.autolog()

def train_model():
    print("Memuat dataset preprocessing...")
    df = pd.read_csv('ecommerce_data_preprocessing.csv')
    X = df.drop(columns=['Is_Churn'])
    y = df['Is_Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        print("Melatih model RandomForest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Trik Advance: Simpan Run ID ke file teks untuk dibaca oleh GitHub Actions
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Pelatihan selesai! Run ID: {run_id}")

if __name__ == "__main__":
    train_model()