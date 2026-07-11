import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

# Mengaktifkan autologging khusus untuk XGBoost agar hasil lebih detail
mlflow.xgboost.autolog()


def train_model():
    print("Memuat dataset untuk training produksi...")
    df = pd.read_csv('ecommerce_data_preprocessing.csv')
    X = df.drop(columns=['Is_Churn'])
    y = df['Is_Churn']

    # Menghitung ratio untuk scale_pos_weight (penting untuk imbalance data)
    ratio = (y == 0).sum() / (y == 1).sum()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        print("Melatih model XGBoost versi Production...")

        # Gunakan parameter terbaik yang Anda temukan di modelling_tunning.py
        # Sesuaikan nilai di bawah ini dengan hasil GridSearch terbaik Anda
        model = XGBClassifier(
            n_estimators=200,  # Sesuaikan dengan hasil grid search terbaik
            max_depth=10,  # Sesuaikan dengan hasil grid search terbaik
            learning_rate=0.1,  # Sesuaikan dengan hasil grid search terbaik
            scale_pos_weight=ratio,  # Penting: ini membuat model fokus ke Churn
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        # Simpan Run ID ke file teks untuk dibaca oleh GitHub Actions
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Pelatihan selesai! Run ID: {run_id}")


if __name__ == "__main__":
    train_model()