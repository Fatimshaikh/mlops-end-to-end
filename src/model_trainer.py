import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelTrainer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, "model.pkl")

        # ✅ CI-SAFE MLflow backend (NO mlruns corruption)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("mlops_end_to_end")

    def train_model(self):
        df = pd.read_csv(self.data_path)

        X = df.drop("churn", axis=1)
        y = df["churn"]

        # ✅ SAFE SPLIT (no stratify for small datasets)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)

        with mlflow.start_run():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # ✅ Modern MLflow logging
            mlflow.sklearn.log_model(model, name="model")

            print("Training completed")
            print(f"Accuracy : {acc}")
            print(f"Precision: {prec}")
            print(f"Recall   : {rec}")
            print(f"F1 Score : {f1}")

        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Model saved at: {self.model_path}")
        return self.model_path


if __name__ == "__main__":
    trainer = ModelTrainer("data/processed/processed_data.csv")
    trainer.train_model()
