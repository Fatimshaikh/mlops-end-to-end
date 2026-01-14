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

        # ✅ MLflow tracking directory (CI-safe, OS-independent)
        mlflow_tracking_dir = os.path.join(os.getcwd(), "mlruns")
        mlflow.set_tracking_uri(f"file:///{mlflow_tracking_dir}")
        mlflow.set_experiment("mlops_end_to_end")

    def train_model(self):
        # Load data
        df = pd.read_csv(self.data_path)

        X = df.drop("churn", axis=1)
        y = df["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(max_iter=1000)

        with mlflow.start_run():
            # Train
            model.fit(X_train, y_train)

            # Predict
            preds = model.predict(X_test)

            # Metrics (CI-safe)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)

            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # ✅ Log model (new MLflow API)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )

            print("Training Metrics:")
            print(f"Accuracy : {acc}")
            print(f"Precision: {prec}")
            print(f"Recall   : {rec}")
            print(f"F1 Score : {f1}")

        # Save model locally for API usage
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Model saved at: {self.model_path}")
        return self.model_path


if __name__ == "__main__":
    trainer = ModelTrainer("data/processed/processed_data.csv")
    trainer.train_model()
