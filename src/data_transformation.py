 
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataTransformation:
    def __init__(self, data_path):
        self.data_path = data_path
        self.processed_data_path = os.path.join("data", "processed")
        self.preprocessor_path = os.path.join("data", "processed", "scaler.pkl")

    def transform_data(self):
        df = pd.read_csv(self.data_path)

        X = df.drop("churn", axis=1)
        y = df["churn"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        os.makedirs(self.processed_data_path, exist_ok=True)

        # Save scaler
        with open(self.preprocessor_path, "wb") as f:
            pickle.dump(scaler, f)

        # Save processed data
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled_df["churn"] = y

        processed_file = os.path.join(self.processed_data_path, "processed_data.csv")
        X_scaled_df.to_csv(processed_file, index=False)

        return processed_file


if __name__ == "__main__":
    transformer = DataTransformation("data/raw/data.csv")
    path = transformer.transform_data()
    print(f"Data transformed and saved at: {path}")
