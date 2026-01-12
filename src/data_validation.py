 
import pandas as pd
import os

class DataValidation:
    def __init__(self, data_path):
        self.data_path = data_path

        # Define expected schema
        self.schema = {
            "age": "int64",
            "monthly_charges": "int64",
            "tenure": "int64",
            "total_charges": "int64",
            "churn": "int64"
        }

    def validate_schema(self, df):
        for column, dtype in self.schema.items():
            if column not in df.columns:
                raise ValueError(f"Missing column: {column}")

            if df[column].dtype != dtype:
                raise ValueError(f"Invalid data type for {column}")

    def validate_missing_values(self, df):
        if df.isnull().sum().any():
            raise ValueError("Dataset contains missing values")

    def run_validation(self):
        df = pd.read_csv(self.data_path)

        self.validate_schema(df)
        self.validate_missing_values(df)

        print("✅ Data validation passed")


if __name__ == "__main__":
    validator = DataValidation("data/raw/data.csv")
    validator.run_validation()
